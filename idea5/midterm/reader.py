"""Step 2 (GPU): Qwen2.5-VL-7B answers every test question from its gold pages under each input condition.

  CUDA_VISIBLE_DEVICES=0 python reader.py --shard 0 --nshards 48   # reads out/test/, cache/pages/
                                                                    # writes out/reader/answers_<shard>.jsonl

The reader always gets the question and the text layer of each gold page; what changes is how much of the page
images it gets and how the visual tokens are chosen. The budget is K = 64, 128 or 256 visual tokens per page (out of
about 1,250), so K x N for a question with N gold pages; the selectors choose across all pages jointly:
  text              text layers only                      full            text layers + all visual tokens
  full_imgfirst     as full, images before the text (the order used in the analysis)
  down_K            text layers + every page downscaled to K tokens
  rand_K, ink_K, comp_K   text layers + K x N tokens of the full-resolution pages chosen before the LLM (token_select.py)
  fastv_K           FastV (Chen et al., ECCV 2024): after layer 1, keep the K x N visual tokens that the last prompt
                    token attends to most (averaged over heads) and drop the rest for every later layer
  svlm_K            SparseVLM (Zhang et al., ICML 2025): text raters are the prompt tokens after the images whose
                    softmax relevance to the visual tokens is above average; after layer 2, visual tokens are ranked
                    by the rater-averaged attention of layer 2 and the top ones are kept; part of the dropped tokens
                    is recycled into a few merged tokens (density-peak clustering). Kept + merged = K x N.
  svlmnr_K          SparseVLM without recycling: the top K x N by the same score (ablation)
In every condition except full_imgfirst each page's text layer comes before its image and the question comes last,
so the visual tokens are computed in the context of the text and SparseVLM's raters are the question tokens, as in
the original method. With one page the prompts are worded exactly as in the analysis; with several, pages are
numbered in reading order and the text layers share a budget of 36,000 characters. Kept tokens keep their original
3D rotary positions (as FastV and SparseVLM+ do), which Qwen2.5-VL's 2D image positions require. Decoding is greedy
with a KV cache whose per-layer lengths may differ after pruning.
Resumable: (qid, condition) rows already written by any shard are skipped.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from conds import INPUT_METHODS, MODEL_METHODS, conditions, parse
from shared import OUT, PAGES, TEST, fit
from token_select import select_input

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
READER = Path(os.environ.get("IDEA5_MT_READER", OUT / "reader"))
MAX_TEXT_CHARS = 12000      # per page, as in the analysis
TOTAL_TEXT_CHARS = 36000    # all pages of a question together (only binds with more than 3 gold pages)
MAX_NEW_TOKENS = 64
FASTV_K = 2                 # FastV's default K: prune before layer 2 using the attention of layer 1
SVLM_LAYER = 2              # SparseVLM's first sparsification layer (its layers are 2, 6, 15)
SVLM_RECYCLE = 0.3          # SparseVLM v1: recycle the top 30% of the dropped tokens ...
SVLM_CLUSTER = 10           # ... into one merged token per 10 of them
FORMAT = {"Int": "an integer", "Float": "a number", "Str": "a short phrase",
          "List": "a Python list of strings, e.g. ['first', 'second']"}


# ---------- prompts (wording as in the analysis) ----------

def question_block(item):
    return (f"Question: {item['question'].strip()}\nAnswer with only {FORMAT[item['answer_format']]}, "
            "no explanation. If you are unsure, give your best guess.")


def page_text(text, n_pages=1):
    cap = min(MAX_TEXT_CHARS, TOTAL_TEXT_CHARS // n_pages)
    return "<<<\n" + ((text or "")[:cap] or "(no text could be extracted from this page)") + "\n>>>"


def conversation(item, texts, images, variant):
    """texts: the text layer of each gold page; images: one image per page, or None for the text-only condition."""
    n, q = len(texts), question_block(item)
    blocks = [page_text(t, n) for t in texts]
    if n == 1:  # the analysis's wording
        if images is None:
            content = [{"type": "text", "text": "Below is the text extracted from one page of a document.\n\n"
                        + blocks[0] + "\n\n" + q}]
        elif variant == "if":
            content = [{"type": "image", "image": images[0]},
                       {"type": "text", "text": "Below are an image of one page of a document and the text "
                        "extracted from the same page.\n\n" + blocks[0] + "\n\n" + q}]
        else:
            content = [{"type": "text", "text": "Below is the text extracted from one page of a document, followed "
                        "by an image of the same page.\n\n" + blocks[0] + "\n"},
                       {"type": "image", "image": images[0]},
                       {"type": "text", "text": "\n" + q}]
        return [{"role": "user", "content": content}]
    numbered = "\n\n".join(f"Page {i}:\n{b}" for i, b in enumerate(blocks, 1))
    if images is None:
        content = [{"type": "text", "text": f"Below is the text extracted from {n} pages of a document.\n\n"
                    + numbered + "\n\n" + q}]
    elif variant == "if":
        content = [{"type": "image", "image": im} for im in images] + [
            {"type": "text", "text": f"Below are images of {n} pages of a document and the text extracted from the "
             "same pages, in the same order.\n\n" + numbered + "\n\n" + q}]
    else:
        content = [{"type": "text", "text": f"Below are {n} pages of a document. For each page, the text extracted "
                    "from it is followed by an image of the page.\n"}]
        for i, (b, im) in enumerate(zip(blocks, images), 1):
            content += [{"type": "text", "text": f"\nPage {i}:\n{b}\n"}, {"type": "image", "image": im}]
        content.append({"type": "text", "text": "\n" + q})
    return [{"role": "user", "content": content}]


# ---------- SparseVLM token recycling (density-peak clustering, ported from SparseVLMs/llava/.../utils.py) ----------

def cluster_and_merge(x, cluster_num, generator=None):
    """x: [1, N, C] -> merged tokens [1, cluster_num, C] and the index of each cluster's centre in x."""
    _, n, c = x.shape
    x = x.float()
    dist = torch.cdist(x, x) / (c ** 0.5)
    dist_nearest, _ = torch.topk(dist, k=min(cluster_num, n), dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, generator=generator, device=density.device) * 1e-6
    mask = (density[:, None, :] > density[:, :, None]).float()
    dist_max = dist.flatten(1).max(dim=-1)[0][:, None, None]
    dist_parent, _ = (dist * mask + dist_max * (1 - mask)).min(dim=-1)
    _, centres = torch.topk(dist_parent * density, k=cluster_num, dim=-1)         # [1, cluster_num]
    assign = dist[0, centres[0]].argmin(dim=0)                                     # [N]
    assign[centres[0]] = torch.arange(cluster_num, device=x.device)
    merged = torch.zeros(cluster_num, c, device=x.device).index_add_(0, assign, x[0])
    counts = torch.zeros(cluster_num, device=x.device).index_add_(0, assign, torch.ones(n, device=x.device))
    return (merged / counts[:, None])[None], centres[0]


def svlm_split(n_vis, k):
    """(selected, recycled candidates, merged) with selected + merged <= k, following SparseVLM v1's recycling rule."""
    for sel in range(min(k, n_vis), 0, -1):
        dropped = n_vis - sel
        cand = int(dropped * SVLM_RECYCLE) + 1 if dropped > 0 else 0
        merged = int(cand / SVLM_CLUSTER) + 1 if cand > 0 else 0
        if sel + merged <= k:
            return sel, cand, merged
    return 1, 0, 0


# ---------- the reader ----------

class Reader:
    def __init__(self, model=None, processor=None, model_name=MODEL, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.proc = processor or AutoProcessor.from_pretrained(
            model_name, size={"shortest_edge": 4 * 28 * 28, "longest_edge": 1400 * 28 * 28})  # images pre-sized
        self.model = model or Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.device = self.model.device
        self.vl = self.model.model
        self.lm = self.vl.language_model
        self.image_token_id = self.model.config.image_token_id
        self.merge = self.model.config.vision_config.spatial_merge_size
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.sdpa = self.lm.config._attn_implementation == "sdpa"  # prefill uses SDPA's causal mode, no mask
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb, repeat_kv
        self._rope, self._repeat_kv = apply_rotary_pos_emb, repeat_kv

    # -- inputs --

    @torch.inference_mode()
    def prepare(self, conv):
        """Prompt -> input embeddings with the image features in place, 3D positions, the visual tokens."""
        enc = self.proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self.device)
        ids = enc["input_ids"]
        emb = self.vl.get_input_embeddings()(ids)
        vis = ids[0] == self.image_token_id
        prep = {"ids": ids, "vis_seq": vis.nonzero()[:, 0], "grids": []}
        if "pixel_values" in enc:
            feats = self.vl.get_image_features(enc["pixel_values"], image_grid_thw=enc["image_grid_thw"],
                                               return_dict=True).pooler_output
            emb = emb.clone()
            emb[0, vis] = torch.cat(feats, dim=0).to(emb.dtype)
            pos, _ = self.vl.get_rope_index(ids, mm_token_type_ids=vis[None].int(),
                                            image_grid_thw=enc["image_grid_thw"])
            prep["grids"] = [[gh // self.merge, gw // self.merge] for _, gh, gw in enc["image_grid_thw"].tolist()]
        else:
            pos = torch.arange(ids.shape[1], device=self.device).view(1, 1, -1).expand(3, 1, -1)
        prep["emb"], prep["pos"] = emb, pos
        return prep

    # -- helpers --

    def _causal(self, n, dtype):
        if self.sdpa:
            return None  # q_len == k_len in every prefill call, so SDPA's is_causal is exact
        m = torch.full((n, n), torch.finfo(dtype).min, device=self.device, dtype=dtype).triu(1)
        return m[None, None]

    @staticmethod
    def _rebuild(h, pos, seq, vis_cur, keep_local, extra_h=None, extra_pos=None):
        """Drop the visual tokens vis_cur not in keep_local; insert extra tokens right after the last visual token."""
        keep = torch.ones(h.shape[1], dtype=torch.bool, device=h.device)
        keep[vis_cur] = False
        keep[vis_cur[keep_local]] = True
        idx = torch.nonzero(keep)[:, 0]
        if extra_h is None:
            return h[:, idx], pos[:, :, idx], seq[idx]
        cut = int((idx <= vis_cur[-1]).sum())
        a, b = idx[:cut], idx[cut:]
        marker = torch.full((extra_h.shape[1],), -1, device=h.device, dtype=seq.dtype)
        return (torch.cat([h[:, a], extra_h.to(h.dtype), h[:, b]], 1),
                torch.cat([pos[:, :, a], extra_pos, pos[:, :, b]], 2), torch.cat([seq[a], marker, seq[b]], 0))

    def _attention_rows(self, layer, h, cos, sin, rows):
        """Head-averaged attention probabilities of `layer` from the tokens `rows` to every token (causal)."""
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        n, hd = x.shape[1], attn.head_dim
        q = attn.q_proj(x[:, rows]).view(1, len(rows), -1, hd).transpose(1, 2)
        k = attn.k_proj(x).view(1, n, -1, hd).transpose(1, 2)
        q, _ = self._rope(q, q, cos[:, rows], sin[:, rows])
        k, _ = self._rope(k, k, cos, sin)
        k = self._repeat_kv(k, attn.num_key_value_groups)
        s = torch.matmul(q.float(), k.float().transpose(2, 3)) * attn.scaling          # [1, H, R, n]
        future = torch.arange(n, device=h.device)[None, :] > rows[:, None]
        s = s.masked_fill(future[None, None], float("-inf"))
        return s.softmax(dim=-1).mean(dim=1)[0]                                       # [R, n]

    # -- answering --

    @torch.inference_mode()
    def answer(self, prep, method=None, k=None, keep=None, max_new_tokens=MAX_NEW_TOKENS, qid=0):
        """Greedy answer. method None: all tokens; 'input' with keep (indices into the visual tokens of all pages,
        in sequence order); 'fastv', 'svlm' or 'svlmnr' with k (total visual tokens kept)."""
        h, pos = prep["emb"], prep["pos"]
        seq = torch.arange(h.shape[1], device=self.device)
        vis_seq = prep["vis_seq"]
        n_vis = int(vis_seq.numel())
        n_layers = len(self.lm.layers)
        info = {"input_tokens": int(h.shape[1]), "n_vis": n_vis, "grids": prep["grids"], "n_kept": n_vis,
                "n_merged": 0, "kept": None, "vis_per_layer": float(n_vis)}
        next_pos = int(pos.max()) + 1
        prune_at, rows_fn = None, None
        if method == "input" and n_vis:
            keep_t = torch.as_tensor(np.asarray(keep), device=self.device, dtype=torch.long)
            h, pos, seq = self._rebuild(h, pos, seq, vis_seq, keep_t)
            info.update(n_kept=int(keep_t.numel()), kept=[int(i) for i in keep_t], vis_per_layer=float(keep_t.numel()))
        elif method in MODEL_METHODS and n_vis and k <= n_vis:  # k == n_vis keeps every token (tested on CPU)
            prune_at = FASTV_K - 1 if method == "fastv" else SVLM_LAYER
            if method in ("svlm", "svlmnr"):  # SparseVLM 3.2: text raters (after the last image), input embeddings
                v = h[0, vis_seq].float()
                t_idx = torch.arange(int(vis_seq[-1]) + 1, h.shape[1], device=self.device)
                rel = (v @ h[0, t_idx].float().T).softmax(dim=1).mean(dim=0)
                raters = t_idx[rel > rel.mean()]
                rows_fn = lambda s: torch.nonzero(torch.isin(s, raters))[:, 0]
            else:                 # FastV: the last token of the prompt
                rows_fn = lambda s: torch.tensor([s.numel() - 1], device=self.device)

        cache = self._new_cache()
        cos, sin = self.lm.rotary_emb(h, pos)
        mask = self._causal(h.shape[1], h.dtype)
        for i, layer in enumerate(self.lm.layers):
            if i == prune_at:
                vis_cur = torch.nonzero(torch.isin(seq, vis_seq))[:, 0]
                p = self._attention_rows(layer, h, cos, sin, rows_fn(seq))
                score = p[:, vis_cur].mean(dim=0)
            h = layer(h, attention_mask=mask, position_embeddings=(cos, sin), past_key_values=cache, use_cache=True)
            if i == prune_at:
                if method in ("fastv", "svlmnr"):
                    keep_local = torch.topk(score, k).indices.sort().values
                    h, pos, seq = self._rebuild(h, pos, seq, vis_cur, keep_local)
                    n_merged = 0
                else:
                    sel, cand, n_merged = svlm_split(n_vis, k)
                    keep_local = torch.topk(score, sel).indices.sort().values
                    extra_h = extra_pos = None
                    if n_merged:
                        dropped = torch.ones(n_vis, dtype=torch.bool, device=self.device)
                        dropped[keep_local] = False
                        dropped = torch.nonzero(dropped)[:, 0]
                        cand_local = dropped[torch.topk(score[dropped], min(cand, dropped.numel())).indices]
                        g = torch.Generator(device=self.device).manual_seed(int(qid))
                        extra_h, centres = cluster_and_merge(h[:, vis_cur[cand_local]], n_merged, generator=g)
                        extra_pos = pos[:, :, vis_cur[cand_local[centres]]]
                    h, pos, seq = self._rebuild(h, pos, seq, vis_cur, keep_local, extra_h, extra_pos)
                cos, sin = self.lm.rotary_emb(h, pos)
                mask = self._causal(h.shape[1], h.dtype)
                n_after = int(keep_local.numel()) + n_merged
                info.update(n_kept=int(keep_local.numel()), n_merged=n_merged, kept=[int(j) for j in keep_local],
                            vis_per_layer=((i + 1) * n_vis + (n_layers - i - 1) * n_after) / n_layers)
        logits = self.model.lm_head(self.lm.norm(h[:, -1:]))
        info["seq_after"] = int(h.shape[1])

        out, tok = [], int(logits[0, -1].argmax())
        embed = self.vl.get_input_embeddings()
        while tok not in self.eos and len(out) < max_new_tokens:
            out.append(tok)
            x = embed(torch.tensor([[tok]], device=self.device))
            p = torch.full((3, 1, 1), next_pos, device=self.device, dtype=pos.dtype)
            next_pos += 1
            c, s = self.lm.rotary_emb(x, p)
            for layer in self.lm.layers:
                x = layer(x, attention_mask=None, position_embeddings=(c, s), past_key_values=cache, use_cache=True)
            tok = int(self.model.lm_head(self.lm.norm(x))[0, -1].argmax())
        info["n_out"] = len(out)
        return self.proc.tokenizer.decode(out, skip_special_tokens=True).strip(), out, info

    def _new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.lm.config)


# ---------- run ----------

def load_items():
    return [json.loads(l) for l in (TEST / "items.jsonl").open()]


def load_pages():
    return {p["key"]: p for p in map(json.loads, (TEST / "pages.jsonl").open())}


def shard_items(items, shard, nshards):
    """Round-robin over items sorted by gold-page count, so every shard gets a similar share of the long questions."""
    order = sorted(items, key=lambda it: (-it["n_gold"], it["qid"]))
    return [it for i, it in enumerate(order) if i % nshards == shard]


def load_done():
    done = set()
    for p in READER.glob("answers_*.jsonl"):
        for line in p.open():
            r = json.loads(line)
            done.add((r["qid"], r["cond"]))
    return done


def main(shard, nshards, conds, limit):
    items = shard_items(load_items(), shard, nshards)[: limit or None]
    pages = load_pages()
    tok = np.load(TEST / "tokinfo.npz")
    READER.mkdir(parents=True, exist_ok=True)
    done = load_done()
    reader = Reader()
    out_file = READER / f"answers_{shard}.jsonl"
    print(f"shard {shard}/{nshards}: {len(items)} items, {len(conds)} conditions, {len(done)} rows done", flush=True)
    with out_file.open("a") as f:
        for n_item, it in enumerate(items, 1):
            todo = [c for c in conds if (it["qid"], c) not in done]
            if not todo:
                continue
            keys = it["page_keys"]
            imgs = [Image.open(PAGES / f"{key}.png").convert("RGB") for key in keys]
            texts = [pages[key]["text"] for key in keys]
            ink = np.concatenate([tok[f"{key}_ink"].ravel() for key in keys])   # page after page, row-major
            txt = np.concatenate([tok[f"{key}_txt"].ravel() for key in keys])
            grids = [list(tok[f"{key}_ink"].shape) for key in keys]
            preps = {}
            for cond in todo:
                variant, method, k = parse(cond)
                if variant not in preps:
                    images = (None if variant == "text" else [fit(im, k) for im in imgs] if variant.startswith("down")
                              else imgs)
                    preps[variant] = reader.prepare(conversation(it, texts, images, variant))
                    if variant in ("tf", "if"):
                        assert preps[variant]["grids"] == grids, (it["qid"], preps[variant]["grids"], grids)
                prep = preps[variant]
                total = k * len(keys) if k else None
                t0 = time.time()
                if method in INPUT_METHODS:
                    keep = select_input(method, total, ink, txt, it["qid"])
                    pred, _, info = reader.answer(prep, method="input", keep=keep, qid=it["qid"])
                elif method in MODEL_METHODS:
                    pred, _, info = reader.answer(prep, method=method, k=total, qid=it["qid"])
                else:
                    pred, _, info = reader.answer(prep, qid=it["qid"])
                f.write(json.dumps({"qid": it["qid"], "cond": cond, "pred": pred, **info,
                                    "sec": round(time.time() - t0, 3)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n_item}/{len(items)}] qid {it['qid']}: {len(todo)} conditions", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--conds", default="", help="comma-separated conditions (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="first N items of the shard only (smoke test)")
    a = ap.parse_args()
    main(a.shard, a.nshards, a.conds.split(",") if a.conds else conditions(), a.limit)
