"""CPU tests of reader.py with a tiny random Qwen2.5-VL (real processor and config, random small weights).

  python tests/test_reader_cpu.py     # needs out/test/ from build_test.py

Checks, on real test pages, for a single-page and a multi-page question:
  1. With all visual tokens, the layer-by-layer prefill and the cached greedy decoding of Reader.answer give the same
     tokens as the model's own generate(), with and without images, in both page orders.
  2. _attention_rows equals the model's own (eager) attention probabilities, averaged over heads.
  3. Pruning that keeps every token (input selection of all tokens, FastV and SparseVLM with K = all) changes nothing.
  4. Every condition of the experiment runs, keeps at most K x N visual tokens (kept + merged), and keeps indices that
     are sorted, unique and inside the pages' grids; the residual selector picks the top of its score.
  5. reader.main writes one row per (question, condition) and resumes without duplicates.
Values are random; nothing here is a result.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conds import INPUT_METHODS, conditions, parse  # noqa: E402
from reader import Reader, conversation, load_pages, svlm_split  # noqa: E402
from shared import PAGES, TEST, fit  # noqa: E402
from token_select import select_input  # noqa: E402

REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json"]
ITEMS = [json.loads(l) for l in (TEST / "items.jsonl").open()]
PAGE_INFO = load_pages()
TOK = np.load(TEST / "tokinfo.npz")


def tiny(attn="sdpa"):
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    path = snapshot_download(REPO, allow_patterns=FILES)
    cfg = AutoConfig.from_pretrained(path)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 4
    t.num_attention_heads, t.num_key_value_heads, t.layer_types = 4, 2, ["full_attention"] * 4
    t.rope_parameters["mrope_section"] = [4, 6, 6]          # head_dim 32 -> 16 rotary frequencies
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 64, 128, 4
    v.out_hidden_size, v.fullatt_block_indexes = 128, [1]
    cfg.initializer_range = t.initializer_range = 0.2      # peaked logits, so greedy decoding is not a tie-break test
    cfg._attn_implementation = attn
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
    model.generation_config.eos_token_id = [0]              # random weights: decode the full length
    proc = AutoProcessor.from_pretrained(path, size={"shortest_edge": 4 * 28 * 28, "longest_edge": 1400 * 28 * 28})
    return model, proc


def pick(n_gold):
    """The first test question with exactly n_gold gold pages, with its page images, texts and token info."""
    it = next(x for x in ITEMS if x["n_gold"] == n_gold)
    keys = it["page_keys"]
    imgs = [Image.open(PAGES / f"{k}.png").convert("RGB") for k in keys]
    texts = [PAGE_INFO[k]["text"] for k in keys]
    ink = np.concatenate([TOK[f"{k}_ink"].ravel() for k in keys])
    txt = np.concatenate([TOK[f"{k}_txt"].ravel() for k in keys])
    return it, imgs, texts, ink, txt


def hf_tokens(model, proc, conv, n_new):
    enc = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt")
    with torch.inference_mode():
        gen = model.generate(**enc, max_new_tokens=n_new, do_sample=False)
    return gen[0, enc["input_ids"].shape[1]:].tolist()


def test_equivalence(model, proc, reader):
    for n_gold in (1, 2):
        it, imgs, texts, _, _ = pick(n_gold)
        for variant, images in [("text", None), ("tf", imgs), ("if", imgs)]:
            conv = conversation(it, texts, images, variant)
            ref = hf_tokens(model, proc, conv, 6)
            prep = reader.prepare(conv)
            _, tokens, info = reader.answer(prep, max_new_tokens=6)
            assert tokens == ref, (n_gold, variant, tokens, ref)
            assert len(prep["grids"]) == (0 if images is None else n_gold)
        print(f"ok 1 equivalence with generate(), {n_gold} page(s), text / text-first / image-first "
              f"({info['n_vis']} visual tokens)")


def test_attention_rows(proc):
    model, _ = tiny(attn="eager")
    reader = Reader(model=model, processor=proc)
    for n_gold in (1, 2):
        it, imgs, texts, _, _ = pick(n_gold)
        conv = conversation(it, texts, imgs, "tf")
        enc = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt")
        with torch.inference_mode():
            out = model(**enc, output_attentions=True, output_hidden_states=True)
        prep = reader.prepare(conv)
        cos, sin = reader.lm.rotary_emb(prep["emb"], prep["pos"])
        rows = torch.tensor([prep["emb"].shape[1] - 1, int(prep["vis_seq"][-1]) + 5])
        for i in (1, 2):
            mine = reader._attention_rows(reader.lm.layers[i], out.hidden_states[i], cos, sin, rows)
            ref = out.attentions[i][0, :, rows].float().mean(dim=0)
            wrong = out.attentions[i + 1][0, :, rows].float().mean(dim=0)
            err = (mine - ref).abs().max().item()
            assert err < 1e-4 and (mine - wrong).abs().max().item() > 1e-2, (n_gold, i, err)
        print(f"ok 2 attention rows match eager attention, {n_gold} page(s) (max abs difference {err:.1e})")


def test_keep_all(reader):
    for n_gold in (1, 2):
        it, imgs, texts, ink, _ = pick(n_gold)
        prep = reader.prepare(conversation(it, texts, imgs, "tf"))
        n = prep["vis_seq"].numel()
        assert n == ink.size, (n, ink.size)
        _, full, _ = reader.answer(prep, max_new_tokens=6)
        _, inp, _ = reader.answer(prep, method="input", keep=np.arange(n), max_new_tokens=6)
        _, fv, i_fv = reader.answer(prep, method="fastv", k=n, max_new_tokens=6)
        _, sv, i_sv = reader.answer(prep, method="svlm", k=n, max_new_tokens=6)
        _, nr, _ = reader.answer(prep, method="svlmnr", k=n, max_new_tokens=6)
        assert full == inp == fv == sv == nr, (full, inp, fv, sv, nr)
        assert i_fv["n_kept"] == i_sv["n_kept"] == n and i_sv["n_merged"] == 0
        print(f"ok 3 keep-all pruning is a no-op, {n_gold} page(s) ({n} visual tokens)")


def test_conditions(reader):
    for n_gold in (1, 2):
        it, imgs, texts, ink, txt = pick(n_gold)
        preps = {}
        for cond in conditions():
            variant, method, k = parse(cond)
            if variant not in preps:
                images = None if variant == "text" else [fit(im, k) for im in imgs] if variant.startswith("down") else imgs
                preps[variant] = reader.prepare(conversation(it, texts, images, variant))
            prep = preps[variant]
            total = k * n_gold if k else None
            if method in INPUT_METHODS:
                keep = select_input(method, total, ink, txt, it["qid"])
                _, _, info = reader.answer(prep, method="input", keep=keep, max_new_tokens=3)
            elif method:
                _, _, info = reader.answer(prep, method=method, k=total, max_new_tokens=3, qid=it["qid"])
            else:
                _, _, info = reader.answer(prep, max_new_tokens=3)
            if variant.startswith("down"):
                assert info["n_vis"] <= total and len(info["grids"]) == n_gold, (cond, info["n_vis"])
            if method:
                kept = info["kept"]
                assert info["n_kept"] + info["n_merged"] <= total, (cond, info)
                assert kept == sorted(set(kept)) and 0 <= kept[0] and kept[-1] < ink.size, cond
                assert info["seq_after"] == info["input_tokens"] - info["n_vis"] + info["n_kept"] + info["n_merged"]
            if method == "comp":
                s = (ink.astype(np.float32) * (1 - txt.astype(np.float32)))
                assert s[kept].min() >= np.sort(s)[-total] - 1e-6, cond
        n = preps["tf"]["vis_seq"].numel()
        print(f"ok 4 all {len(conditions())} conditions run, {n_gold} page(s), qid {it['qid']} ({n} visual tokens; "
              f"SparseVLM K=64 split {svlm_split(n, 64 * n_gold)})")


def test_main_resume(model, proc):
    import reader as R
    out = Path(__file__).resolve().parent / "out_SYNTHETIC_main"
    shutil.rmtree(out, ignore_errors=True)
    R.READER = out
    R.Reader = lambda: Reader(model=model, processor=proc)
    conds = ["text", "full", "down_64", "comp_64", "fastv_64", "svlm_64"]
    nshards = len(ITEMS) // 2                       # shard 0 then holds the two longest questions
    R.main(0, nshards, conds, limit=2)
    rows = [json.loads(l) for l in (out / "answers_0.jsonl").open()]
    R.main(0, nshards, conds, limit=2)
    again = [json.loads(l) for l in (out / "answers_0.jsonl").open()]
    assert len(rows) == len(again) == 2 * len(conds), (len(rows), len(again))
    shards = [R.shard_items(ITEMS, s, 48) for s in range(48)]
    assert sorted(q["qid"] for s in shards for q in s) == sorted(q["qid"] for q in ITEMS)
    shutil.rmtree(out)
    print(f"ok 5 reader.main wrote {len(rows)} rows (gold pages {[r['n_vis'] for r in rows if r['cond'] == 'full']} "
          f"visual tokens) and resumed without duplicates; 48 shards cover every question once")


if __name__ == "__main__":
    model, proc = tiny()
    reader = Reader(model=model, processor=proc)
    test_equivalence(model, proc, reader)
    test_attention_rows(proc)
    test_keep_all(reader)
    test_conditions(reader)
    test_main_resume(model, proc)
    print("all reader tests passed")
