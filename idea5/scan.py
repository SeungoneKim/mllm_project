"""Step 1: ColQwen2.5 within-document retrieval for every eligible question of MMLongBench-Doc.

  python scan.py render --workers 8                                  # CPU: pages missing from both caches
  CUDA_VISIBLE_DEVICES=0 python scan.py embed --shard 0 --nshards 2   # GPU: page embeddings, one process per GPU
  CUDA_VISIBLE_DEVICES=1 python scan.py embed --shard 1 --nshards 2
  CUDA_VISIBLE_DEVICES=0 python scan.py score                         # GPU: question scores + page-to-page similarity

Eligible questions (common.eligible): answerable, at least one evidence page, every evidence page inside the PDF.
Writes out/scan/scores.jsonl ({qid, doc_id, scores}) and out/scan/pagesim/<doc_stem>.npy ([n, n] float32).
Every step is resumable. Add --limit-docs N to any step for a smoke test.

Retriever, page rendering and scoring are the same as idea4/retrieve.py, so André's cached embeddings are reused
and his question-as-text ranks are reproduced (attribute.py checks this).

Page-to-page similarity: sim(i, j) = mean over the tokens of page i of the max dot product with the tokens of
page j (late interaction, the same geometry as retrieval), symmetrised as (sim(i, j) + sim(j, i)) / 2.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import (COLQWEN, EMB_CACHE, MAX_VISUAL_TOKENS, SCAN, doc_stem, eligible, emb_path, load_qa,
                    page_image_path, render_doc)

BATCH = 8
DEVICE = os.environ.get("IDEA5_DEVICE", "cuda")  # "cpu" only for tests/test_scan_cpu.py


def docs_and_questions(limit_docs=0):
    qa = eligible(load_qa())
    docs = sorted(qa["doc_id"].unique())
    if limit_docs:
        docs = docs[:limit_docs]
        qa = qa[qa["doc_id"].isin(docs)]
    return docs, qa


# ---------- render (CPU) ----------

def render(workers, limit_docs):
    docs, _ = docs_and_questions(limit_docs)
    with ProcessPoolExecutor(workers) as ex:
        for d, n in zip(docs, ex.map(render_doc, docs)):
            print(f"{d}: {n} pages written", flush=True)


# ---------- model (same as idea4/retrieve.py) ----------

def load_model():
    import torch
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    model = ColQwen2_5.from_pretrained(COLQWEN, dtype=torch.bfloat16, device_map="cuda",
                                       attn_implementation="sdpa").eval()
    proc = ColQwen2_5_Processor.from_pretrained(COLQWEN, max_num_visual_tokens=MAX_VISUAL_TOKENS)
    return model, proc


def embed_images(model, proc, images):
    import torch
    out = []
    for k in range(0, len(images), BATCH):
        batch = proc.process_images(images[k:k + BATCH]).to(model.device)
        with torch.inference_mode():
            e = model(**batch)
        mask = batch["attention_mask"].bool()
        out += [e[i][mask[i]].to(torch.float16).cpu() for i in range(e.shape[0])]
    return out


def embed_texts(model, proc, texts):
    import torch
    out = []
    for k in range(0, len(texts), BATCH):
        batch = proc.process_queries(texts[k:k + BATCH]).to(model.device)
        with torch.inference_mode():
            e = model(**batch)
        mask = batch["attention_mask"].bool()
        out += [e[i][mask[i]].to(torch.float16).cpu() for i in range(e.shape[0])]
    return out


# ---------- embed (GPU) ----------

def embed(shard, nshards, limit_docs):
    import torch
    from PIL import Image
    docs, qa = docs_and_questions(limit_docs)
    n_pages = qa.groupby("doc_id")["n_pages"].first()
    todo = [d for i, d in enumerate(docs) if i % nshards == shard and not emb_path(d).exists()]
    print(f"{len(docs)} documents, {len(todo)} to embed in this shard", flush=True)
    if not todo:
        return
    model, proc = load_model()
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    for d in todo:
        n = int(n_pages[d])
        embs = []
        for k in range(0, n, BATCH):  # load images one batch at a time; long documents have 400+ pages
            imgs = [Image.open(page_image_path(d, p)).convert("RGB") for p in range(k + 1, min(n, k + BATCH) + 1)]
            embs += embed_images(model, proc, imgs)
        tmp = EMB_CACHE / f"{doc_stem(d)}.pt.tmp"
        torch.save(embs, tmp)  # one float16 tensor per page, [n_tokens, 128]
        tmp.rename(EMB_CACHE / f"{doc_stem(d)}.pt")
        print(d, len(embs), flush=True)


# ---------- score (GPU) ----------

def pad(pages):
    """List of [T_i, 128] tensors -> padded [n, T, 128] float32 on the GPU and a [n, T] mask."""
    import torch
    n, t = len(pages), max(p.shape[0] for p in pages)
    P = torch.zeros(n, t, pages[0].shape[1], device=DEVICE)
    mask = torch.zeros(n, t, dtype=torch.bool, device=DEVICE)
    for i, p in enumerate(pages):
        P[i, :p.shape[0]] = p.to(DEVICE).float()
        mask[i, :p.shape[0]] = True
    return P, mask


def maxsim_all(q, P, mask):
    """Late-interaction score of one query against every page: sum over query tokens of the max over page tokens."""
    import torch
    s = torch.einsum("qd,ntd->nqt", q.to(DEVICE).float(), P)
    s = s.masked_fill(~mask[:, None, :], float("-inf"))
    return s.max(dim=2).values.sum(dim=1).tolist()


def page_sim(P, mask, chunk=32):
    import torch
    n = P.shape[0]
    lens = mask.sum(dim=1).tolist()
    out = torch.zeros(n, n)
    for i in range(n):
        pi = P[i, :lens[i]]
        for j0 in range(0, n, chunk):
            s = torch.einsum("td,csd->cts", pi, P[j0:j0 + chunk])
            s = s.masked_fill(~mask[j0:j0 + chunk, None, :], float("-inf"))
            out[i, j0:j0 + chunk] = s.max(dim=2).values.mean(dim=1).cpu()
    return ((out + out.T) / 2).numpy().astype(np.float32)


def score(limit_docs):
    import torch
    docs, qa = docs_and_questions(limit_docs)
    missing = [d for d in docs if not emb_path(d).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} documents have no embeddings yet (run embed first), e.g. {missing[:3]}")
    out_file = SCAN / "scores.jsonl"
    sim_dir = SCAN / "pagesim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in out_file.open()} if out_file.exists() else set()
    model, proc = load_model()
    with out_file.open("a") as f:
        for d in docs:
            qs = qa[(qa["doc_id"] == d) & ~qa["qid"].isin(done)]
            sim_path = sim_dir / f"{doc_stem(d)}.npy"
            if qs.empty and sim_path.exists():
                continue
            pages = torch.load(emb_path(d))
            assert len(pages) == int(qa.loc[qa["doc_id"] == d, "n_pages"].iloc[0]), (d, len(pages))
            P, mask = pad(pages)
            if len(qs):
                for qid, q in zip(qs["qid"], embed_texts(model, proc, list(qs["question"]))):
                    f.write(json.dumps({"qid": int(qid), "doc_id": d, "scores": maxsim_all(q, P, mask)}) + "\n")
                f.flush()
            if not sim_path.exists():
                np.save(sim_path, page_sim(P, mask))
            print(f"{d}: {len(pages)} pages, {len(qs)} questions", flush=True)
            del P, mask
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "embed", "score"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit-docs", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "render":
        render(args.workers, args.limit_docs)
    elif args.cmd == "embed":
        embed(args.shard, args.nshards, args.limit_docs)
    else:
        score(args.limit_docs)
