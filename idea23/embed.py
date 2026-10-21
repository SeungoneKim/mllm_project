"""Step 2 (GPU): ColQwen2.5 page embeddings, and page scores for every question and every decomposed need.

  python embed.py pages     # embeds every page of the test documents -> cache/colqwen/<doc>.pt (resumable)
  python embed.py score     # questions + needs of out/decomp.jsonl vs all pages of their document -> out/scores.pkl

out/scores.pkl holds
  q       {qid: float32[N]}       ColQwen2.5 MaxSim score of the question against each of the N pages
  need    {qid: float32[m, N]}    the same for each of the m decomposed needs
  pooled  {doc_id: float32[N, D]} mean-pooled, L2-normalised page embeddings (page-page similarity for MMR)
Pages are rendered as in idea4/idea5 (144 dpi) and resized by the processor to at most 768 visual tokens (as idea4).
"""
import sys

import numpy as np
import pandas as pd
import pymupdf
import torch

from common import BAD_DOCS, CACHE, DOC_DIR, OUT, doc_stem, load_qa, read_jsonl, render, test_questions

MODEL = "vidore/colqwen2.5-v0.2"
MAX_VISUAL_TOKENS = 768
BATCH = 8
EMB = CACHE / "colqwen"


def load_model():
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    model = ColQwen2_5.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                       attn_implementation="sdpa").eval()
    proc = ColQwen2_5_Processor.from_pretrained(MODEL, max_num_visual_tokens=MAX_VISUAL_TOKENS)
    return model, proc


def run(model, batch):
    with torch.inference_mode():
        e = model(**batch)
    mask = batch["attention_mask"].bool()
    return [e[i][mask[i]].to(torch.float16).cpu() for i in range(e.shape[0])]


def embed_doc(model, proc, doc_id):
    out = []
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        for k in range(0, doc.page_count, BATCH):
            imgs = [render(doc[i]) for i in range(k, min(k + BATCH, doc.page_count))]
            out += run(model, proc.process_images(imgs).to(model.device))
    return out


def embed_texts(model, proc, texts):
    out = []
    for k in range(0, len(texts), BATCH):
        out += run(model, proc.process_queries(texts[k:k + BATCH]).to(model.device))
    return out


def pages():
    docs = sorted(set(test_questions(load_qa()).doc_id) - BAD_DOCS)
    EMB.mkdir(parents=True, exist_ok=True)
    model, proc = load_model()
    for n, d in enumerate(docs, 1):
        f = EMB / f"{doc_stem(d)}.pt"
        if f.exists():
            continue
        emb = embed_doc(model, proc, d)
        torch.save(emb, f)
        print(f"[{n}/{len(docs)}] {d}: {len(emb)} pages, {sum(len(e) for e in emb) / len(emb):.0f} vectors/page",
              flush=True)


def score():
    qs = test_questions(load_qa())
    decomp = {r["qid"]: r["needs"] for r in read_jsonl(OUT / "decomp.jsonl")}
    missing = set(qs.qid) - set(decomp)
    if missing:
        print(f"warning: {len(missing)} questions have no decomposition yet; their needs are left out", flush=True)
    model, proc = load_model()
    res = {"q": {}, "need": {}, "pooled": {}}
    for n, (d, g) in enumerate(qs.groupby("doc_id"), 1):
        emb = torch.load(EMB / f"{doc_stem(d)}.pt")
        pooled = torch.stack([torch.nn.functional.normalize(e.float().mean(0), dim=0) for e in emb])
        res["pooled"][d] = pooled.numpy()
        texts, keys = [], []
        for r in g.itertuples():
            texts.append(r.question); keys.append((r.qid, None))
            for j, need in enumerate(decomp.get(r.qid, [])):
                texts.append(need); keys.append((r.qid, j))
        s = proc.score_multi_vector(embed_texts(model, proc, texts), emb, device=model.device).float().numpy()
        for (qid, j), row in zip(keys, s):
            if j is None:
                res["q"][qid] = row
        for r in g.itertuples():
            rows = [row for (qid, j), row in zip(keys, s) if qid == r.qid and j is not None]
            if rows:
                res["need"][r.qid] = np.stack(rows)
        print(f"[{n}/{qs.doc_id.nunique()}] {d}: {len(g)} questions, {len(texts)} queries", flush=True)
    pd.to_pickle(res, OUT / "scores.pkl")
    print("wrote", OUT / "scores.pkl", flush=True)


if __name__ == "__main__":
    {"pages": pages, "score": score}[sys.argv[1]]()
