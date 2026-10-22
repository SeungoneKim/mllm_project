"""Step 2 (GPU): ColQwen2.5 page embeddings, and page scores for every question and every decomposed need.

  python embed.py pages     # embeds every page of the test documents -> cache/colqwen/<doc>.pt (resumable)
  python embed.py score     # questions + needs of out/decomp.jsonl vs all pages of their document -> out/scores.pkl
  python embed.py elements [shard nshards]   # v2: embeds every element of segment.py -> cache/colqwen_elements/
  python embed.py score_elements             # v2: questions + needs vs all elements -> out/element_scores.pkl

out/scores.pkl holds
  q       {qid: float32[N]}       ColQwen2.5 MaxSim score of the question against each of the N pages
  need    {qid: float32[m, N]}    the same for each of the m decomposed needs
  pooled  {doc_id: float32[N, D]} mean-pooled, L2-normalised page embeddings (page-page similarity for MMR)
Pages are rendered as in idea4/idea5 (144 dpi) and resized by the processor to at most 768 visual tokens (as idea4).
An element is embedded from its crop of the same rendering (common.crop), with the same processor.
out/element_scores.pkl holds q {qid: float32[E]} and need {qid: float32[m, E]}, E elements of the document in the
order of cache/elements/<doc>.json.
"""
import sys

import numpy as np
import pandas as pd
import pymupdf
import torch

from common import (BAD_DOCS, CACHE, DOC_DIR, OUT, crop, doc_stem, load_elements, load_qa, read_jsonl, render,
                    test_questions, zoom)

MODEL = "vidore/colqwen2.5-v0.2"
MAX_VISUAL_TOKENS = 768
BATCH = 8
EMB = CACHE / "colqwen"
ELEM_EMB = CACHE / "colqwen_elements"
ELEM_BATCH = 16


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


def element_crops(doc_id):
    """Crops of every element of a document, in element order."""
    els = load_elements(doc_id)
    out = []
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        for p in sorted({e["page"] for e in els}):
            page = doc[p - 1]
            img, z = render(page), zoom(page)
            out += [crop(img, e["bbox"], z) for e in els if e["page"] == p]
    return out


def elements(shard=0, nshards=1):
    docs = sorted(set(test_questions(load_qa()).doc_id) - BAD_DOCS)[shard::nshards]
    ELEM_EMB.mkdir(parents=True, exist_ok=True)
    todo = [d for d in docs if not (ELEM_EMB / f"{doc_stem(d)}.pt").exists()]
    print(f"shard {shard}/{nshards}: {len(todo)} of {len(docs)} documents to embed", flush=True)
    if not todo:
        return
    model, proc = load_model()
    for n, d in enumerate(todo, 1):
        crops, emb = element_crops(d), []
        for k in range(0, len(crops), ELEM_BATCH):
            emb += run(model, proc.process_images(crops[k:k + ELEM_BATCH]).to(model.device))
        torch.save(emb, ELEM_EMB / f"{doc_stem(d)}.pt")
        print(f"[{n}/{len(todo)}] {d}: {len(emb)} elements, {sum(len(e) for e in emb) / len(emb):.0f} vectors each",
              flush=True)


def score_elements():
    qs = test_questions(load_qa())
    decomp = {r["qid"]: r["needs"] for r in read_jsonl(OUT / "decomp.jsonl")}
    model, proc = load_model()
    res = {"q": {}, "need": {}}
    for n, (d, g) in enumerate(qs.groupby("doc_id"), 1):
        emb = torch.load(ELEM_EMB / f"{doc_stem(d)}.pt")
        texts, keys = [], []
        for r in g.itertuples():
            texts.append(r.question); keys.append((r.qid, None))
            for j, need in enumerate(decomp.get(r.qid, [])):
                texts.append(need); keys.append((r.qid, j))
        s = proc.score_multi_vector(embed_texts(model, proc, texts), emb, device=model.device).float().numpy()
        for r in g.itertuples():
            res["q"][r.qid] = s[keys.index((r.qid, None))]
            rows = [row for (qid, j), row in zip(keys, s) if qid == r.qid and j is not None]
            if rows:
                res["need"][r.qid] = np.stack(rows)
        print(f"[{n}/{qs.doc_id.nunique()}] {d}: {len(g)} questions, {len(emb)} elements", flush=True)
    pd.to_pickle(res, OUT / "element_scores.pkl")
    print("wrote", OUT / "element_scores.pkl", flush=True)


if __name__ == "__main__":
    {"pages": pages, "score": score, "elements": elements, "score_elements": score_elements}[sys.argv[1]](
        *map(int, sys.argv[2:]))
