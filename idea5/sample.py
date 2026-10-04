"""Part B, step 1 (CPU): sample questions and prepare every input of the modality comparison.

  python sample.py [--per-type 16]   # writes out/sample/items.jsonl and cache/sample/<qid>_{hi,lo,mlo}.png

Questions: answerable, one evidence page inside the PDF, one evidence type, from documents outside idea4's sample,
at most 2 per document, --per-type of each evidence type (fewer when a type has fewer). Seed 555.
For each gold page:
  hi    the page image at up to 1280 visual tokens
  lo    the page image at up to 256 visual tokens
  mlo   the lo image with every text-layer word painted white (what the pixels hold beyond the text layer)
  text  the PDF text layer
"""
import argparse
import json

import numpy as np
import pymupdf

from common import (CACHE, DOC_DIR, MIN_TEXT_WORDS, OUT, SEED, TOKENS_HIGH, TOKENS_LOW, TYPES, andre_docs, eligible,
                    fit, load_qa, masked, n_words, render, text_layer)

SAMPLE = OUT / "sample"
IMG = CACHE / "sample"
MAX_PER_DOC = 2


def select(per_type):
    qa = eligible(load_qa())
    qa = qa[(qa["ev_pages"].map(len) == 1) & (qa["ev_types"].map(len) == 1) & ~qa["doc_id"].isin(andre_docs())]
    qa["type"] = qa["ev_types"].map(lambda t: t[0])
    rng = np.random.default_rng(SEED)
    qa = qa.iloc[rng.permutation(len(qa))]
    taken, per_doc = [], {}
    for t in TYPES:
        k = 0
        for _, r in qa[qa["type"] == t].iterrows():
            if k >= per_type or per_doc.get(r["doc_id"], 0) >= MAX_PER_DOC:
                continue
            taken.append(r)
            per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
            k += 1
    return taken


def main(per_type):
    SAMPLE.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    items = []
    for r in select(per_type):
        page = r["ev_pages"][0]
        with pymupdf.open(DOC_DIR / r["doc_id"]) as doc:
            p = doc[page - 1]
            img, zoom = render(p)
            text = text_layer(p)
            fit(img, TOKENS_HIGH).save(IMG / f"{r['qid']}_hi.png")
            fit(img, TOKENS_LOW).save(IMG / f"{r['qid']}_lo.png")
            fit(masked(img, p, zoom), TOKENS_LOW).save(IMG / f"{r['qid']}_mlo.png")
        items.append({"qid": int(r["qid"]), "doc_id": r["doc_id"], "doc_type": r["doc_type"], "page": page,
                      "n_pages": int(r["n_pages"]), "type": r["type"], "question": r["question"],
                      "answer": r["answer"], "answer_format": r["answer_format"], "text": text,
                      "text_words": n_words(text), "no_text_layer": n_words(text) < MIN_TEXT_WORDS})
    with (SAMPLE / "items.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = {t: sum(it["type"] == t for it in items) for t in TYPES}
    print(f"{len(items)} items from {len({it['doc_id'] for it in items})} documents: {counts}; "
          f"no text layer: {sum(it['no_text_layer'] for it in items)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=16)
    main(ap.parse_args().per_type)
