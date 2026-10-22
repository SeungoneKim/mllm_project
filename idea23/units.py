"""v2 step (GPU): the final retrieval over atomic facts, with a text embedding model.

  python units.py        # reads out/<v>/aggregate/<model>/facts_*.jsonl, writes out/<v>/unit_scores.pkl

The facts the VLM read off the retrieved elements (aggregate.py) are short statements, so they are matched to the
question and to each need with Qwen3-Embedding-4B: r(u, x) = cos(emb(x), emb(u)), the query side with the model's
retrieval instruction. selection.py (method `evidence`) runs coverage over the facts with these scores.
out/v2/unit_scores.pkl holds {qid: {"facts": [...], "r_q": float32[U], "r_need": float32[m, U]}}; a question whose
VLM reply did not parse, or that has no fact, is left out (evidence then falls back to elem_coverage_mod).
"""
import os

import numpy as np
import pandas as pd

from aggregate import AGG
from common import VOUT, load_qa, read_jsonl

MODEL = os.environ.get("UNIT_MODEL", "Qwen/Qwen3-Embedding-4B")
INSTRUCT = "Instruct: Given a search query about a document, retrieve the facts from the document that answer it\nQuery: "
BATCH = 64


def embed(tok, model, texts):
    import torch
    out = []
    for k in range(0, len(texts), BATCH):
        enc = tok(texts[k:k + BATCH], padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
        last = h[:, -1]  # left padding: the last token pools the sequence (as the model card)
        out.append(torch.nn.functional.normalize(last.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    rows = {r["qid"]: r for p in sorted(AGG.glob("facts_*.jsonl")) for r in read_jsonl(p)}
    cands = {r["qid"]: r for r in read_jsonl(VOUT / "candidates.jsonl")}
    keep = {q: r for q, r in rows.items() if r["parsed"] and r["facts"]}
    qa = load_qa().set_index("qid")
    print(f"{len(rows)} aggregated questions from {AGG}, {len(keep)} with facts", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    queries, facts, index = [], [], []
    for qid, r in keep.items():
        needs = cands[qid]["needs"]
        queries += [INSTRUCT + x for x in [qa.loc[qid].question.strip()] + needs]
        facts += [f["fact"] for f in r["facts"]]
        index.append((qid, len(needs), len(r["facts"])))
    eq, ef = embed(tok, model, queries), embed(tok, model, facts)
    out, qi, fi = {}, 0, 0
    for qid, m, u in index:
        sim = eq[qi:qi + 1 + m] @ ef[fi:fi + u].T
        out[qid] = {"facts": keep[qid]["facts"], "r_q": sim[0], "r_need": sim[1:]}
        qi, fi = qi + 1 + m, fi + u
    pd.to_pickle(out, VOUT / "unit_scores.pkl")
    print("wrote", VOUT / "unit_scores.pkl", flush=True)


if __name__ == "__main__":
    main()
