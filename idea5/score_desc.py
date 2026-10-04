"""Step 3c: self-retrieval of every description (GPU). This is the training reward proposed for Idea 5.

  CUDA_VISIBLE_DEVICES=0 python score_desc.py   # reads exports/*.json and out/human/vlm_descriptions.jsonl,
                                                # writes out/human/desc_scores.jsonl

Each description is embedded as a ColQwen query and scored against every page of its document (the cached page
embeddings, as in scan.py). rank_all: rank of the target among all pages; rank_sib: rank of the target among
itself and its 3 look-alikes (1 to 4).
"""
import json

import pandas as pd

from common import EXPORTS, OUT, emb_path
from scan import embed_texts, load_model, maxsim_all, pad

HUMAN = OUT / "human"


def load_descriptions():
    rows = []
    for p in sorted(EXPORTS.glob("annotations_*.json")):
        data = json.loads(p.read_text())
        if "SYNTHETIC" in p.name and "SYNTHETIC" not in str(OUT):
            continue  # synthetic test exports never reach the real outputs
        for it in data["items"]:
            if it.get("practice") or not it.get("done"):
                continue
            for kind in ["generic", "distinct"]:
                rows.append({"source": "human", "who": data["annotator"], "item_id": it["item_id"], "kind": kind,
                             "text": it[kind]})
    vf = HUMAN / "vlm_descriptions.jsonl"
    if vf.exists():
        for r in map(json.loads, vf.open()):
            rows.append({"source": "vlm", "who": r["model"], "item_id": r["item_id"], "kind": r["kind"], "text": r["text"]})
    return pd.DataFrame(rows)


def main():
    import torch
    items = pd.read_csv(HUMAN / "items.csv").set_index("item_id")
    desc = load_descriptions()
    desc = desc[desc["item_id"].isin(items.index) & (desc["text"].str.strip() != "")]
    print(desc.groupby(["source", "who", "kind"]).size().to_string())
    model, proc = load_model()
    with (HUMAN / "desc_scores.jsonl").open("w") as f:
        for d, ids in items.groupby("doc_id").groups.items():
            sub = desc[desc["item_id"].isin(ids)]
            if sub.empty:
                continue
            P, mask = pad(torch.load(emb_path(d)))
            for r, q in zip(sub.itertuples(), embed_texts(model, proc, list(sub["text"]))):
                it = items.loc[r.item_id]
                s = maxsim_all(q, P, mask)
                t = int(it["target"])
                group = [t] + json.loads(it["neighbours"])
                rank_all = 1 + sum(v > s[t - 1] for v in s)
                rank_sib = 1 + sum(s[p - 1] > s[t - 1] for p in group[1:])
                f.write(json.dumps({"source": r.source, "who": r.who, "item_id": r.item_id, "kind": r.kind,
                                    "rank_all": int(rank_all), "rank_sib": int(rank_sib), "n_pages": len(s)}) + "\n")
            print(d, len(sub), flush=True)


if __name__ == "__main__":
    main()
