"""Step 1: draw the Part A sample and render every page of the sampled documents.

Writes out/sample.csv, out/sample_docs.txt and cache/pages/<doc_stem>/<page:04d>.jpg.
"""
import numpy as np
import pandas as pd

from common import OUT, SEED, build_pool, load_qa, page_counts, render_doc

QUOTA = {"Chart": 12, "Table": 12, "Figure": 8, "Pure-text (Plain-text)": 8}
PRACTICE = ["Chart", "Table", "Pure-text (Plain-text)"]
MAX_PER_DOC = 2


def draw(pool, rng):
    # One random pass over the pool; accept while the stratum is open and the document has room.
    order = rng.permutation(len(pool))
    taken, per_doc, per_type = [], {}, {k: 0 for k in QUOTA}
    for i in order:
        r = pool.iloc[i]
        src = r["gold_source"]
        if src not in QUOTA or per_type[src] >= QUOTA[src] or per_doc.get(r["doc_id"], 0) >= MAX_PER_DOC:
            continue
        taken.append(i)
        per_type[src] += 1
        per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
    assert per_type == QUOTA, per_type
    return pool.iloc[taken]


def draw_practice(pool, real, rng):
    # Practice items come from documents outside the real sample, so their gold pages
    # (shown at step 3) give no hint about a real item.
    rest = pool[~pool["doc_id"].isin(set(real["doc_id"]))]
    picked, used_docs = [], set()
    for src in PRACTICE:
        cand = rest[(rest["gold_source"] == src) & ~rest["doc_id"].isin(used_docs)]
        r = cand.iloc[rng.integers(len(cand))]
        picked.append(r)
        used_docs.add(r["doc_id"])
    return pool.loc[[r.name for r in picked]]


def main():
    rng = np.random.default_rng(SEED)
    counts = page_counts()
    pool = build_pool(load_qa(counts))
    print("pool:", pool.groupby("gold_source")["doc_id"].agg(["size", "nunique"]).to_dict("index"))

    real = draw(pool, rng)
    practice = draw_practice(pool, real, rng)

    real = real.iloc[rng.permutation(len(real))].copy()
    real["item_id"] = [f"A{i + 1:02d}" for i in range(len(real))]
    real["practice"] = False
    practice = practice.copy()
    practice["item_id"] = [f"P{i + 1}" for i in range(len(practice))]
    practice["practice"] = True
    sample = pd.concat([practice, real])

    cols = ["item_id", "practice", "qid", "doc_id", "doc_type", "n_pages", "question", "answer",
            "answer_format", "gold_page", "gold_source", "gold_type", "evidence_pages", "evidence_sources"]
    OUT.mkdir(parents=True, exist_ok=True)
    sample[cols].to_csv(OUT / "sample.csv", index=False)
    docs = sorted(sample["doc_id"].unique())
    (OUT / "sample_docs.txt").write_text("\n".join(docs) + "\n")

    print("\nitems per type (real):", real["gold_source"].value_counts().to_dict())
    print("practice:", practice[["item_id", "gold_source"]].values.tolist())
    print("documents: real", real["doc_id"].nunique(), "| with practice", len(docs))
    print("max items per document:", real["doc_id"].value_counts().max())
    print("document types (real items):", real["doc_type"].value_counts().to_dict())
    print("total pages to process:", sum(counts[d] for d in docs))

    for d in docs:
        render_doc(d)
    print("pages rendered to cache/pages/")


if __name__ == "__main__":
    main()
