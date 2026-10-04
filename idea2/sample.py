"""Draw the stratified sample of multi-source questions for manual categorization.

Usage: python sample.py DATA_DIR [--per-stratum 10]

Strata (answerable questions whose listed evidence pages all exist in the PDF):
  same_mod    one modality label, >= 2 distinct evidence pages          (Figure 2a)
  cross_same  >= 2 modality labels, exactly one evidence page           (Figure 2b, "same page")
  cross_multi >= 2 modality labels, >= 2 distinct evidence pages        (Figure 2b, other spans)
At most one question per document within a stratum, so no single document dominates. Questions on the two
documents in BAD_DOCS are skipped while drawing (the pool and the permutation are unchanged).

Writes out/sample.csv and renders every evidence page to cache/pages/<qid>/p<page>.jpg, with the
text layer next to it (p<page>.txt), for the annotator.
"""
import argparse

import numpy as np
import pandas as pd

from common import BAD_DOCS, CACHE, OUT, SEED, load_qa, multi_source_groups, page_text, render_page


def draw(pool, n, rng):
    order = rng.permutation(len(pool))
    picked, docs = [], set()
    for i in order:
        row = pool.iloc[i]
        if row.doc_id in docs or row.doc_id in BAD_DOCS:
            continue
        picked.append(row.qid)
        docs.add(row.doc_id)
        if len(picked) == n:
            break
    return pool[pool.qid.isin(picked)].set_index("qid").loc[picked].reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--per-stratum", type=int, default=10)
    args = ap.parse_args()

    df = load_qa(args.data_dir)
    same, cross = multi_source_groups(df)
    ok = lambda d: d[d.answerable & d.pages_in_pdf]
    strata = {
        "same_mod": ok(same[same.n_distinct >= 2]),
        "cross_same": ok(cross[cross.same_page]),
        "cross_multi": ok(cross[cross.n_distinct >= 2]),
    }
    rng = np.random.default_rng(SEED)
    parts = []
    for name, pool in strata.items():
        s = draw(pool.sort_values("qid"), args.per_stratum, rng)
        s.insert(0, "stratum", name)
        parts.append(s)
        print(f"{name}: pool {len(pool)} ({int((~pool.doc_id.isin(BAD_DOCS)).sum())} outside BAD_DOCS), sampled {len(s)}")
    sample = pd.concat(parts, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    cols = ["stratum", "qid", "doc_id", "doc_type", "combo", "pages", "sources", "question", "answer", "answer_format"]
    sample[cols].to_csv(OUT / "sample.csv", index=False)

    for row in sample.itertuples():
        for p in sorted(set(row.pages)):
            render_page(args.data_dir, row.doc_id, p, CACHE / "pages" / str(row.qid) / f"p{p}.jpg")
            (CACHE / "pages" / str(row.qid) / f"p{p}.txt").write_text(page_text(args.data_dir, row.doc_id, p))
    print("wrote", OUT / "sample.csv", "and page renders under", CACHE / "pages")


if __name__ == "__main__":
    main()
