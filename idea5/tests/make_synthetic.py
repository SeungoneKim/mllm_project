"""Synthetic scan outputs to test attribute.py, build_form.py and analyze_human.py without a GPU.

  IDEA5_OUT=tests/out_SYNTHETIC python tests/make_synthetic.py
  IDEA5_OUT=tests/out_SYNTHETIC python attribute.py

Writes random question scores and page-to-page similarities (values are random, not model outputs) for every
eligible question of the documents on disk. Some gold pages are planted with a look-alike page that outscores them,
so every cause shows up. Refuses to write anywhere whose path does not contain SYNTHETIC.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, SCAN, doc_stem, eligible, load_qa  # noqa: E402

if "SYNTHETIC" not in str(OUT):
    raise SystemExit(f"refusing to write synthetic data to {OUT}; set IDEA5_OUT to a path containing SYNTHETIC")

rng = np.random.default_rng(0)
qa = eligible(load_qa())
(SCAN / "pagesim").mkdir(parents=True, exist_ok=True)
with (SCAN / "scores.jsonl").open("w") as f:
    for d, qs in qa.groupby("doc_id"):
        n = int(qs["n_pages"].iloc[0])
        base = rng.normal(0.6, 0.03, (n, n))
        sim = (base + base.T) / 2
        np.fill_diagonal(sim, 1.0)
        for _, q in qs.iterrows():
            s = rng.normal(10, 1, n)
            gold = [p - 1 for p in q["ev_pages"]]
            s[gold] += rng.choice([4, 0, -1], p=[0.6, 0.2, 0.2])
            if rng.random() < 0.3:  # plant a look-alike that beats the first gold page
                twin = int(rng.choice([p for p in range(n) if p not in gold]))
                sim[gold[0], twin] = sim[twin, gold[0]] = 0.95
                s[twin] = s[gold].max() + 3
            f.write(json.dumps({"qid": int(q["qid"]), "doc_id": d, "scores": s.tolist()}) + "\n")
        np.save(SCAN / "pagesim" / f"{doc_stem(d)}.npy", sim.astype(np.float32))
print(f"SYNTHETIC: {len(qa)} questions over {qa['doc_id'].nunique()} documents -> {SCAN}")
