"""Synthetic Step 3 inputs to test analyze_human.py without annotators or a GPU (run after build_form.py).

  IDEA5_OUT=tests/out_SYNTHETIC IDEA5_EXPORTS=tests/exports_SYNTHETIC python tests/make_synthetic_human.py
  IDEA5_OUT=tests/out_SYNTHETIC IDEA5_EXPORTS=tests/exports_SYNTHETIC python analyze_human.py

Writes two random annotator exports, random VLM descriptions and random self-retrieval ranks (values are random,
not annotations or model outputs). Refuses to write anywhere whose path does not contain SYNTHETIC.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXPORTS, OUT  # noqa: E402

if "SYNTHETIC" not in str(OUT) or "SYNTHETIC" not in str(EXPORTS):
    raise SystemExit("refusing: IDEA5_OUT and IDEA5_EXPORTS must both contain SYNTHETIC")

rng = np.random.default_rng(1)
HUMAN = OUT / "human"
items = pd.read_csv(HUMAN / "items.csv")
EXPORTS.mkdir(parents=True, exist_ok=True)
cues = ["title", "caption", "labels", "period", "numbers", "entities", "pagelabel", "visual", "layout"]
scores = []
for who in ["SYNTHETIC_A", "SYNTHETIC_B"]:
    out = {"annotator": who, "form_version": "1", "items": []}
    for r in items.itertuples():
        words = r.question.split()
        out["items"].append({
            "item_id": r.item_id, "practice": r.group == "practice", "done": True,
            "generic": f"Page {r.target} of a {r.doc_type.lower()} with tables and text.",
            "distinct": " ".join(rng.choice(words, size=min(8, len(words)), replace=False)) + f" on page {r.target}.",
            "cues": list(rng.choice(cues, size=rng.integers(1, 4), replace=False)),
            "confidence": int(rng.integers(1, 6)), "notes": "", "t_generic_s": int(rng.integers(30, 120)),
            "t_distinct_s": int(rng.integers(40, 200))})
        if r.group != "practice":
            for kind, boost in [("generic", 0), ("distinct", 1)]:
                scores.append({"source": "human", "who": who, "item_id": r.item_id, "kind": kind,
                               "rank_all": int(rng.integers(1, 8)) if boost else int(rng.integers(1, 20)),
                               "rank_sib": int(rng.integers(1, 3)) if boost else int(rng.integers(1, 5)),
                               "n_pages": int(r.n_pages)})
    (EXPORTS / f"annotations_{who}_20261004-1500.json").write_text(json.dumps(out, indent=1))
model = "Qwen/Qwen2.5-VL-7B-Instruct"
with (HUMAN / "vlm_descriptions.jsonl").open("w") as f:
    for r in items.itertuples():
        for kind in ["generic", "distinct"]:
            f.write(json.dumps({"item_id": r.item_id, "kind": kind, "model": model,
                                "text": f"A {kind} synthetic description of page {r.target}."}) + "\n")
            if r.group != "practice":
                scores.append({"source": "vlm", "who": model, "item_id": r.item_id, "kind": kind,
                               "rank_all": int(rng.integers(1, 20)), "rank_sib": int(rng.integers(1, 5)),
                               "n_pages": int(r.n_pages)})
with (HUMAN / "desc_scores.jsonl").open("w") as f:
    for s in scores:
        f.write(json.dumps(s) + "\n")
print(f"SYNTHETIC: 2 exports, {len(scores)} scores -> {HUMAN}")
