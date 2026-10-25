"""Random reader outputs to test evaluate.py without a GPU (values are random, not results).

  IDEA5_MT_READER=tests/out_SYNTHETIC/reader python tests/make_synthetic_answers.py
  IDEA5_MT_READER=tests/out_SYNTHETIC/reader IDEA5_MT_EVAL=tests/out_SYNTHETIC/eval python evaluate.py

Each row answers correctly with a random probability; kept tokens come from the real input selectors (rand, ink,
comp) and random indices for the in-model ones. Refuses to write anywhere whose path does not contain SYNTHETIC.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conds import INPUT_METHODS, MODEL_METHODS, conditions, parse  # noqa: E402
from shared import TEST  # noqa: E402
from token_select import select_input  # noqa: E402

READER = Path(os.environ.get("IDEA5_MT_READER", ""))
if "SYNTHETIC" not in str(READER):
    raise SystemExit("refusing: set IDEA5_MT_READER to a path containing SYNTHETIC")

rng = np.random.default_rng(0)
items = [json.loads(l) for l in (TEST / "items.jsonl").open()]
tok = np.load(TEST / "tokinfo.npz")
READER.mkdir(parents=True, exist_ok=True)
with (READER / "answers_0.jsonl").open("w") as f:
    for it in items:
        ink = np.concatenate([tok[f"{key}_ink"].ravel() for key in it["page_keys"]])
        txt = np.concatenate([tok[f"{key}_txt"].ravel() for key in it["page_keys"]])
        n = ink.size
        for cond in conditions():
            _, method, k = parse(cond)
            k = k * it["n_gold"] if k else None
            kept, merged = None, 0
            if method in INPUT_METHODS:
                kept = select_input(method, k, ink, txt, it["qid"]).tolist()
            elif method in MODEL_METHODS:
                merged = k // 2 if method == "svlm" else 0
                kept = sorted(rng.choice(n, size=min(n, k - merged), replace=False).tolist())
            n_vis = 0 if cond == "text" else (min(n, k) if cond.startswith("down") else n)
            n_kept = len(kept) if kept is not None else n_vis
            pred = it["answer"] if rng.random() < 0.5 else "wrong"
            f.write(json.dumps({"qid": it["qid"], "cond": cond, "pred": pred, "input_tokens": 700 + n_vis,
                                "n_vis": n_vis, "grids": [], "n_kept": n_kept, "n_merged": merged,
                                "kept": kept, "vis_per_layer": float(n_kept + merged),
                                "seq_after": 700 + n_kept + merged, "sec": 0.1}) + "\n")
print(f"SYNTHETIC: {len(items) * len(conditions())} rows -> {READER}")
