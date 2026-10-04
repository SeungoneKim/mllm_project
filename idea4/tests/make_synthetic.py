"""Random SYNTHETIC annotation exports, for testing analyze_A.py only. Not real annotations.

Values are drawn at random from the form's options; spec words are random words from other
questions in the QA file, not from the item. Files: tests/annotations_SYNTHETIC_<k>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import IDEA4, OUT, SEED, load_qa  # noqa: E402

TYPES = ["text", "table", "chart", "figure", "layout"]
SPEC_KEYS = ["element", "subtype", "title", "x_label", "y_label", "categories", "series", "values", "style",
             "caption", "columns", "rows", "heading", "passage", "description"]
LISTS = {"categories", "series", "columns", "rows"}


def random_spec(rng, t, vocab):
    words = lambda k: " ".join(rng.choice(vocab, k))
    spec = {k: [] if k in LISTS else "" for k in SPEC_KEYS}
    spec["element"] = t
    if t == "chart":
        spec.update(subtype=str(rng.choice(["bar", "line", "pie or donut"])), title=words(4), x_label=words(1),
                    y_label=words(1), categories=list(rng.choice(vocab, 3)), series=list(rng.choice(vocab, 2)))
    elif t == "table":
        spec.update(caption=words(4), columns=list(rng.choice(vocab, 3)), rows=list(rng.choice(vocab, 3)))
    elif t == "text":
        spec.update(passage=words(20))
    else:
        spec.update(description=words(10))
    return spec


def main(n_files=2):
    rng = np.random.default_rng(SEED)
    sample = pd.read_csv(OUT / "sample.csv")
    vocab = np.array(sorted({w for q in load_qa()["question"].sample(200, random_state=SEED)
                             for w in q.lower().split() if w.isalpha() and len(w) > 3}))
    for k in range(n_files):
        items, t = [], 0
        for _, r in sample.iterrows():
            s1t = str(rng.choice(TYPES))
            s2t = s1t if rng.random() < 0.7 else str(rng.choice(TYPES))
            spec1 = random_spec(rng, s1t, vocab)
            spec2 = spec1 if s2t == s1t else random_spec(rng, s2t, vocab)
            r1, r2 = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
            step1 = {"type": s1t, "render": r1, "spec": spec1, "cues": "SYNTHETIC", "confidence": int(rng.integers(1, 6)),
                     "t_open": t, "t_close": t + int(rng.integers(20000, 120000))}
            t = step1["t_close"]
            step2 = {"type": s2t, "render": r2, "spec": spec2, "changed": s2t != s1t or r1 != r2 or spec1 != spec2,
                     "t_open": t, "t_close": t + int(rng.integers(5000, 60000))}
            t = step2["t_close"]
            x0, y0 = rng.random(2) * 0.5
            step3 = {"bbox": [round(x0, 3), round(y0, 3), round(x0 + 0.3, 3), round(y0 + 0.3, 3)], "no_clear_region": False,
                     "subtype_match": str(rng.choice(["yes", "partly", "no", "not applicable"])),
                     "label_ok": str(rng.choice(["yes", "yes", "yes", "no", "unsure"])), "notes": "SYNTHETIC",
                     "t_open": t, "t_close": t + int(rng.integers(10000, 60000))}
            t = step3["t_close"]
            items.append({"item_id": r["item_id"], "practice": bool(r["practice"]),
                          "step1": step1, "step2": step2, "step3": step3})
        data = {"annotator": f"SYNTHETIC_{k}", "form_version": "1", "exported_at": "2026-10-03T00:00:00Z",
                "inventory_source": "labels", "items": items}
        path = IDEA4 / "tests" / f"annotations_SYNTHETIC_{k}.json"
        path.write_text(json.dumps(data, indent=1))
        print("wrote", path)


if __name__ == "__main__":
    main()
