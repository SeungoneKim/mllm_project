"""Part B proposer inputs and outputs.

  python proposer.py inputs    # writes proposer/inputs_<k>.json (blind inputs) and proposer/id_map.csv
  python proposer.py collect   # checks proposer/outputs_<k>.json and writes proposer/specs.jsonl

The proposer sees only what an annotator saw at step 2: question, document type, page count and the
VLM inventory sentence. Ids are opaque (Q001...). The gold fields stay in id_map.csv, which the proposer never sees.
"""
import json
import sys

import numpy as np
import pandas as pd

from build_form import vlm_inventory
from common import IDEA4, OUT, SEED, build_pool, load_qa

PDIR = IDEA4 / "proposer"
N_CHUNKS = 3
SPEC_KEYS = ["element", "subtype", "title", "x_label", "y_label", "categories", "series", "values", "style",
             "caption", "columns", "rows", "heading", "passage", "description"]
LISTS = {"categories", "series", "columns", "rows"}
TYPES = {"text", "table", "chart", "figure", "layout"}

PROMPT = """You are the proposer in a document retrieval system. A user asks a question about one long PDF document.
Before the system looks at any page, you predict what the evidence that answers the question looks like.
Your specification will be drawn as an image and used to find the page.

For each item you get the question, the document type, the page count, and an inventory of the document
(which element types it contains, made by a vision model at index time; it can be wrong).

For each item, output:
- "type": one of "text" (paragraphs or lists), "table", "chart", "figure" (photo, diagram, screenshot, map), "layout"
  (the answer depends on the page layout, for example a title slide or a cover).
- "render": true if a drawn picture of the evidence would help a retriever find the page; false if the answer is in a paragraph.
- "spec": the evidence you expect, with these keys (use "" or [] for keys that do not apply):
  "element" (same as type), "subtype" (for charts: bar, horizontal bar, grouped bar, stacked bar, line, pie or donut,
  scatter, area, other), "title", "x_label", "y_label", "categories" (list), "series" (list), "values" (free text,
  may be invented), "style" (colours, fonts), "caption", "columns" (list), "rows" (list), "heading",
  "passage" (1 to 3 sentences of the text you expect), "description" (1 or 2 sentences, for figures and layout).
  Titles, labels, headers and passages should use the words you expect to be printed on the page. Values may be invented.

Fill the spec fully for the chosen type: a chart needs subtype, title, axis labels, categories and series; a table needs
caption, columns and rows; text needs a passage; a figure or layout needs a description.

Do not use any tool. Do not read any file. Work only from the items below.
Answer with one JSON object and nothing else: {"items": [{"id": "Q001", "type": ..., "render": ..., "spec": {...}}, ...]}
with one entry per item, in the same order.

ITEMS:
"""


def make_inputs():
    qa = load_qa()
    docs = set((OUT / "sample_docs.txt").read_text().split())
    pool = build_pool(qa)
    pool = pool[pool["doc_id"].isin(docs)].copy()
    pool = pool.iloc[np.random.default_rng(SEED).permutation(len(pool))]
    pool["pid"] = [f"Q{i + 1:03d}" for i in range(len(pool))]
    inv = json.loads((OUT / "inventory" / "docs.json").read_text())
    sample = pd.read_csv(OUT / "sample.csv")
    pool["item_id"] = pool["qid"].map(dict(zip(sample["qid"], sample["item_id"])))
    PDIR.mkdir(exist_ok=True)
    pool[["pid", "qid", "item_id", "doc_id", "gold_page", "gold_type", "question"]].to_csv(PDIR / "id_map.csv", index=False)
    items = [{"id": r["pid"], "question": r["question"], "document_type": r["doc_type"], "pages": int(r["n_pages"]),
              "inventory": vlm_inventory(inv[r["doc_id"]])} for _, r in pool.iterrows()]
    (PDIR / "prompt.txt").write_text(PROMPT)
    for k, chunk in enumerate(np.array_split(np.arange(len(items)), N_CHUNKS)):
        (PDIR / f"inputs_{k}.json").write_text(json.dumps([items[i] for i in chunk], indent=1, ensure_ascii=False))
    print(f"{len(items)} items in {N_CHUNKS} chunks; types {pool['gold_type'].value_counts().to_dict()}")


def clean_spec(t, spec):
    out = {}
    for k in SPEC_KEYS:
        v = spec.get(k, [] if k in LISTS else "")
        if k in LISTS:
            out[k] = [str(x) for x in v] if isinstance(v, list) else [s.strip() for s in str(v).split(",") if s.strip()]
        else:
            out[k] = v if isinstance(v, str) else json.dumps(v)
    out["element"] = t
    return out


def collect():
    ids = pd.read_csv(PDIR / "id_map.csv")["pid"].tolist()
    got = {}
    for k in range(N_CHUNKS):
        data = json.loads((PDIR / f"outputs_{k}.json").read_text())
        want = [x["id"] for x in json.loads((PDIR / f"inputs_{k}.json").read_text())]
        have = [x["id"] for x in data["items"]]
        assert have == want, f"chunk {k}: ids differ"
        for x in data["items"]:
            t = str(x["type"]).lower()
            assert t in TYPES, (x["id"], t)
            got[x["id"]] = {"pid": x["id"], "type": t, "render": bool(x["render"]), "spec": clean_spec(t, x["spec"])}
    assert sorted(got) == sorted(ids)
    with (PDIR / "specs.jsonl").open("w") as f:
        for pid in ids:
            f.write(json.dumps(got[pid], ensure_ascii=False) + "\n")
    print(f"wrote {len(got)} specs")


if __name__ == "__main__":
    {"inputs": make_inputs, "collect": collect}[sys.argv[1]]()
