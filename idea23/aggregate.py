"""v2 step (GPU): a VLM reads the retrieved evidence elements of each question and merges them into atomic facts.

  python aggregate.py [--shard i --nshards n] [--limit N] [--sets multi]
      reads out/<v>/candidates.jsonl (selection.py --candidates), writes out/<v>/aggregate/<model>/facts_<i>.jsonl
      the VLM is the version's agg_model (versions.py), or AGG_MODEL

Each question's candidates are at most 16 elements found by modality-decomposed retrieval (evidence.Elements.
candidates): the best elements of each form for the question and for each need. The VLM sees their crops, each
labelled with an id, its page and its form, and lists the facts that help answer the question:
  {"fact": short statement with the exact values, "evidence": [ids it is read from], "same_as": [ids of regions that
   repeat it], "needs": [numbers of the needs it serves]}
A fact with several evidence ids can only be read from those regions together (jointly necessary evidence); same_as
marks redundant copies. "missing" lists needed information that no region holds. The facts select pages only; the
reader never sees them. The prompt is written by hand; nothing is tuned on the test questions.
"""
import argparse
import json
import os
import re
from pathlib import Path

from common import DOC_DIR, VERSION, VOUT, crop, fit, load_elements, load_qa, read_jsonl, render, zoom
from versions import VERSIONS

MODEL = os.environ.get("AGG_MODEL") or VERSIONS[VERSION].get("agg_model", "Qwen/Qwen3-VL-8B-Instruct")
UNIT = VERSIONS[VERSION].get("agg_unit", "element")  # "page": the control that shows whole candidate pages
AGG = Path(os.environ.get("AGG_OUT", VOUT / "aggregate" / Path(MODEL).name))  # AGG_OUT: smoke tests
PATCH = 32          # Qwen3-VL: one visual token per 32 x 32 pixels
TOKENS_PER_ELEMENT = 640
TOKENS_PER_PAGE = VERSIONS[VERSION].get("page_tokens", 1024)  # page controls: 1,024 as the reader, or the crops' budget
BATCH = 1 if UNIT == "page" else 2  # whole pages: about 11k visual tokens per question
MAX_NEW = 768
FORM_NAME = {"Text": "text", "Layout": "heading or title", "Table": "table", "Chart": "chart", "Figure": "figure",
             "Page": "whole page"}

TASK = """Above are {n} {what} of a long document, found by a search engine for the question below. They may be \
relevant or not, and some may repeat each other.

Question: {question}
Information the question needs:
{needs}

List the facts in these regions that help answer the question.
- A fact is one short statement taken from a region, with its exact numbers, names and units, e.g. "Asia segment \
revenue in 2019: $4.2 billion".
- "evidence": the ids of the regions the fact is read from. Give one id when one region states the fact. Give several \
ids only when the fact can be read only by combining them, e.g. a chart and the separate legend or note that says what \
it shows.
- "same_as": ids of other regions that state the same fact again, or [].
- "needs": the numbers of the needed information that the fact helps with.
- Leave out regions that do not help. Do not answer the question, do not compute, and do not invent facts.
- "missing": needed information that no region provides, as short search queries, or [].
Reply with JSON only: {{"facts": [{{"fact": "...", "evidence": ["R1"], "same_as": [], "needs": [1]}}], "missing": []}}"""


def conversation(question, needs, regions):
    """regions: [(label, page, form, image, element id)]"""
    content = []
    for label, page, form, img, _ in regions:
        content += [{"type": "text", "text": f"[{label}] page {page}, {FORM_NAME[form]}:"}, {"type": "image", "image": img}]
    what = "pages" if UNIT == "page" else "regions cut from the pages"
    content.append({"type": "text", "text": TASK.format(n=len(regions), what=what, question=question.strip(),
                                                        needs="\n".join(f"{j}. {d}" for j, d in enumerate(needs, 1)))})
    return [{"role": "user", "content": content}]


def parse(reply, labels, m):
    """-> (facts with element indices, missing, ok). Unknown ids and needs are dropped; a fact without evidence too."""
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    hit = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(hit.group())
    except (AttributeError, ValueError):
        return [], [], False
    ids = {l: i for i, l in enumerate(labels)}

    def to_idx(xs):
        out = []
        for x in xs if isinstance(xs, list) else [xs]:
            x = str(x).strip().strip("[]")
            if x in ids and ids[x] not in out:
                out.append(ids[x])
        return out

    facts = []
    for f in obj.get("facts", []) if isinstance(obj, dict) else []:
        if not isinstance(f, dict):
            continue
        ev = to_idx(f.get("evidence", []))
        if not ev:
            continue
        same = [i for i in to_idx(f.get("same_as", [])) if i not in ev]
        needs = []
        for j in f.get("needs", []) if isinstance(f.get("needs"), list) else []:
            try:
                if 1 <= int(j) <= m and int(j) - 1 not in needs:
                    needs.append(int(j) - 1)
            except (TypeError, ValueError):
                pass
        facts.append({"fact": str(f.get("fact", "")).strip(), "evidence": ev, "same_as": same, "needs": needs})
    missing = [str(x) for x in obj.get("missing", [])] if isinstance(obj, dict) and isinstance(obj.get("missing"), list) else []
    return facts, missing, True


class VLM:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto",
                                                                 attn_implementation="sdpa").eval()

    def generate(self, convs, max_tokens=MAX_NEW):
        inputs = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return [t.strip() for t in self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                                          skip_special_tokens=True)]


def regions_of(doc_id, idx):
    """Crops of the given elements of a document, labelled R1, R2, ... in the given order; with UNIT "page", their
    pages instead (in order of first appearance), each standing for its first candidate element."""
    import pymupdf
    els = load_elements(doc_id)
    if UNIT == "page":
        first = {}
        for i in idx:
            first.setdefault(els[i]["page"], i)
        with pymupdf.open(DOC_DIR / doc_id) as doc:
            return [(f"R{n}", p, "Page", fit(render(doc[p - 1]), TOKENS_PER_PAGE, PATCH), first[p])
                    for n, p in enumerate(first, 1)]
    crops = {}
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        for p in sorted({els[i]["page"] for i in idx}):
            page = doc[p - 1]
            img, z = render(page), zoom(page)
            for i in idx:
                if els[i]["page"] == p:
                    crops[i] = fit(crop(img, els[i]["bbox"], z), TOKENS_PER_ELEMENT, PATCH)
    return [(f"R{n}", els[i]["page"], els[i]["form"], crops[i], i) for n, i in enumerate(idx, 1)]


def main(shard, nshards, sets, limit):
    AGG.mkdir(parents=True, exist_ok=True)
    qa = load_qa().set_index("qid")
    cands = [r for r in read_jsonl(VOUT / "candidates.jsonl") if r["set"] in sets]
    todo = [r for i, r in enumerate(sorted(cands, key=lambda r: r["qid"])) if i % nshards == shard]
    done = {r["qid"] for p in AGG.glob("facts_*.jsonl") for r in read_jsonl(p)}
    todo = [r for r in todo if r["qid"] not in done][: limit or None]
    print(f"shard {shard}/{nshards}: {len(todo)} questions to aggregate with {MODEL}", flush=True)
    if not todo:
        return
    vlm = VLM()
    with (AGG / f"facts_{shard}.jsonl").open("a") as f:
        for k in range(0, len(todo), BATCH):
            batch = todo[k:k + BATCH]
            convs, regs = [], []
            for r in batch:
                q = qa.loc[r["qid"]]
                rg = regions_of(q.doc_id, r["elements"])
                regs.append(rg)
                convs.append(conversation(q.question, r["needs"], rg))
            for r, rg, reply in zip(batch, regs, vlm.generate(convs)):
                facts, missing, ok = parse(reply, [x[0] for x in rg], len(r["needs"]))
                for fact in facts:  # region indices -> element ids of the document
                    fact["evidence"] = [rg[i][4] for i in fact["evidence"]]
                    fact["same_as"] = [rg[i][4] for i in fact["same_as"]]
                f.write(json.dumps({"qid": r["qid"], "elements": [x[4] for x in rg], "facts": facts, "missing": missing,
                                    "parsed": ok, "raw": reply}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"aggregated {k + len(batch)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--sets", default=VERSIONS[VERSION].get("agg_sets", "multi,control"),
                    help="question sets to aggregate (default: the version's agg_sets)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions of the shard only (smoke test)")
    a = ap.parse_args()
    main(a.shard, a.nshards, a.sets.split(","), a.limit)
