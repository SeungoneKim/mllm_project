"""Optional step (GPU): which modality holds the evidence on each gold page of a multi-page question.

  python tag_pages.py [--limit N]       # writes out/page_tags.jsonl, one row per (question, gold page); resumable

MMLongBench-Doc labels evidence sources per question, not per page, so for a Chart + Text question with two pages we
do not know which page holds the chart. The reader model sees one gold page with the question and picks the form of
the evidence on that page among the question's own labelled sources (for single-source questions the answer is
fixed and the model is not called). This lets analyze.py ask which kind of page the retriever misses.
"""
import argparse
import json
import re

from common import OUT, doc_images, fit, load_qa, read_jsonl, test_questions
from reader import Reader

NAMES = {"Text": "text (paragraphs or sentences)", "Layout": "layout (headings, titles, page structure, positions)",
         "Table": "table", "Chart": "chart or plot", "Figure": "figure (photo, drawing, diagram, map, icon)"}
BATCH = 8
TOKENS = 1280

PROMPT = """This is page {page} of a long document. The question below needs information from several pages, and this \
page is one of them.

Question: {question}

In which form does THIS page hold the information it contributes to the answer? Choose exactly one of:
{options}
Reply with JSON only: {{"form": "<one of: {keys}>"}}"""


def items():
    qs = test_questions(load_qa())
    for r in qs[qs.set == "multi"].itertuples():
        for p in r.gold:
            yield r, p


def main(limit):
    out_file = OUT / "page_tags.jsonl"
    done = {(r["qid"], r["page"]) for r in read_jsonl(out_file)}
    todo = [(r, p) for r, p in items() if (r.qid, p) not in done][: limit or None]
    print(f"{len(todo)} gold pages to tag", flush=True)
    fixed = [(r, p) for r, p in todo if r.n_mods == 1]
    ask = [(r, p) for r, p in todo if r.n_mods > 1]
    with out_file.open("a") as f:
        for r, p in fixed:
            f.write(json.dumps({"qid": int(r.qid), "page": p, "form": r.mods[0], "by": "label"}) + "\n")
        if not ask:
            return
        reader = Reader()
        for k in range(0, len(ask), BATCH):
            batch = ask[k:k + BATCH]
            convs = []
            for r, p in batch:
                img = fit(doc_images(r.doc_id, [p])[0], TOKENS)
                options = "\n".join(f"- {m}: {NAMES[m]}" for m in r.mods)
                text = PROMPT.format(page=p, question=r.question.strip(), options=options, keys=", ".join(r.mods))
                convs.append([{"role": "user", "content": [{"type": "image", "image": img},
                                                           {"type": "text", "text": text}]}])
            preds, _, _ = reader.generate(convs, max_tokens=24)
            for (r, p), pred in zip(batch, preds):
                found = [m for m in r.mods if re.search(rf"\b{m}\b", pred, re.I)]
                f.write(json.dumps({"qid": int(r.qid), "page": p, "form": found[0] if found else None, "by": "vlm",
                                    "options": list(r.mods), "raw": pred}) + "\n")
            f.flush()
            print(f"tagged {k + len(batch)}/{len(ask)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args().limit)
