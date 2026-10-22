"""Index step (GPU, v1 and v2): the evidence forms each decomposed need expects, predicted from the question only.

  python need_forms.py [--limit N]       # reads out/decomp.jsonl, writes out/need_forms.jsonl (resumable)

For every need d_j of a question, Qwen3-8B lists the forms its evidence most likely takes, at most two of Text, Layout,
Table, Chart, Figure (the evidence sources of MMLongBench-Doc), or "any" when it cannot tell. The benchmark's own
evidence-source labels are never shown to the model; they stay evaluation labels. The few-shot examples are written by
hand. A reply that does not parse gives "any" for every need.
"""
import argparse
import json
import re

from common import FORMS, OUT, load_qa, read_jsonl, test_questions
from decompose import BATCH, MODEL

MAX_FORMS = 2

SYSTEM = """A search engine looks for the evidence of each piece of information in a long document (a report, paper, \
slide deck, brochure, guidebook or form). Evidence comes in five forms:
- Text: paragraphs, sentences, bullet lists
- Layout: titles, headings, slide or page structure, where something sits on the page
- Table: values in rows and columns
- Chart: a bar, line, pie or other plot of data
- Figure: a photo, drawing, diagram, map, logo, icon or screenshot
For each search query below, give the forms in which its evidence is most likely to appear, most likely first, at most \
two. If no form is clearly more likely than the others, give ["any"]. Judge from the wording only; do not answer.
Reply with JSON only: {"forms": [[...], ...]}, one list per query, in the order of the queries."""

EXAMPLES = [
    ("What is the difference between the revenue of the Asia segment in 2019 and in 2021?",
     ["revenue of the Asia segment in 2019", "revenue of the Asia segment in 2021"],
     [["Table", "Chart"], ["Table", "Chart"]]),
    ("According to the bar chart on employee satisfaction, which department scored lowest?",
     ["bar chart of employee satisfaction by department"], [["Chart"]]),
    ("What colour is the logo on the cover page?", ["logo on the cover page"], [["Figure"]]),
    ("What is the title of the section that describes the refund policy?",
     ["section heading of the refund policy"], [["Layout", "Text"]]),
    ("Which city is home to the headquarters of the company that acquired Brightline in 2015?",
     ["company that acquired Brightline in 2015", "headquarters city of the company that acquired Brightline"],
     [["Text"], ["Text"]]),
    ("How many people appear in the photos of the volunteer section?", ["photos in the volunteer section"],
     [["Figure"]]),
    ("What does the report say about the outlook for next year?", ["outlook for next year"], [["any"]]),
]


def query_block(question, needs):
    return f"Question: {question.strip()}\nQueries:\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(needs, 1))


def messages(question, needs):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, ns, forms in EXAMPLES:
        msgs += [{"role": "user", "content": query_block(q, ns)},
                 {"role": "assistant", "content": json.dumps({"forms": forms})}]
    msgs.append({"role": "user", "content": query_block(question, needs)})
    return msgs


def parse(reply, m):
    """-> (forms per need, ok). A need's forms are a list of FORMS, or ["any"]."""
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    hit = re.search(r"\{.*\}", text, re.S)
    try:
        forms = json.loads(hit.group())["forms"]
        if len(forms) == m:
            out = []
            for f in forms:
                f = [x for x in (f if isinstance(f, list) else [f]) if x in FORMS][:MAX_FORMS]
                out.append(f or ["any"])
            return out, True
    except (AttributeError, ValueError, KeyError, TypeError):
        pass
    return [["any"]] * m, False


def main(limit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qs = test_questions(load_qa()).set_index("qid")
    decomp = read_jsonl(OUT / "decomp.jsonl")
    out_file = OUT / "need_forms.jsonl"
    done = {r["qid"] for r in read_jsonl(out_file)}
    todo = [r for r in decomp if r["qid"] not in done][: limit or None]
    print(f"{len(decomp)} decomposed questions, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    failed = 0
    with out_file.open("a") as f:
        for k in range(0, len(todo), BATCH):
            batch = todo[k:k + BATCH]
            prompts = [tok.apply_chat_template(messages(qs.loc[r["qid"]].question, r["needs"]), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=120, do_sample=False)
            replies = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, reply in zip(batch, replies):
                forms, ok = parse(reply, len(r["needs"]))
                failed += not ok
                f.write(json.dumps({"qid": r["qid"], "needs": r["needs"], "forms": forms, "parsed": ok, "raw": reply},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"labelled {k + len(batch)}/{len(todo)} (unparsed so far: {failed})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    main(ap.parse_args().limit)
