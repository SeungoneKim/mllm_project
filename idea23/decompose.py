"""Step 1 (GPU): decompose every test question into at most 4 atomic information needs with an open LLM.

  python decompose.py [--limit N]        # writes out/decomp.jsonl, one row per question; resumable

The needs are the sub-queries d_1..d_m of the coverage objective (selection.py) and of the decomposition baseline.
The few-shot examples are written by hand and do not come from MMLongBench-Doc, so nothing is fit to the test set.
A reply that does not parse falls back to the question itself (m = 1), and is counted in the log.
"""
import argparse
import json
import os
import re

from common import OUT, load_qa, read_jsonl, test_questions

MODEL = os.environ.get("DECOMP_MODEL", "Qwen/Qwen3-8B")
MAX_NEEDS = 4
BATCH = 32

SYSTEM = """You help a search engine find the pages of a long document (a report, paper, slide deck, brochure, \
guidebook or form) that are needed to answer a question. The engine compares one search query at a time with every \
page, using the page's words and its visual content (text, tables, charts, figures, layout).

Break the question into the distinct pieces of information that must be looked up, and write one short search query \
for each piece.
- One query per separate fact, entity, group, time point or quantity that the answer needs. If the question compares, \
combines, lists, counts or computes over several things, each of them gets its own query.
- If one piece is needed to find another (first identify something, then look up a property of it), write a query \
for each step.
- Each query must stand on its own: repeat the names, dates and terms from the question, and use no pronouns.
- Do not answer the question and do not invent names, numbers or facts that are not in the question.
- Use at most 4 queries. If the question needs a single piece of information, return exactly one query.
Reply with JSON only: {"queries": ["...", "..."]}"""

EXAMPLES = [
    ("What is the difference between the revenue of the Asia segment in 2019 and in 2021?",
     ["revenue of the Asia segment in 2019", "revenue of the Asia segment in 2021"]),
    ("Which city is home to the headquarters of the company that acquired Brightline in 2015?",
     ["company that acquired Brightline in 2015", "headquarters city of the company that acquired Brightline"]),
    ("What percentage of respondents aged 18-29 say they use Instagram?",
     ["percentage of respondents aged 18-29 who use Instagram"]),
    ("List the two warning signs shown in the safety section and the colour of the alarm light.",
     ["warning signs in the safety section", "colour of the alarm light"]),
    ("How many bar charts in the report show data from 2010?",
     ["bar charts showing data from 2010"]),
]


def messages(question):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, needs in EXAMPLES:
        msgs += [{"role": "user", "content": f"Question: {q}"},
                 {"role": "assistant", "content": json.dumps({"queries": needs})}]
    msgs.append({"role": "user", "content": f"Question: {question.strip()}"})
    return msgs


def parse(reply, question):
    """-> (needs, ok). Strips a thinking block if the model wrote one."""
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        needs = json.loads(m.group())["queries"]
        needs = [str(n).strip() for n in needs if str(n).strip()]
        if needs:
            return needs[:MAX_NEEDS], True
    except (AttributeError, ValueError, KeyError, TypeError):
        pass
    return [question.strip()], False


def main(limit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qs = test_questions(load_qa())
    out_file = OUT / "decomp.jsonl"
    done = {r["qid"] for r in read_jsonl(out_file)}
    todo = qs[~qs.qid.isin(done)].head(limit or None)
    print(f"{len(qs)} test questions, {len(done)} done, {len(todo)} to go", flush=True)
    if todo.empty:
        return

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    with out_file.open("a") as f:
        rows = list(todo.itertuples())
        for k in range(0, len(rows), BATCH):
            batch = rows[k:k + BATCH]
            prompts = [tok.apply_chat_template(messages(r.question), tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=200, do_sample=False)
            replies = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, reply in zip(batch, replies):
                needs, ok = parse(reply, r.question)
                failed += not ok
                f.write(json.dumps({"qid": int(r.qid), "question": r.question, "needs": needs, "parsed": ok,
                                    "raw": reply}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"decomposed {k + len(batch)}/{len(rows)} (unparsed so far: {failed})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    main(ap.parse_args().limit)
