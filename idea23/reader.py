"""Step 4 (GPU): a frozen Qwen2.5-VL-7B reader answers each question from the pages a method selected.

  python reader.py --shard 0 --nshards 4 [--methods topk,coverage] [--ks 3,5] [--limit N]
      reads out/<version>/selections.jsonl, writes out/reader/answers_<version>_<shard>.jsonl (resumable)
      the methods default to the version's `reader` list (versions.py); v0's first run wrote answers_<shard>.jsonl

Every method is read by the same model, prompt and image budget, so only the selected pages differ. A reader job is
a (question, set of pages) pair: methods that select the same pages share one answer, across all versions (the
answers in out/reader/ are one cache). Pages are shown in document
order, each preceded by its page number, at most TOKENS_PER_PAGE visual tokens each. The prompt and short-answer
format follow idea5/reader.py, so answers are scored with idea5/score.py.
"""
import argparse
import json
import os
import re
from pathlib import Path

from common import OUT, VERSION, VOUT, doc_images, fit, load_qa, read_jsonl
from versions import VERSIONS

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
READER = Path(os.environ.get("IDEA23_READER", OUT / "reader"))  # smoke tests write elsewhere
METHODS = VERSIONS[VERSION]["reader"]
KS = [3, 5]
TOKENS_PER_PAGE = 1024
BATCH = 4
FORMAT = {"Int": "an integer", "Float": "a number", "Str": "a short phrase",
          "List": "a Python list of strings, e.g. ['first', 'second']"}


def jobs(methods, ks, sets=("multi", "control")):
    """Unique (qid, pages) pairs, pages sorted, in a fixed order so that sharding is stable."""
    keep = [r for r in read_jsonl(VOUT / "selections.jsonl")
            if r["method"] in methods and (r["method"] == "gold" or r["k"] in ks) and r["set"] in sets]
    return sorted({(r["qid"], tuple(sorted(r["pages"]))) for r in keep})


def conversation(question, fmt, pages, images, reason=False):
    n = len(pages)
    content = [{"type": "text", "text": f"Below {'is' if n == 1 else 'are'} {n} page{'s' if n > 1 else ''} "
                                        f"of a long document, in page order."}]
    for p, img in zip(pages, images):
        content += [{"type": "text", "text": f"Page {p}:"}, {"type": "image", "image": img}]
    if reason:  # ablation: let the reader work through the pages before answering
        ask = (f"Question: {question.strip()}\nFirst reason step by step, briefly, using the pages above. Then write "
               f"the final answer on the last line as 'Answer: <answer>', where <answer> is "
               f"{FORMAT.get(fmt, 'a short phrase')}. If you are unsure, give your best guess.")
    else:
        ask = (f"Question: {question.strip()}\nAnswer with only {FORMAT.get(fmt, 'a short phrase')}, no explanation. "
               "If you are unsure, give your best guess.")
    content.append({"type": "text", "text": ask})
    return [{"role": "user", "content": content}]


def final_answer(text):
    """The answer after the last 'Answer:' marker of a reasoning reply (else its last non-empty line)."""
    hits = re.findall(r"answer\s*[:：]\s*(.+)", text, flags=re.I)
    if hits:
        return hits[-1].strip().strip("*").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


class Reader:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        # Images are resized beforehand (common.fit); the processor bounds are wide enough to leave them as they are.
        self.proc = AutoProcessor.from_pretrained(MODEL, size={"shortest_edge": 4 * 28 * 28,
                                                                "longest_edge": 1400 * 28 * 28})
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        self.image_token = self.model.config.image_token_id

    def generate(self, convs, max_tokens=96):
        inputs = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        texts = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        mask = inputs["attention_mask"].bool()
        n_in = mask.sum(dim=1).tolist()
        n_img = ((inputs["input_ids"] == self.image_token) & mask).sum(dim=1).tolist()
        return [t.strip() for t in texts], n_in, n_img


def main(shard, nshards, methods, ks, sets, reason, limit):
    READER.mkdir(parents=True, exist_ok=True)
    qa = load_qa().set_index("qid")
    todo = [j for i, j in enumerate(jobs(methods, ks, sets)) if i % nshards == shard]
    done = {(r["qid"], tuple(r["pages"])) for p in READER.glob("answers_*.jsonl") for r in read_jsonl(p)}
    todo = [j for j in todo if j not in done][: limit or None]
    print(f"shard {shard}/{nshards}: {len(todo)} reader jobs to run ({MODEL})", flush=True)
    if not todo:
        return
    reader = Reader()
    with (READER / f"answers_{VERSION}_{shard}.jsonl").open("a") as f:
        for k in range(0, len(todo), BATCH):
            batch = todo[k:k + BATCH]
            convs = []
            for qid, pages in batch:
                row = qa.loc[qid]
                images = [fit(im, TOKENS_PER_PAGE) for im in doc_images(row.doc_id, pages)]
                convs.append(conversation(row.question, row.answer_format, pages, images, reason))
            preds, n_in, n_img = reader.generate(convs, 512 if reason else 96)
            for (qid, pages), pred, ni, nm in zip(batch, preds, n_in, n_img):
                out = {"qid": qid, "pages": list(pages), "pred": final_answer(pred) if reason else pred,
                       "input_tokens": ni, "image_tokens": nm}
                if reason:
                    out["raw"] = pred
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            print(f"answered {k + len(batch)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--sets", default="multi,control")
    ap.add_argument("--reason", action="store_true",
                    help="ablation: reason before answering (write to another IDEA23_READER, e.g. out/reader_cot)")
    ap.add_argument("--limit", type=int, default=0, help="first N jobs of the shard only (smoke test)")
    a = ap.parse_args()
    main(a.shard, a.nshards, a.methods.split(","), [int(k) for k in a.ks.split(",")], a.sets.split(","), a.reason,
         a.limit)
