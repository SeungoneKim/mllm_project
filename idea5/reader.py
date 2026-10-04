"""Part B, step 2 (GPU): Qwen2.5-VL-7B answers every sampled question under 9 input conditions.

  CUDA_VISIBLE_DEVICES=0 python reader.py --shard 0 --nshards 2   # reads out/sample/items.jsonl, cache/sample/
                                                                   # writes out/reader/{transcripts,answers}_<shard>.jsonl

Conditions (the gold page only, no retrieval):
  Q      question only
  T      text layer                    X      markdown transcription of the page by the same VLM (no image)
  I_hi   image, 1280 tokens            I_lo   image, 256 tokens
  TI_hi  text layer + image (1280)     TI_lo  text layer + image (256)
  M_lo   words-masked image (256)      TM_lo  text layer + words-masked image (256)
Per condition: the greedy short answer, the mean log-probability of the reference answer (teacher forced), and the
input tokens. Resumable: rows already written are skipped.
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image

from common import CACHE, OUT

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
READER = Path(os.environ.get("IDEA5_READER", OUT / "reader"))  # smoke tests write elsewhere
IMG = CACHE / "sample"
BATCH = 8
MAX_TEXT_CHARS = 12000
CONDS = {"Q": (None, None), "T": (None, "text"), "X": (None, "transcript"),
         "I_hi": ("hi", None), "I_lo": ("lo", None), "TI_hi": ("hi", "text"), "TI_lo": ("lo", "text"),
         "M_lo": ("mlo", None), "TM_lo": ("mlo", "text")}
FORMAT = {"Int": "an integer", "Float": "a number", "Str": "a short phrase",
          "List": "a Python list of strings, e.g. ['first', 'second']"}
TRANSCRIBE = ("Transcribe this document page into Markdown. Copy all text in reading order. Write tables as Markdown "
              "tables. For every chart, give its title, axis labels, legend and every data value you can read. For "
              "every figure or photo, describe what it shows in one or two sentences. Output only the transcription.")


def intro(img, txt):
    if img and txt:
        what = "an image of one page of a document" + (", with its text painted out," if img == "mlo" else "")
        return f"Below are {what} and the text extracted from the same page."
    if img:
        return "Below is an image of one page of a document" + (", with its text painted out." if img == "mlo" else ".")
    if txt == "text":
        return "Below is the text extracted from one page of a document."
    if txt == "transcript":
        return "Below is a transcription of one page of a document."
    return "Answer a question about a document. The document is not shown; give your best guess."


def conversation(item, cond, transcript):
    img, txt = CONDS[cond]
    content = []
    if img:
        content.append({"type": "image", "image": Image.open(IMG / f"{item['qid']}_{img}.png").convert("RGB")})
    body = intro(img, txt)
    if txt:
        t = (item["text"] if txt == "text" else transcript) or "(no text could be extracted from this page)"
        body += "\n\n<<<\n" + t[:MAX_TEXT_CHARS] + "\n>>>"
    body += (f"\n\nQuestion: {item['question'].strip()}\nAnswer with only {FORMAT[item['answer_format']]}, "
             "no explanation. If you are unsure, give your best guess.")
    content.append({"type": "text", "text": body})
    return [{"role": "user", "content": content}]


class Reader:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        # Images are resized beforehand (common.fit); the processor bounds are wide enough to leave them as they are.
        self.proc = AutoProcessor.from_pretrained(MODEL, size={"shortest_edge": 4 * 28 * 28, "longest_edge": 1400 * 28 * 28})
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        self.image_token = self.model.config.image_token_id

    def encode(self, convs, generation_prompt=True):
        return self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=generation_prompt,
                                             return_dict=True, return_tensors="pt", padding=True).to(self.model.device)

    def generate(self, convs, max_tokens):
        inputs = self.encode(convs)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        texts = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        mask = inputs["attention_mask"].bool()
        n_in = mask.sum(dim=1).tolist()
        n_img = ((inputs["input_ids"] == self.image_token) & mask).sum(dim=1).tolist()
        n_out = [len(self.proc.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
        return [t.strip() for t in texts], n_in, n_img, n_out

    def answer_logprob(self, conv, answer):
        """Mean and sum log-probability of the reference answer after the prompt (batch of one)."""
        torch = self.torch
        prompt = self.encode([conv])
        full = self.encode([conv + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]],
                           generation_prompt=False)
        start = prompt["input_ids"].shape[1]
        n = len(self.proc.tokenizer(answer, add_special_tokens=False)["input_ids"])
        ids = full["input_ids"][0, start:start + n]
        with torch.inference_mode():
            logits = self.model(**full).logits[0, start - 1:start - 1 + n].float()
        lp = torch.log_softmax(logits, dim=-1).gather(1, ids[:, None])[:, 0]
        aligned = self.proc.tokenizer.decode(ids).strip() == answer.strip()
        return float(lp.mean()), float(lp.sum()), n, aligned


def read_all(pattern):
    """Rows of every shard's file, so a rerun with another shard count reuses finished work."""
    return [json.loads(l) for p in sorted(READER.glob(pattern)) for l in p.open()]


def load_done(pattern, key):
    return {tuple(r[k] for k in key) for r in read_all(pattern)}


def main(shard, nshards, limit):
    items = [json.loads(l) for l in (OUT / "sample" / "items.jsonl").open()]
    items = [it for i, it in enumerate(items) if i % nshards == shard][: limit or None]
    READER.mkdir(parents=True, exist_ok=True)
    tr_file, ans_file = READER / f"transcripts_{shard}.jsonl", READER / f"answers_{shard}.jsonl"
    reader = Reader()

    # 1. transcriptions (the X condition needs them)
    print(f"shard {shard}/{nshards}: {len(items)} items", flush=True)
    done = load_done("transcripts_*.jsonl", ["qid"])
    todo = [it for it in items if (it["qid"],) not in done]
    with tr_file.open("a") as f:
        for k in range(0, len(todo), BATCH):
            batch = todo[k:k + BATCH]
            convs = [[{"role": "user", "content": [
                {"type": "image", "image": Image.open(IMG / f"{it['qid']}_hi.png").convert("RGB")},
                {"type": "text", "text": TRANSCRIBE}]}] for it in batch]
            texts, _, _, n_out = reader.generate(convs, 2048)
            for it, t, n in zip(batch, texts, n_out):
                f.write(json.dumps({"qid": it["qid"], "transcript": t, "tokens": n}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"transcribed {k + len(batch)}/{len(todo)}", flush=True)
    transcripts = {r["qid"]: r["transcript"] for r in read_all("transcripts_*.jsonl")}

    # 2. answers and reference log-probabilities, one condition at a time
    done = load_done("answers_*.jsonl", ["qid", "cond"])
    with ans_file.open("a") as f:
        for cond in CONDS:
            todo = [it for it in items if (it["qid"], cond) not in done]
            for k in range(0, len(todo), BATCH):
                batch = todo[k:k + BATCH]
                convs = [conversation(it, cond, transcripts.get(it["qid"], "")) for it in batch]
                preds, n_in, n_img, _ = reader.generate(convs, 64)
                for it, conv, pred, ni, nm in zip(batch, convs, preds, n_in, n_img):
                    lp_mean, lp_sum, n_ans, aligned = reader.answer_logprob(conv, it["answer"])
                    f.write(json.dumps({"qid": it["qid"], "cond": cond, "pred": pred, "lp_mean": lp_mean,
                                        "lp_sum": lp_sum, "n_answer_tokens": n_ans, "aligned": aligned,
                                        "input_tokens": ni, "image_tokens": nm}, ensure_ascii=False) + "\n")
                f.flush()
            print(f"{cond}: {len(todo)} answered", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N items of the shard only (smoke test)")
    a = ap.parse_args()
    main(a.shard, a.nshards, a.limit)
