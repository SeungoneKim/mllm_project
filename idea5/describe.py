"""Step 3b: the untrained describer. A VLM writes the same two descriptions as the annotators, for the same items.

  CUDA_VISIBLE_DEVICES=0 python describe.py     # reads out/human/items.csv, writes out/human/vlm_descriptions.jsonl

Generic: the target page alone. Distinct: the target page followed by its 3 most similar pages. The prompts repeat
the annotators' instructions. Greedy decoding. Set VLM_MODEL to try another model (default: the 7B model used for
idea4's inventory; Qwen/Qwen2.5-VL-3B-Instruct is the planned trainable describer).
"""
import json
import math
import os

import pandas as pd
from PIL import Image

from common import OUT, page_image_path

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
PATCH = 28
PIXELS_ONE = 1024 * 1024  # one page: same budget as idea4/inventory.py
PIXELS_EACH = 640 * 640   # four pages in one prompt
HUMAN = OUT / "human"

GENERIC = ("This image is page {p} of a {n}-page document ({t}). Write 2 to 4 sentences describing this page for a "
           "search index, so that someone looking for the information on it would find it. Output only the description.")
DISTINCT = ("The first image is page {p} of a {n}-page document ({t}). The next three images are the pages of the same "
            "document that look most similar to it (pages {s}). Write 2 to 4 sentences describing the first page for a "
            "search index, so that someone looking for the information on it would find it and a search could tell it "
            "apart from the other three pages. Output only the description of the first page.")


def fit(path, budget):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, math.sqrt(budget / (w * h)))
    nw, nh = max(PATCH, int(w * scale // PATCH) * PATCH), max(PATCH, int(h * scale // PATCH) * PATCH)
    return img.resize((nw, nh), Image.BICUBIC)


class Describer:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL, size={"shortest_edge": PATCH * PATCH, "longest_edge": PIXELS_ONE})
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    def __call__(self, images, prompt, max_tokens=200):
        conv = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] +
                 [{"type": "text", "text": prompt}]}]
        inputs = self.proc.apply_chat_template([conv], tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    items = pd.read_csv(HUMAN / "items.csv")
    out_file = HUMAN / "vlm_descriptions.jsonl"
    done = set()
    if out_file.exists():
        done = {(r["item_id"], r["kind"], r["model"]) for r in map(json.loads, out_file.open())}
    vlm = Describer()
    with out_file.open("a") as f:
        for r in items.itertuples():
            sibs = json.loads(r.neighbours)
            fmt = {"p": r.target, "n": r.n_pages, "t": r.doc_type, "s": ", ".join(map(str, sibs))}
            jobs = {"generic": ([fit(page_image_path(r.doc_id, r.target), PIXELS_ONE)], GENERIC.format(**fmt)),
                    "distinct": ([fit(page_image_path(r.doc_id, p), PIXELS_EACH) for p in [r.target] + sibs],
                                 DISTINCT.format(**fmt))}
            for kind, (images, prompt) in jobs.items():
                if (r.item_id, kind, MODEL) in done:
                    continue
                text = vlm(images, prompt)
                f.write(json.dumps({"item_id": r.item_id, "kind": kind, "model": MODEL, "text": text},
                                   ensure_ascii=False) + "\n")
                f.flush()
                print(r.item_id, kind, text[:100].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
