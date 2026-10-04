"""Step 2: index-time content inventory over every page of the sampled documents.

  CUDA_VISIBLE_DEVICES=0 python inventory.py run --shard 0 --nshards 2   # GPU, one process per GPU
  CUDA_VISIBLE_DEVICES=1 python inventory.py run --shard 1 --nshards 2
  python inventory.py aggregate   # CPU: merges shards into pages.jsonl, writes docs.json and check.txt

Each shard writes out/inventory/pages.shard<k>.jsonl. The run is resumable: pages already written are skipped.
"""
import argparse
import json
import math
import re
from collections import Counter

import pandas as pd
from PIL import Image

from common import (MIN_TEXT_WORDS, OUT, SOURCE_TO_TYPE, build_pool, doc_texts, load_qa, n_words,
                    page_image_path)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 1024 * 1024  # image budget of about 1 megapixel
PATCH = 28
INV_DIR = OUT / "inventory"
PAGES_FILE = INV_DIR / "pages.jsonl"

CHART_SUBTYPES = ["bar", "horizontal bar", "grouped bar", "stacked bar", "line", "pie or donut",
                  "scatter", "area", "other"]
FIGURE_SUBTYPES = ["photo", "diagram or flowchart", "screenshot", "illustration or icon", "map", "other"]
ELEMENT_TYPES = ["text", "table", "chart", "figure", "other"]

PROMPT = f"""List every content element on this document page.

Element types:
- chart: a plot of data. Subtypes: {", ".join(CHART_SUBTYPES)}.
- table: a grid of cells.
- figure: a picture. Subtypes: {", ".join(FIGURE_SUBTYPES)}.
- text: paragraphs or lists.
- other: anything else (page numbers, logos, headers, footers).

For each element give its type, a subtype (for chart and figure, from the lists above; otherwise ""), its title or caption if one is visible (otherwise ""), and optionally its bounding box in pixel coordinates of this image as [x0, y0, x1, y1].
Group the paragraphs of one column or section into a single text element, and use the section heading as its title. Never copy body text into a title. Titles have at most 15 words. List at most 20 elements.
Also describe the page style (colours and fonts) in at most 10 words.

Answer with JSON only, in this format:
{{"elements": [{{"type": "chart", "subtype": "bar", "title": "...", "bbox_2d": [x0, y0, x1, y1]}}], "style": "..."}}"""

STRICT_PROMPT = PROMPT + """

Your previous answer was not valid JSON. Output a single JSON object and nothing else: no markdown fences, no comments, no text before or after it. Use double quotes. Keep at most 30 elements."""

OCR_PROMPT = "Transcribe all visible text on this page, in reading order. Output only the text."


# ---------- image and parsing helpers ----------

def model_image(path):
    """Resize so that pixels <= MAX_PIXELS and both sides are multiples of 28.
    The processor then leaves the image unchanged, so boxes are in this image's pixels."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, math.sqrt(MAX_PIXELS / (w * h)))
    nw = max(PATCH, int(w * scale // PATCH) * PATCH)
    nh = max(PATCH, int(h * scale // PATCH) * PATCH)
    return img.resize((nw, nh), Image.BICUBIC)


def extract_json(raw):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("elements"), list):
        return None
    return obj


def norm_subtype(etype, sub):
    s = (sub or "").lower()
    if etype == "chart":
        rules = [("pie", "pie or donut"), ("donut", "pie or donut"), ("doughnut", "pie or donut"),
                 ("horizontal", "horizontal bar"), ("grouped", "grouped bar"), ("clustered", "grouped bar"),
                 ("stacked", "stacked bar"), ("scatter", "scatter"), ("bubble", "scatter"), ("area", "area"),
                 ("line", "line"), ("bar", "bar"), ("column", "bar"), ("histogram", "bar")]
    elif etype == "figure":
        rules = [("photo", "photo"), ("diagram", "diagram or flowchart"), ("flow", "diagram or flowchart"),
                 ("screenshot", "screenshot"), ("screen", "screenshot"), ("illustration", "illustration or icon"),
                 ("icon", "illustration or icon"), ("logo", "illustration or icon"), ("map", "map")]
    else:
        return ""
    for key, val in rules:
        if key in s:
            return val
    return "other"


def clean(obj, size):
    """Normalise types and subtypes, and convert boxes to [0, 1] using the size the model saw."""
    w, h = size
    elements = []
    for e in obj.get("elements", []):
        if not isinstance(e, dict):
            continue
        t = str(e.get("type", "")).lower().strip()
        t = t if t in ELEMENT_TYPES else "other"
        box = e.get("bbox_2d")
        nbox = None
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x0, y0, x1, y1 = box
            nbox = [round(min(max(v, 0.0), 1.0), 4) for v in (x0 / w, y0 / h, x1 / w, y1 / h)]
        elements.append({"type": t, "subtype": norm_subtype(t, e.get("subtype")),
                         "subtype_raw": str(e.get("subtype", "")), "title": str(e.get("title", "")),
                         "bbox": nbox})
    return {"elements": elements, "style": str(obj.get("style", ""))}


def tesseract_ocr(img_path):
    """Returns (text, readable). Readable: at least 5 words and mean confidence >= 70."""
    import pytesseract
    d = pytesseract.image_to_data(Image.open(img_path), output_type=pytesseract.Output.DICT)
    words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
    text = pytesseract.image_to_string(Image.open(img_path))
    if len(words) < 5:
        return text, False
    return text, sum(c for _, c in words) / len(words) >= 70


# ---------- backends ----------

class VLLMBackend:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=MODEL, max_model_len=8192, limit_mm_per_prompt={"image": 1},
                       mm_processor_kwargs={"min_pixels": PATCH * PATCH, "max_pixels": MAX_PIXELS},
                       gpu_memory_utilization=0.85, seed=0)
        self.SamplingParams = SamplingParams

    def generate(self, images, prompts, max_tokens):
        sp = self.SamplingParams(temperature=0.0, max_tokens=max_tokens)
        msgs = [[{"role": "user", "content": [{"type": "image_pil", "image_pil": im},
                                               {"type": "text", "text": p}]}] for im, p in zip(images, prompts)]
        outs = self.llm.chat(msgs, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class HFBackend:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL, size={"shortest_edge": PATCH * PATCH, "longest_edge": MAX_PIXELS})
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    def generate(self, images, prompts, max_tokens):
        convs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": p}]}]
                 for im, p in zip(images, prompts)]
        inputs = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt", padding=True)
        # model_image() already fits the budget, so the processor must not resize: boxes stay in our pixels.
        for im, (_, gh, gw) in zip(images, inputs["image_grid_thw"].tolist()):
            assert (gw * 14, gh * 14) == im.size, (im.size, gh, gw)
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(out, skip_special_tokens=True)


# ---------- run ----------

def sampled_docs():
    return (OUT / "sample_docs.txt").read_text().split()


def shard_files():
    return sorted(INV_DIR.glob("pages.shard*.jsonl"))


def run(backend_name, limit, batch, shard, nshards):
    INV_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    for path in shard_files():
        for line in path.open():
            r = json.loads(line)
            done.add((r["doc_id"], r["page"]))

    todo, k = [], 0
    for d in sampled_docs():
        for i, text in enumerate(doc_texts(d)):
            if k % nshards == shard and (d, i + 1) not in done:
                todo.append((d, i + 1, n_words(text)))
            k += 1
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} pages done, {len(todo)} to do", flush=True)
    if not todo:
        return

    be = VLLMBackend() if backend_name == "vllm" else HFBackend()
    with (INV_DIR / f"pages.shard{shard}.jsonl").open("a") as f:
        for k in range(0, len(todo), batch):
            chunk = todo[k:k + batch]
            imgs = [model_image(page_image_path(d, p)) for d, p, _ in chunk]
            raws = be.generate(imgs, [PROMPT] * len(chunk), 1024)
            parsed = [extract_json(r) for r in raws]
            retry = [i for i, o in enumerate(parsed) if o is None]
            retry_raw = {}
            if retry:
                outs = be.generate([imgs[i] for i in retry], [STRICT_PROMPT] * len(retry), 1536)
                for i, r in zip(retry, outs):
                    retry_raw[i] = r
                    parsed[i] = extract_json(r)

            # OCR for pages with almost no text layer: tesseract if readable, else the VLM.
            ocr = {}
            vlm_ocr = []
            for i, (d, p, nw) in enumerate(chunk):
                if nw >= MIN_TEXT_WORDS:
                    ocr[i] = ("", "text_layer")
                    continue
                text, ok = tesseract_ocr(page_image_path(d, p))
                if ok:
                    ocr[i] = (text, "tesseract")
                else:
                    vlm_ocr.append(i)
            if vlm_ocr:
                outs = be.generate([imgs[i] for i in vlm_ocr], [OCR_PROMPT] * len(vlm_ocr), 1536)
                for i, t in zip(vlm_ocr, outs):
                    ocr[i] = (t, "vlm")

            for i, (d, p, nw) in enumerate(chunk):
                rec = {"doc_id": d, "page": p, "model": MODEL, "image_size": list(imgs[i].size),
                       "text_layer_words": nw, "raw": raws[i], "raw_retry": retry_raw.get(i),
                       "parse_ok": parsed[i] is not None,
                       "parsed": clean(parsed[i], imgs[i].size) if parsed[i] is not None else None,
                       "ocr_text": ocr[i][0], "ocr_method": ocr[i][1]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{k + len(chunk)}/{len(todo)} pages, retries {len(retry)}, vlm ocr {len(vlm_ocr)}", flush=True)


# ---------- aggregate and check ----------

STYLE_STOP = {"and", "with", "the", "a", "an", "of", "on", "in", "font", "fonts", "text", "colour", "colours",
              "color", "colors", "background", "style", "page", "mostly", "uses", "use", "using", "accents",
              "accent", "layout", "headings", "heading", "body", "title", "titles", "for", "is", "are", "typeface"}


def load_pages():
    """Merge the shard files into pages.jsonl, keeping the last record per page."""
    rows = [json.loads(line) for path in shard_files() for line in path.open()]
    pages = {(r["doc_id"], r["page"]): r for r in rows}
    with PAGES_FILE.open("w") as f:
        for key in sorted(pages):
            f.write(json.dumps(pages[key], ensure_ascii=False) + "\n")
    return pages


def aggregate():
    pages = load_pages()
    docs = {}
    for (d, p), r in sorted(pages.items()):
        a = docs.setdefault(d, {"n_pages": 0, "parse_failures": 0, "pages_with_type": Counter(),
                                "chart_subtypes": Counter(), "figure_subtypes": Counter(), "style": Counter()})
        a["n_pages"] += 1
        if not r["parse_ok"]:
            a["parse_failures"] += 1
            continue
        els = r["parsed"]["elements"]
        for t in {e["type"] for e in els}:
            a["pages_with_type"][t] += 1
        for e in els:
            if e["type"] == "chart":
                a["chart_subtypes"][e["subtype"]] += 1
            elif e["type"] == "figure":
                a["figure_subtypes"][e["subtype"]] += 1
        words = set(re.findall(r"[a-z][a-z\-]+", r["parsed"]["style"].lower())) - STYLE_STOP
        a["style"].update(words)

    out = {}
    for d, a in docs.items():
        out[d] = {"n_pages": a["n_pages"], "parse_failures": a["parse_failures"],
                  "pages_with_type": dict(a["pages_with_type"].most_common()),
                  "chart_subtypes": dict(a["chart_subtypes"].most_common()),
                  "figure_subtypes": dict(a["figure_subtypes"].most_common()),
                  "style_words": [w for w, _ in a["style"].most_common(5)]}
    (INV_DIR / "docs.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    check(pages)


def check(pages):
    """For every pool question in the sampled documents: does the inventory list the gold type on the gold page?
    The inventory has no layout type, so layout gold counts as found when text or other is listed."""
    pool = build_pool(load_qa())
    pool = pool[pool["doc_id"].isin(set(sampled_docs()))]
    rows = []
    for _, q in pool.iterrows():
        r = pages.get((q["doc_id"], q["gold_page"]))
        if r is None:
            continue
        types = {e["type"] for e in r["parsed"]["elements"]} if r["parse_ok"] else set()
        want = {"layout": {"text", "other"}}.get(q["gold_type"], {q["gold_type"]})
        rows.append({"gold_type": q["gold_type"], "parse_ok": r["parse_ok"], "found": bool(types & want)})
    df = pd.DataFrame(rows)
    n_fail = sum(not r["parse_ok"] for r in pages.values())
    methods = Counter(r["ocr_method"] for r in pages.values())
    lines = [f"model: {MODEL}",
             f"pages: {len(pages)}, parse failures after retry: {n_fail} ({n_fail / len(pages):.1%})",
             f"OCR method per page: {dict(methods)}",
             "",
             f"Gold type listed on gold page, pool questions in sampled docs ({len(df)} questions):",
             "type     n  found  recall"]
    for t, g in df.groupby("gold_type"):
        lines.append(f"{t:7s} {len(g):3d}  {g['found'].sum():5d}  {g['found'].mean():.2f}")
    lines.append(f"{'all':7s} {len(df):3d}  {df['found'].sum():5d}  {df['found'].mean():.2f}")
    lines.append("(layout counts as found when text or other is listed; failed parses count as not found)")
    (INV_DIR / "check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "aggregate"])
    ap.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.backend, args.limit, args.batch, args.shard, args.nshards)
    else:
        aggregate()
