"""Index step (GPU, v1 and v2): split every page of the test documents into evidence elements.

  python segment.py [--shard i --nshards n] [--docs a.pdf,b.pdf] [--draw N]
      writes $IDEA23_CACHE/elements/<doc>.json (resumable); --draw saves N annotated pages per document for a look

An element is one region of one page holding evidence in one form, in reading order:
  {"id": index in the document, "page": 1-indexed, "form": Text | Layout | Table | Chart | Figure | Page,
   "label": detector class, "bbox": [x0, y0, x1, y1] in PDF points, "score": detector confidence,
   "text": the PDF text layer inside the box (empty for scans and pictures)}
The forms are the five evidence sources of MMLongBench-Doc. PP-DocLayoutV3 (PaddlePaddle, Apache-2.0) finds the
regions on the rendered page. Its image regions are passed to Docling's figure classifier
(DocumentFigureClassifier-v2.5, MIT), and those it calls a data plot become Chart. Captions are merged into the
table, chart or figure they label; page numbers, running headers and footers are dropped, and so is a region that
lies mostly inside a table, chart or figure. A page with no region left becomes one element of form Page.
v1 uses the set of forms on each page; v2 retrieves the elements themselves.
"""
import argparse
import json

import numpy as np
import pymupdf
import torch

from common import BAD_DOCS, DOC_DIR, ELEMENTS, crop, doc_stem, load_qa, render, test_questions, zoom

DETECTOR = "PaddlePaddle/PP-DocLayoutV3_safetensors"
FIGURES = "docling-project/DocumentFigureClassifier-v2.5"
THRESHOLD = 0.4
BATCH = 8
LABEL_FORM = {
    "text": "Text", "abstract": "Text", "content": "Text", "reference": "Text", "reference_content": "Text",
    "algorithm": "Text", "formula": "Text", "footnote": "Text", "aside_text": "Text",
    "doc_title": "Layout", "paragraph_title": "Layout",
    "table": "Table", "chart": "Chart", "image": "Figure", "seal": "Figure",
    "figure_title": "Caption", "vision_footnote": "Caption",
    "header": None, "footer": None, "number": None, "formula_number": None,
}
PLOTS = {"bar_chart", "line_chart", "pie_chart", "scatter_plot", "box_plot"}
VISUAL = {"Table", "Chart", "Figure"}
CAPTION_GAP = 0.08  # a caption joins a visual element at most this fraction of the page height away
CONTAINED = 0.8     # a region with this share of its area inside a visual element is part of it


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def clean(regions, page_h):
    """Detected regions of one page (reading order) -> elements: drop noise, merge captions into visuals."""
    regs = [r for r in regions if LABEL_FORM.get(r["label"]) is not None and area(r["bbox"]) > 0]
    for r in regs:
        r["form"] = LABEL_FORM[r["label"]]
    # duplicates and regions inside a visual element
    keep = []
    for r in regs:
        inside = any(o is not r and o["form"] in VISUAL and area(o["bbox"]) > area(r["bbox"])
                     and inter(r["bbox"], o["bbox"]) >= CONTAINED * area(r["bbox"]) for o in regs)
        dup = any(o["form"] == r["form"] and inter(r["bbox"], o["bbox"]) >= CONTAINED * area(r["bbox"])
                  and (o["score"], -regs.index(o)) > (r["score"], -regs.index(r)) for o in keep)
        if not inside and not dup:
            keep.append(r)
    visuals = [r for r in keep if r["form"] in VISUAL]
    out = []
    for r in keep:
        if r["form"] != "Caption":
            out.append(r)
            continue
        best, best_gap = None, CAPTION_GAP * page_h
        for v in visuals:
            a, b = r["bbox"], v["bbox"]
            overlap = min(a[2], b[2]) - max(a[0], b[0])
            if overlap < 0.3 * min(a[2] - a[0], b[2] - b[0]):
                continue
            gap = max(b[1] - a[3], a[1] - b[3], 0)
            if gap <= best_gap:
                best, best_gap = v, gap
        if best is None:
            r["form"] = "Text"
            out.append(r)
        else:
            best["bbox"] = union(best["bbox"], r["bbox"])
            best["caption"] = (best.get("caption", "") + " " + r["text"]).strip()
    return out


class Segmenter:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModelForObjectDetection
        self.proc = AutoImageProcessor.from_pretrained(DETECTOR)
        self.det = AutoModelForObjectDetection.from_pretrained(DETECTOR).cuda().eval()
        self.labels = self.det.config.id2label
        self.fproc = AutoImageProcessor.from_pretrained(FIGURES)
        self.fig = AutoModelForImageClassification.from_pretrained(FIGURES).cuda().eval()

    @torch.inference_mode()
    def detect(self, images):
        inputs = self.proc(images=images, return_tensors="pt").to("cuda")
        outputs = self.det(**inputs)
        sizes = [(im.height, im.width) for im in images]
        res = self.proc.post_process_object_detection(outputs, threshold=THRESHOLD, target_sizes=sizes)
        return [[{"label": self.labels[int(l)], "score": round(float(s), 3), "box": b.tolist()}
                 for s, l, b in zip(r["scores"], r["labels"], r["boxes"])] for r in res]

    @torch.inference_mode()
    def figure_classes(self, crops):
        if not crops:
            return []
        inputs = self.fproc(images=crops, return_tensors="pt").to("cuda")
        probs = self.fig(**inputs).logits.softmax(-1)
        p, i = probs.max(-1)
        return [(self.fig.config.id2label[int(k)], float(v)) for k, v in zip(i, p)]


def segment_doc(seg, doc_id, draw=0):
    elements, drawn = [], []
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        for k in range(0, doc.page_count, BATCH):
            pages = [doc[i] for i in range(k, min(k + BATCH, doc.page_count))]
            images = [render(p) for p in pages]
            for page, img, dets in zip(pages, images, seg.detect(images)):
                z = zoom(page)
                regions = []
                for d in dets:
                    bbox = [round(v / z, 1) for v in d["box"]]
                    text = page.get_text("text", clip=pymupdf.Rect(bbox)).strip()
                    regions.append({"label": d["label"], "score": d["score"], "bbox": bbox, "text": text[:4000]})
                els = clean(regions, page.rect.height)
                figs = [e for e in els if e["form"] == "Figure"]
                for e, (cls, p) in zip(figs, seg.figure_classes([crop(img, e["bbox"], z) for e in figs])):
                    e["figure_class"] = cls
                    if cls in PLOTS and p >= 0.5:
                        e["form"] = "Chart"
                if not els:
                    r = page.rect
                    els = [{"label": "page", "score": 1.0, "bbox": [0, 0, round(r.width, 1), round(r.height, 1)],
                            "text": page.get_text("text").strip()[:4000], "form": "Page"}]
                for e in els:
                    if e.get("caption"):
                        e["text"] = (e["caption"] + "\n" + e["text"]).strip()
                    elements.append({"id": len(elements), "page": page.number + 1, "form": e["form"],
                                     "label": e["label"], "bbox": e["bbox"], "score": e["score"], "text": e["text"],
                                     **({"figure_class": e["figure_class"]} if "figure_class" in e else {})})
                if len(drawn) < draw:
                    drawn.append((page.number + 1, img, z, [x for x in elements if x["page"] == page.number + 1]))
    return elements, drawn


def save_drawing(doc_id, page, img, z, els):
    from PIL import ImageDraw
    colors = {"Text": "blue", "Layout": "purple", "Table": "green", "Chart": "red", "Figure": "orange",
              "Page": "gray"}
    img = img.copy()
    d = ImageDraw.Draw(img)
    for e in els:
        b = [v * z for v in e["bbox"]]
        d.rectangle(b, outline=colors[e["form"]], width=3)
        d.text((b[0] + 3, b[1] + 2), f"{e['id']} {e['form']}", fill=colors[e["form"]])
    out = ELEMENTS.parent / "elements_draw"
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / f"{doc_stem(doc_id)}_p{page}.jpg", quality=80)


def main(shard, nshards, docs, draw):
    if docs is None:
        docs = sorted(set(test_questions(load_qa()).doc_id) - BAD_DOCS)[shard::nshards]
    ELEMENTS.mkdir(parents=True, exist_ok=True)
    todo = [d for d in docs if not (ELEMENTS / f"{doc_stem(d)}.json").exists()]
    print(f"shard {shard}/{nshards}: {len(docs)} documents, {len(todo)} to segment", flush=True)
    if not todo:
        return
    seg = Segmenter()
    for n, d in enumerate(todo, 1):
        els, drawn = segment_doc(seg, d, draw)
        (ELEMENTS / f"{doc_stem(d)}.json").write_text(json.dumps({"doc_id": d, "elements": els}, ensure_ascii=False))
        for args in drawn:
            save_drawing(d, *args)
        forms = {f: sum(e["form"] == f for e in els) for f in ["Text", "Layout", "Table", "Chart", "Figure", "Page"]}
        pages = max(e["page"] for e in els)
        print(f"[{n}/{len(todo)}] {d}: {pages} pages, {len(els)} elements {forms}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--docs", default=None, help="comma-separated document names (smoke test)")
    ap.add_argument("--draw", type=int, default=0, help="save this many annotated pages per document")
    a = ap.parse_args()
    main(a.shard, a.nshards, a.docs.split(",") if a.docs else None, a.draw)
