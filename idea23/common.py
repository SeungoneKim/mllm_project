"""Shared paths, data loading, evaluation sets and page rendering for the merged Idea 2+3 baseline.

Paths can be overridden with environment variables (see README.md). Definitions follow idea2/common.py (Analysis 2):
same BAD_DOCS, same modality names, same qid (row index of the Hugging Face parquet, also the index of
idea1/scores.pkl), so the numbers here line up with the data analysis report.
"""
import ast
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image

IDEA23 = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MLLM_ROOT", IDEA23.parents[1]))  # <ROOT>/<repo>/idea23, as in idea4 and idea5
MMLB = Path(os.environ.get("MMLB_DIR", ROOT / "data" / "mmlongbench"))
QA_FILE = MMLB / "data" / "train-00000-of-00001.parquet"
DOC_DIR = MMLB / "documents"
IDEA1_SCORES = IDEA23.parent / "idea1" / "scores.pkl"  # BM25 page scores (and the original ColQwen2.5 run)
ANNOTATIONS = IDEA23 / "idea2_data_analysis" / "artifacts" / "annotations.csv"  # partition / composition labels
OUT = Path(os.environ.get("IDEA23_OUT", IDEA23 / "out"))  # inputs shared by all versions, and the reader's answers
CACHE = Path(os.environ.get("IDEA23_CACHE", IDEA23 / "cache"))  # page and element embeddings, element index
VERSION = os.environ.get("IDEA23_VERSION", "v0")  # which pipeline version (versions.py); pipeline.py sets it
VOUT = OUT / VERSION  # the selections, metrics and analysis of that version

# The Hugging Face copies of these two PDFs do not match their questions (see idea2/README.md).
BAD_DOCS = {"mi_phone.pdf", "dr-vorapptchapter1emissionsources-121120210508-phpapp02_95.pdf"}

MODALITY = {
    "Pure-text (Plain-text)": "Text",
    "Generalized-text (Layout)": "Layout",
    "Table": "Table",
    "Chart": "Chart",
    "Figure": "Figure",
}

FORMS = ["Text", "Layout", "Table", "Chart", "Figure"]  # evidence forms, in the order of MODALITY
ELEMENTS = CACHE / "elements"  # segment.py: the evidence elements of each document

RENDER_DPI = 144
RENDER_MAX_SIDE = 1600
PATCH = 28  # Qwen2.5-VL: one visual token per 28 x 28 pixels

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


def parse_list(s):
    """evidence_pages / evidence_sources are strings holding Python lists."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return []
    return list(ast.literal_eval(s)) if isinstance(s, str) else list(s)


def page_counts(doc_ids):
    counts = {}
    for d in doc_ids:
        with pymupdf.open(DOC_DIR / d) as doc:
            counts[d] = doc.page_count
    return counts


def load_qa():
    """All QA rows with parsed labels; qid is the row index of the parquet."""
    df = pd.read_parquet(QA_FILE).reset_index(names="qid")
    df["pages"] = df["evidence_pages"].map(parse_list)
    df["sources"] = df["evidence_sources"].map(parse_list)
    df["mods"] = df["sources"].map(lambda l: tuple(sorted({MODALITY[s] for s in l})))
    df["combo"] = df["mods"].map(" + ".join)
    df["n_mods"] = df["mods"].map(len)
    df["gold"] = df["pages"].map(lambda l: sorted(set(l)))
    df["n_gold"] = df["gold"].map(len)
    df["answerable"] = df["answer"] != "Not answerable"
    df["doc_pages"] = df["doc_id"].map(page_counts(df["doc_id"].unique()))
    df["pages_in_pdf"] = [all(1 <= p <= n for p in ps) for ps, n in zip(df["pages"], df["doc_pages"])]
    return df


def eval_sets(df):
    """The two evaluation sets. Nothing is trained or tuned on them, so both are test data.

    Valid labels: every evidence page inside the PDF and at least one evidence-source label (17 answerable
    questions list pages but no source; Analysis 2 left them out by the same rule), outside BAD_DOCS.
    multi    answerable, valid labels, 2-5 distinct evidence pages (310 questions): the main test set, the
             same population as the retrieval check of Analysis 2 (same_mod and cross_multi, at most 5 pages,
             so that a 5-page budget can hold all the evidence).
    control  answerable, valid labels, exactly one evidence page (453): a selector built for multi-page
             questions should not hurt these.
    `stratum` follows Analysis 2: same_mod (one modality), cross_multi (two or more modalities), single.
    """
    ok = df[df.answerable & df.pages_in_pdf & (df.n_gold >= 1) & (df.n_mods >= 1) & ~df.doc_id.isin(BAD_DOCS)].copy()
    ok["stratum"] = np.where(ok.n_gold == 1, "single", np.where(ok.n_mods == 1, "same_mod", "cross_multi"))
    multi = ok[(ok.n_gold >= 2) & (ok.n_gold <= 5)].copy()
    control = ok[ok.n_gold == 1].copy()
    return multi, control


def test_questions(df):
    """multi and control together, with a `set` column."""
    multi, control = eval_sets(df)
    return pd.concat([multi.assign(set="multi"), control.assign(set="control")], ignore_index=True)


def read_jsonl(path):
    path = Path(path)
    return [json.loads(l) for l in path.open()] if path.exists() else []


def zoom(pdf_page):
    """Pixels per PDF point of render()."""
    r = pdf_page.rect
    return min(RENDER_DPI / 72, RENDER_MAX_SIDE / max(r.width, r.height))


def render(pdf_page):
    """Page image at 144 dpi, long side <= 1600 px (same as idea4 and idea5)."""
    zoom_ = zoom(pdf_page)
    pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom_, zoom_), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def fit(img, tokens, patch=PATCH):
    """Resize to at most `tokens` visual tokens (patch x patch px each, 28 for Qwen2.5-VL), sides multiples of the
    patch (as idea5)."""
    w, h = img.size
    scale = min(1.0, math.sqrt(tokens * patch * patch / (w * h)))
    nw = max(patch, int(w * scale // patch) * patch)
    nh = max(patch, int(h * scale // patch) * patch)
    return img.resize((nw, nh), Image.BICUBIC)


def doc_images(doc_id, pages=None):
    """Rendered images of the given 1-indexed pages (all pages when None)."""
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        idx = range(1, doc.page_count + 1) if pages is None else pages
        return [render(doc[p - 1]) for p in idx]


def doc_stem(doc_id):
    return Path(doc_id).stem


def load_elements(doc_id):
    """The evidence elements of one document (segment.py), in page and reading order."""
    return json.loads((ELEMENTS / f"{doc_stem(doc_id)}.json").read_text())["elements"]


def crop(page_img, bbox, zoom_, pad=4, min_side=56):
    """The part of a rendered page inside `bbox` (PDF points), with `pad` points of margin, on a white canvas of at
    least min_side pixels per side."""
    x0, y0, x1, y1 = [round(v * zoom_) for v in (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)]
    w, h = page_img.size
    img = page_img.crop((max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
    if min(img.size) < min_side:
        canvas = Image.new("RGB", (max(img.width, min_side), max(img.height, min_side)), "white")
        canvas.paste(img, ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2))
        img = canvas
    return img
