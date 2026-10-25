"""Shared paths, data loading, page rendering and text layers for Idea 5 (image vs text layer of a page).

Paths can be overridden with environment variables (see README.md). Rendering matches idea4 (144 dpi, long side at
most 1600 px) so page images are comparable across analyses.
"""
import ast
import math
import os
import re
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image

IDEA5 = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MLLM_ROOT", IDEA5.parents[1]))  # <ROOT>/<repo>/idea5, the same convention as idea4
MMLB = Path(os.environ.get("MMLB_DIR", ROOT / "data" / "mmlongbench"))
QA_FILE = MMLB / "data" / "train-00000-of-00001.parquet"
DOC_DIR = MMLB / "documents"
IDEA4 = Path(os.environ.get("IDEA4_DIR", IDEA5.parent / "idea4"))
OUT = Path(os.environ.get("IDEA5_OUT", IDEA5 / "out"))
CACHE = Path(os.environ.get("IDEA5_CACHE", IDEA5 / "cache"))
SEED = 555

SOURCE_TO_TYPE = {
    "Pure-text (Plain-text)": "text",
    "Table": "table",
    "Chart": "chart",
    "Figure": "figure",
    "Generalized-text (Layout)": "layout",
}
TYPES = ["text", "table", "chart", "figure", "layout"]

RENDER_DPI = 144
RENDER_MAX_SIDE = 1600
MIN_TEXT_WORDS = 20  # fewer text-layer words than this: the page has no usable text layer
PATCH = 28           # Qwen2.5-VL: one visual token per 28 x 28 pixels
TOKENS_HIGH = 1280   # image budgets in visual tokens
TOKENS_LOW = 256     # about one page's share when 5 pages split a 1280-token budget

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


def parse_list(s):
    """evidence_pages / evidence_sources are strings holding Python lists."""
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return []
    return list(ast.literal_eval(s)) if isinstance(s, str) else list(s)


def page_counts(doc_ids):
    counts = {}
    for d in doc_ids:
        p = DOC_DIR / d
        if p.exists():
            with pymupdf.open(p) as doc:
                counts[d] = doc.page_count
    return counts


def load_qa():
    """All QA rows with parsed lists, page counts and a qid (row index of the parquet, the same qid as idea4)."""
    df = pd.read_parquet(QA_FILE).reset_index(names="qid")
    counts = page_counts(sorted(df["doc_id"].unique()))
    df = df[df["doc_id"].isin(counts)].copy()
    df["ev_pages"] = df["evidence_pages"].map(parse_list)
    df["ev_sources"] = df["evidence_sources"].map(parse_list)
    df["ev_types"] = df["ev_sources"].map(lambda xs: sorted({SOURCE_TO_TYPE.get(x, "other") for x in xs}))
    df["n_pages"] = df["doc_id"].map(counts)
    df["answerable"] = df["answer"] != "Not answerable"
    df["pages_in_pdf"] = [len(ps) > 0 and all(1 <= p <= n for p in ps) for ps, n in zip(df["ev_pages"], df["n_pages"])]
    return df


def eligible(df):
    """Answerable, at least one evidence page, every evidence page inside the PDF."""
    return df[df["answerable"] & df["pages_in_pdf"]].copy()


def andre_docs():
    """Documents of idea4's sample, excluded here so the two analyses share no documents."""
    f = IDEA4 / "out" / "sample_docs.txt"
    return set(f.read_text().split()) if f.exists() else set()


# ---------- page content ----------

def render(pdf_page):
    """Page image at 144 dpi, long side <= 1600 px (same as idea4)."""
    r = pdf_page.rect
    zoom = min(RENDER_DPI / 72, RENDER_MAX_SIDE / max(r.width, r.height))
    pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples), zoom


def text_layer(pdf_page):
    """The PDF's own text, blocks sorted top-left to bottom-right (what a PDF parser hands a reader)."""
    return pdf_page.get_text("text", sort=True).strip()


def masked(img, pdf_page, zoom, pad=1.5):
    """The page image with every text-layer word painted white: roughly what the pixels hold beyond the text layer."""
    from PIL import ImageDraw
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for w in pdf_page.get_text("words"):
        x0, y0, x1, y1 = (v * zoom for v in w[:4])
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill="white")
    return out


def fit(img, tokens):
    """Resize to at most `tokens` visual tokens (28 x 28 px each), sides multiples of 28."""
    w, h = img.size
    scale = min(1.0, math.sqrt(tokens * PATCH * PATCH / (w * h)))
    nw = max(PATCH, int(w * scale // PATCH) * PATCH)
    nh = max(PATCH, int(h * scale // PATCH) * PATCH)
    return img.resize((nw, nh), Image.BICUBIC)


def n_words(text):
    return len(re.findall(r"\w+", text or ""))
