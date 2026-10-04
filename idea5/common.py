"""Shared paths, data loading and page rendering for Idea 5.

Rendering and ColQwen settings are identical to idea4/common.py and idea4/retrieve.py, so André's cached page
images (idea4/cache/pages) and page embeddings (idea4/cache/colqwen) are reused as they are. New pages and
embeddings are written under idea5/cache. Paths can be overridden with environment variables (see README.md).
"""
import ast
import os
import re
from pathlib import Path

import pandas as pd
import pymupdf

IDEA5 = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MLLM_ROOT", IDEA5.parents[1]))  # same layout as idea4: <ROOT>/<folder>/idea5
MMLB = Path(os.environ.get("MMLB_DIR", ROOT / "data" / "mmlongbench"))
QA_FILE = MMLB / "data" / "train-00000-of-00001.parquet"
DOC_DIR = MMLB / "documents"
IDEA4 = Path(os.environ.get("IDEA4_DIR", IDEA5.parent / "idea4"))

CACHE = Path(os.environ.get("IDEA5_CACHE", IDEA5 / "cache"))
PAGE_CACHE = CACHE / "pages"
EMB_CACHE = CACHE / "colqwen"
OUT = Path(os.environ.get("IDEA5_OUT", IDEA5 / "out"))
EXPORTS = Path(os.environ.get("IDEA5_EXPORTS", IDEA5 / "exports"))  # annotators' JSON exports from the form
SCAN = OUT / "scan"
SEED = 555

SOURCE_TO_TYPE = {
    "Chart": "chart",
    "Table": "table",
    "Figure": "figure",
    "Pure-text (Plain-text)": "text",
    "Generalized-text (Layout)": "layout",
}
TYPES = ["text", "table", "chart", "figure", "layout"]

# Same rendering as idea4 (common.py) and the same retriever settings (retrieve.py).
RENDER_DPI = 144
RENDER_MAX_SIDE = 1600
JPEG_QUALITY = 90
MIN_TEXT_WORDS = 20  # pages with fewer text-layer words count as having no usable text layer
COLQWEN = "vidore/colqwen2.5-v0.2"
MAX_VISUAL_TOKENS = 768
TOP_K = 5  # a question is a miss when no gold page is in the top 5

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


def parse_list(s):
    """evidence_pages / evidence_sources are strings holding Python lists."""
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return []
    return list(ast.literal_eval(s)) if isinstance(s, str) else list(s)


def doc_stem(doc_id):
    return Path(doc_id).stem


def page_counts(doc_ids=None):
    counts = {}
    paths = [DOC_DIR / d for d in doc_ids] if doc_ids is not None else sorted(DOC_DIR.glob("*.pdf"))
    for p in paths:
        if p.exists():
            with pymupdf.open(p) as doc:
                counts[p.name] = doc.page_count
    return counts


def load_qa():
    """All QA rows with parsed lists, page counts and a qid (row index, the same qid as idea4)."""
    df = pd.read_parquet(QA_FILE).reset_index(names="qid")
    counts = page_counts(sorted(df["doc_id"].unique()))
    df = df[df["doc_id"].isin(counts)].copy()  # documents present on disk
    df["ev_pages"] = df["evidence_pages"].map(parse_list)
    df["ev_sources"] = df["evidence_sources"].map(parse_list)
    df["ev_types"] = df["ev_sources"].map(lambda xs: sorted({SOURCE_TO_TYPE.get(x, "other") for x in xs}))
    df["n_pages"] = df["doc_id"].map(counts)
    df["answerable"] = df["answer"] != "Not answerable"
    df["pages_in_pdf"] = [len(ps) > 0 and all(1 <= p <= n for p in ps) for ps, n in zip(df["ev_pages"], df["n_pages"])]
    return df


def eligible(df):
    """Questions the scan scores: answerable, at least one evidence page, every evidence page inside the PDF."""
    return df[df["answerable"] & df["pages_in_pdf"]].copy()


def _cached(sub, doc_id, name):
    """Our cache first, then André's; returns our path when neither exists (the place to write)."""
    own = CACHE / sub / doc_stem(doc_id) / name if name else CACHE / sub / f"{doc_stem(doc_id)}.pt"
    if own.exists():
        return own
    theirs = IDEA4 / "cache" / sub / doc_stem(doc_id) / name if name else IDEA4 / "cache" / sub / f"{doc_stem(doc_id)}.pt"
    return theirs if theirs.exists() else own


def page_image_path(doc_id, page):
    """page is 1-indexed."""
    return _cached("pages", doc_id, f"{page:04d}.jpg")


def emb_path(doc_id):
    """ColQwen page embeddings of one document: a list of float16 tensors [n_tokens, 128], one per page."""
    return _cached("colqwen", doc_id, None)


def render_page(pdf_page):
    """One rendering for every step: 144 dpi, long side <= 1600 px. Returns a pymupdf Pixmap."""
    r = pdf_page.rect
    zoom = min(RENDER_DPI / 72, RENDER_MAX_SIDE / max(r.width, r.height))
    return pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)


def render_doc(doc_id):
    """Render every page that is in neither cache. Returns the number of pages written."""
    written = 0
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        for i, page in enumerate(doc):
            path = page_image_path(doc_id, i + 1)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            render_page(page).save(path, jpg_quality=JPEG_QUALITY)
            written += 1
    return written


def doc_texts(doc_id):
    """Text layer of every page, as a list indexed by page - 1."""
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        return [p.get_text() for p in doc]


def n_words(text):
    return len(re.findall(r"\w+", text or ""))
