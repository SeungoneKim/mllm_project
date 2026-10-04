"""Shared paths, data loading and page rendering for Idea 4 (Parts A and B)."""
import ast
import base64
import re
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
IDEA4 = ROOT / "data_and_modality_analysis" / "idea4"
QA_FILE = ROOT / "data" / "mmlongbench" / "data" / "train-00000-of-00001.parquet"
DOC_DIR = ROOT / "data" / "mmlongbench" / "documents"
PAGE_CACHE = IDEA4 / "cache" / "pages"
OUT = IDEA4 / "out"
SEED = 777

# Gold evidence labels -> the names used in the form and the analysis.
SOURCE_TO_TYPE = {
    "Chart": "chart",
    "Table": "table",
    "Figure": "figure",
    "Pure-text (Plain-text)": "text",
    "Generalized-text (Layout)": "layout",
}
TYPES = ["text", "table", "chart", "figure", "layout"]

RENDER_DPI = 144
RENDER_MAX_SIDE = 1600
JPEG_QUALITY = 90
MIN_TEXT_WORDS = 20  # pages with fewer words in the text layer need OCR

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


def parse_list(s):
    """evidence_pages / evidence_sources are strings holding Python lists."""
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return []
    return list(ast.literal_eval(s)) if isinstance(s, str) else list(s)


def page_counts():
    counts = {}
    for p in sorted(DOC_DIR.iterdir()):
        with pymupdf.open(p) as doc:
            counts[p.name] = doc.page_count
    return counts


def load_qa(counts=None):
    """All QA rows with parsed lists, page counts and a qid (row index)."""
    df = pd.read_parquet(QA_FILE).reset_index(names="qid")
    counts = counts or page_counts()
    df["ev_pages"] = df["evidence_pages"].map(parse_list)
    df["ev_sources"] = df["evidence_sources"].map(parse_list)
    df["n_pages"] = df["doc_id"].map(counts)
    df["answerable"] = df["answer"] != "Not answerable"
    df["pages_in_pdf"] = [all(1 <= p <= n for p in ps) for ps, n in zip(df["ev_pages"], df["n_pages"])]
    return df


def build_pool(df):
    """Part A candidate pool: answerable, 1 page, 1 source, <= 80 pages, page inside the PDF."""
    m = (
        df["answerable"]
        & (df["ev_pages"].map(len) == 1)
        & (df["ev_sources"].map(len) == 1)
        & (df["n_pages"] <= 80)
        & df["pages_in_pdf"]
    )
    pool = df[m].copy()
    pool["gold_page"] = pool["ev_pages"].str[0].astype(int)
    pool["gold_source"] = pool["ev_sources"].str[0]
    pool["gold_type"] = pool["gold_source"].map(SOURCE_TO_TYPE)
    return pool


def doc_stem(doc_id):
    return Path(doc_id).stem


def page_image_path(doc_id, page):
    """page is 1-indexed."""
    return PAGE_CACHE / doc_stem(doc_id) / f"{page:04d}.jpg"


def render_page(pdf_page):
    """One rendering for every step: 144 dpi, long side <= 1600 px. Returns a pymupdf Pixmap."""
    r = pdf_page.rect
    zoom = min(RENDER_DPI / 72, RENDER_MAX_SIDE / max(r.width, r.height))
    return pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)


def render_doc(doc_id, overwrite=False):
    """Render every page of a document to the cache. Returns the number of pages written."""
    out_dir = PAGE_CACHE / doc_stem(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        for i, page in enumerate(doc):
            path = out_dir / f"{i + 1:04d}.jpg"
            if path.exists() and not overwrite:
                continue
            render_page(page).save(path, jpg_quality=JPEG_QUALITY)
            written += 1
    return written


def page_text(doc_id, page):
    """PDF text layer of a 1-indexed page."""
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        return doc[page - 1].get_text()


def doc_texts(doc_id):
    """Text layer of every page, as a list indexed by page - 1."""
    with pymupdf.open(DOC_DIR / doc_id) as doc:
        return [p.get_text() for p in doc]


def n_words(text):
    return len(re.findall(r"\w+", text or ""))


def b64(s):
    return base64.b64encode(str(s).encode("utf-8")).decode("ascii")


def unb64(s):
    return base64.b64decode(s.encode("ascii")).decode("utf-8")
