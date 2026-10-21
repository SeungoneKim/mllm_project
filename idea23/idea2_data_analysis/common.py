"""Shared data loading for the analysis of existing complementary cases (Ideas 2 + 3)."""
import ast
from pathlib import Path

import pandas as pd
import pymupdf

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CACHE = HERE / "cache"
IDEA1_SCORES = HERE.parent / "idea1" / "scores.pkl"  # ColQwen2.5 page scores from Analysis 1
SEED = 777

# The Hugging Face copies of these two PDFs do not match their questions (the served files fail the
# hub's size check; mi_phone.pdf is a university housing guide, the emissions deck is a marketing deck).
# Their 19 questions are excluded from the sample and from the retrieval check.
BAD_DOCS = {"mi_phone.pdf", "dr-vorapptchapter1emissionsources-121120210508-phpapp02_95.pdf"}

# Gold evidence-source labels -> short modality names used in Figure 2 and the report.
MODALITY = {
    "Pure-text (Plain-text)": "Text",
    "Generalized-text (Layout)": "Layout",
    "Table": "Table",
    "Chart": "Chart",
    "Figure": "Figure",
}

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


def parse_list(s):
    """evidence_pages / evidence_sources are strings holding Python lists."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return []
    return list(ast.literal_eval(s)) if isinstance(s, str) else list(s)


def load_qa(data_dir):
    """All 1,091 QA rows with parsed labels.

    qid is the row index of the Hugging Face parquet, which is also the index used by idea1/scores.pkl.
    Figure 2 counts evidence pages by the length of the listed pages (n_listed), as here; only one
    multi-source question lists a page twice (qid 700, [1, 1], Text), so n_distinct differs from it once.
    """
    data_dir = Path(data_dir)
    df = pd.read_parquet(data_dir / "data" / "train-00000-of-00001.parquet").reset_index(names="qid")
    df["pages"] = df["evidence_pages"].map(parse_list)
    df["sources"] = df["evidence_sources"].map(parse_list)
    df["mods"] = df["sources"].map(lambda l: tuple(sorted({MODALITY[s] for s in l})))
    df["combo"] = df["mods"].map(" + ".join)
    df["n_mods"] = df["mods"].map(len)
    df["n_listed"] = df["pages"].map(len)
    df["n_distinct"] = df["pages"].map(lambda l: len(set(l)))
    df["answerable"] = df["answer"] != "Not answerable"
    counts = {d: pymupdf.open(data_dir / "documents" / d).page_count for d in df["doc_id"].unique()}
    df["doc_pages"] = df["doc_id"].map(counts)
    df["pages_in_pdf"] = [all(1 <= p <= n for p in ps) for ps, n in zip(df["pages"], df["doc_pages"])]
    return df


def multi_source_groups(df):
    """The two populations of Figure 2.

    same_modality (panel a): one modality label and at least two listed evidence pages.
    cross_modality (panel b): two or more modality labels and at least one listed evidence page.
    Within cross_modality, `same_page` means exactly one listed page, so all modalities share it.
    The labels are per question, not per page, so for cross-page questions we do not know which
    modality sits on which page.
    """
    same = df[(df["n_mods"] == 1) & (df["n_listed"] >= 2)].copy()
    cross = df[(df["n_mods"] >= 2) & (df["n_listed"] >= 1)].copy()
    cross["same_page"] = cross["n_listed"] == 1
    return same, cross


def span_bucket(n):
    """Figure 2 legend: same page / 2 pages / 3-4 pages / >= 5 pages."""
    return "same page" if n <= 1 else "2 pages" if n == 2 else "3-4 pages" if n <= 4 else ">=5 pages"


def render_page(data_dir, doc_id, page, out_path, dpi=110, max_side=1400):
    """Render one 1-indexed page to JPEG for annotation."""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(Path(data_dir) / "documents" / doc_id) as doc:
        p = doc[page - 1]
        zoom = min(dpi / 72, max_side / max(p.rect.width, p.rect.height))
        p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(out_path, jpg_quality=85)
    return out_path


def page_text(data_dir, doc_id, page):
    with pymupdf.open(Path(data_dir) / "documents" / doc_id) as doc:
        return doc[page - 1].get_text()
