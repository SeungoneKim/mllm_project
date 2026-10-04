"""Page images and text from the source PDFs."""
import pathlib
import pymupdf as fitz

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "documents"


def page_png(doc_id, page_1idx, dpi=150):
    doc = fitz.open(DOCS / doc_id)
    png = doc[page_1idx - 1].get_pixmap(dpi=dpi, annots=False).tobytes("png")
    doc.close()
    return png


def page_text(doc_id, page_1idx):
    doc = fitz.open(DOCS / doc_id)
    t = doc[page_1idx - 1].get_text()
    doc.close()
    return t


def page_jpeg(doc_id, page_1idx, out, dpi=120, quality=85):
    doc = fitz.open(DOCS / doc_id)
    doc[page_1idx - 1].get_pixmap(dpi=dpi).pil_save(out, format="JPEG", quality=quality)
    doc.close()
    return out
