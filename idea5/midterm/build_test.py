"""Step 1 (CPU): build the midterm test set from MMLongBench-Doc.

  python build_test.py [--limit N]   # writes out/test/{items,pages}.jsonl, out/test/tokinfo.npz, cache/pages/<page>.png

Test questions: every eligible question of the benchmark (answerable, at least one evidence page, every evidence page
inside the PDF; ../analysis/common.py), 829 in all. The reader gets the question's gold evidence pages, so the
comparison isolates how each page's two modalities are given to the reader. Nothing is trained or tuned on these
questions.

For each gold page, at the full budget of 1280 visual tokens (the page image resized so that every 28 x 28 pixel
block is one Qwen2.5-VL visual token, as in the analysis):
  image      cache/pages/<doc_stem>_<page>.png
  text       the PDF text layer (PyMuPDF, blocks sorted top-left to bottom-right), in pages.jsonl
  tokinfo    two [rows, cols] arrays over the visual-token grid, keys <page>_ink and <page>_txt:
               ink  share of the block's pixels darker than 200 (0-255 gray): is anything drawn there?
               txt  share of the block covered by text-layer word boxes: is it already in the text layer?
The selectors in token_select.py read tokinfo; the reader checks that its token grids match these shapes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf

from shared import (DOC_DIR, INK_GRAY, MIN_TEXT_WORDS, PAGES, PATCH, TEST, TOKENS_HIGH, eligible, fit, load_qa,
                    n_words, render, text_layer)


def page_key(doc_id, page):
    return f"{Path(doc_id).stem}_{page:04d}"


def token_info(img, pdf_page):
    """ink and text-coverage shares per 28 x 28 block of the resized page image."""
    w, h = img.size
    rows, cols = h // PATCH, w // PATCH
    gray = np.asarray(img.convert("L"))
    ink = (gray < INK_GRAY)[: rows * PATCH, : cols * PATCH]
    covered = np.zeros((h, w), dtype=bool)
    r = pdf_page.rect
    sx, sy = w / r.width, h / r.height  # PDF points -> pixels of the resized image
    for x0, y0, x1, y1, *_ in pdf_page.get_text("words"):
        a, b = int((x0 - r.x0) * sx), int((y0 - r.y0) * sy)
        c, d = int(np.ceil((x1 - r.x0) * sx)), int(np.ceil((y1 - r.y0) * sy))
        covered[max(b, 0):max(d, 0), max(a, 0):max(c, 0)] = True
    covered = covered[: rows * PATCH, : cols * PATCH]
    block = lambda m: m.reshape(rows, PATCH, cols, PATCH).mean(axis=(1, 3))
    return block(ink).astype(np.float16), block(covered).astype(np.float16)


def main(limit):
    qa = eligible(load_qa()).sort_values("qid")
    if limit:
        qa = qa.head(limit)
    TEST.mkdir(parents=True, exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)
    pages, arrays, items = {}, {}, []
    for d, qs in qa.groupby("doc_id"):
        with pymupdf.open(DOC_DIR / d) as doc:
            for _, q in qs.iterrows():
                nums = sorted(set(q["ev_pages"]))
                keys = [page_key(d, p) for p in nums]
                for p, key in zip(nums, keys):
                    if key in pages:
                        continue
                    page = doc[p - 1]
                    img, _ = render(page)
                    hi = fit(img, TOKENS_HIGH)
                    hi.save(PAGES / f"{key}.png")
                    ink, txt = token_info(hi, page)
                    arrays[f"{key}_ink"], arrays[f"{key}_txt"] = ink, txt
                    text = text_layer(page)
                    pages[key] = {"key": key, "doc_id": d, "page": p, "text": text, "text_words": n_words(text),
                                  "size": list(hi.size), "grid": list(ink.shape),
                                  "ink_share": float((ink > 0.02).mean()), "txt_share": float((txt > 0.5).mean())}
                no_text = [pages[k]["text_words"] < MIN_TEXT_WORDS for k in keys]
                items.append({
                    "qid": int(q["qid"]), "doc_id": d, "doc_type": q["doc_type"], "pages": nums, "page_keys": keys,
                    "n_gold": len(nums), "n_pages": int(q["n_pages"]), "ev_types": q["ev_types"],
                    "single_type": len(q["ev_types"]) == 1, "question": q["question"], "answer": q["answer"],
                    "answer_format": q["answer_format"], "n_no_text": int(sum(no_text)),
                    "no_text_layer": bool(any(no_text))})
    items.sort(key=lambda it: it["qid"])
    with (TEST / "items.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (TEST / "pages.jsonl").open("w") as f:
        for p in pages.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    np.savez_compressed(TEST / "tokinfo.npz", **arrays)
    n_single = sum(it["n_gold"] == 1 for it in items)
    print(f"{len(items)} questions ({n_single} single-page, {len(items) - n_single} multi-page), {len(pages)} pages "
          f"from {qa['doc_id'].nunique()} documents; {sum(it['no_text_layer'] for it in items)} questions with a gold "
          f"page that has no text layer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    main(ap.parse_args().limit)
