"""Build a self-contained HTML sheet for checking the 30 labels in out/annotations.csv against the pages.

Usage: python review.py   (after sample.py has rendered the pages)

Writes cache/review.html: per question, the gold labels, the assigned category, its rationale and every
evidence page. Edit out/annotations.csv directly to change a label, then rerun analysis.py.
"""
import base64
import html

import pandas as pd

from common import CACHE, OUT, parse_list


def main():
    sample = pd.read_csv(OUT / "sample.csv")
    ann = pd.read_csv(OUT / "annotations.csv").set_index("qid")
    parts = ["<!doctype html><meta charset='utf-8'><title>Idea 2 label review</title>",
             "<style>body{font:14px sans-serif;max-width:1200px;margin:auto;padding:16px}"
             "section{border-top:2px solid #888;padding:8px 0}img{max-width:48%;margin:4px;border:1px solid #ccc}"
             ".cat{font-weight:bold;color:#a40}</style>"]
    for row in sample.itertuples():
        a = ann.loc[row.qid]
        pages = sorted(set(parse_list(row.pages)))
        parts.append(
            f"<section><h3>[{row.stratum}] qid {row.qid}: {html.escape(row.question)}</h3>"
            f"<p>Answer: <b>{html.escape(str(row.answer))}</b> | {row.combo} | pages {pages} | {row.doc_id}</p>"
            f"<p class='cat'>{a.category}" + (f" (one page suffices: {a.one_page_suffices})"
                                              if isinstance(a.one_page_suffices, str) else "")
            + f"</p><p>{html.escape(a.rationale)}</p>")
        for p in pages:
            img = (CACHE / "pages" / str(row.qid) / f"p{p}.jpg").read_bytes()
            parts.append(f"<img title='p{p}' src='data:image/jpeg;base64,{base64.b64encode(img).decode()}'>")
        parts.append("</section>")
    (CACHE / "review.html").write_text("\n".join(parts))
    print("wrote", CACHE / "review.html")


if __name__ == "__main__":
    main()
