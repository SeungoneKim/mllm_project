"""Pick seed questions from MMLongBench-Doc.

A seed has exactly one evidence page and one evidence modality, so that every
multi-source structure in the item we build is one this pipeline introduced and
can account for.
"""
import ast, collections, pathlib, random
import pandas as pd
import pymupdf as fitz

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "documents"
QA = ROOT / "data" / "qa" / "train-00000-of-00001.parquet"


def load(min_pages=20, max_pages=140):
    df = pd.read_parquet(QA)
    df["ev_pages"] = df.evidence_pages.apply(ast.literal_eval)
    df["ev_src"] = df.evidence_sources.apply(ast.literal_eval)
    df = df[(df.ev_pages.str.len() == 1)
            & (df.ev_src.apply(lambda s: len(set(s)) == 1))
            & (df.answer_format != "None")].copy()
    pc = {}
    for f in DOCS.glob("*.pdf"):
        try:
            pc[f.name] = fitz.open(f).page_count
        except Exception:
            pc[f.name] = -1
    df["doc_pages"] = df.doc_id.map(pc)
    df = df[(df.doc_pages >= min_pages) & (df.doc_pages <= max_pages)]
    df = df[df.apply(lambda r: 1 < r.ev_pages[0] < r.doc_pages, axis=1)]
    df["modality"] = df.ev_src.apply(lambda s: s[0])
    return df.reset_index(drop=True)


def stratified(df, n, seed=0, per_doc=2):
    """n seeds spread over evidence modalities, capped per document."""
    mods = list(df.modality.unique())
    pools = {m: df[df.modality == m].sample(frac=1, random_state=seed).to_dict("records")
             for m in mods}
    used, out, i = collections.Counter(), [], 0
    while len(out) < n and any(pools.values()):
        m = mods[i % len(mods)]; i += 1
        while pools[m]:
            r = pools[m].pop(0)
            if used[r["doc_id"]] < per_doc:
                used[r["doc_id"]] += 1
                out.append(r)
                break
    for k, r in enumerate(out):
        r["sid"] = f"S{k:03d}"
    return out
