"""Step 6 (CPU): the result tables as JSON row files, for the results dashboard.

  python export.py      # after selection.py and analyze.py; writes out/dashboard/*.json

Every file is a JSON array of flat row objects computed from the same per-question table as analyze.py, so the
dashboard shows exactly the numbers in out/analysis/summary.md.
"""
import json

import numpy as np
import pandas as pd

from analyze import (MAIN_K, METHODS, decomposition_diagnostics, failure_table, missed_by_form, paired, per_question,
                     main_table, reasoning_ablation)
from common import OUT, load_qa, read_jsonl, test_questions

DASH = OUT / "dashboard"
BREAKDOWNS = ["stratum", "needs", "qtype", "gap", "n_gold", "combo", "answer_format", "doc_type"]
EXPLORER = ["topk", "coverage", "gold"]


def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write(name, rows):
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    (DASH / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(f"{name}.json: {len(rows)} rows")


def retrieval():
    return pd.read_csv(OUT / "retrieval.csv").rename(columns={"group": "group"}).to_dict("records")


def breakdowns(df, k=MAIN_K):
    m = df[(df.set == "multi") & ((df.k == k) | (df.method == "gold"))]
    rows = []
    for col in BREAKDOWNS:
        counts = m[m.method == "topk"][col].value_counts()
        x = m.assign(value=m[col].where(m[col].map(counts) >= 5, "other").astype(str))
        t = x.groupby(["value", "method"]).agg(n=("qid", "size"),
                                               complete=("outcome", lambda o: 100 * (o == "complete").mean()),
                                               acc=("correct", lambda c: 100 * c.mean())).reset_index()
        rows += [{"dimension": col, **r} for r in t.to_dict("records")]
    return rows


def failures(df):
    rows = []
    for m in ["topk", "coverage"]:
        t = failure_table(df, m).reset_index()
        rows += [{"method": m, **r} for r in t.to_dict("records")]
    return rows


def questions(df):
    """One row per test question: its labels, the needs, and what top-k, coverage and the gold pages gave."""
    qs = test_questions(load_qa()).set_index("qid")
    dec = {r["qid"]: r["needs"] for r in read_jsonl(OUT / "decomp.jsonl")}
    sub = df[df.method.isin(EXPLORER) & ((df.k == MAIN_K) | (df.method == "gold"))]
    piv = {c: sub.pivot(index="qid", columns="method", values=c)
           for c in ["pages", "outcome", "pred", "correct"]}
    rows = []
    for qid, q in qs.iterrows():
        r = {"qid": qid, "set": q.set, "stratum": q.stratum, "doc_id": q.doc_id, "doc_type": q.doc_type,
             "combo": q.combo, "question": q.question.strip(), "answer": q.answer, "format": q.answer_format,
             "gold": " ".join(map(str, q.gold)), "needs": " | ".join(dec.get(qid, [])), "n_needs": len(dec.get(qid, []))}
        for m in EXPLORER:
            pages = piv["pages"].loc[qid, m]
            r[f"{m}_pages"] = " ".join(map(str, sorted(pages)))
            if m != "gold":
                r[f"{m}_retrieval"] = piv["outcome"].loc[qid, m]
            r[f"{m}_pred"] = piv["pred"].loc[qid, m]
            ok = piv["correct"].loc[qid, m]
            r[f"{m}_correct"] = None if pd.isna(ok) else "right" if ok > 0 else "wrong"
        t, c = r["topk_retrieval"], r["coverage_retrieval"]
        r["case"] = ("coverage completes, top-k does not" if c == "complete" and t != "complete" else
                     "top-k completes, coverage does not" if t == "complete" and c != "complete" else
                     "both complete" if t == "complete" else "neither complete")
        rows.append(r)
    return rows


def main():
    DASH.mkdir(parents=True, exist_ok=True)
    df = per_question()
    qtypes = df.drop_duplicates("qid").set_index("qid").qtype
    write("retrieval", retrieval())
    write("accuracy", main_table(df).to_dict("records"))
    write("paired", paired(df).to_dict("records"))
    write("failure", failures(df))
    write("breakdowns", breakdowns(df))
    by_form, pairs = missed_by_form(df)
    write("missed_by_form", by_form.reset_index().rename(columns=lambda c: c.replace(" ", "_").replace("%", "pct"))
          .to_dict("records"))
    write("sensitivity", pd.read_csv(OUT / "sensitivity.csv").to_dict("records"))
    dd = decomposition_diagnostics(test_questions(load_qa()))
    dd["qtype"] = dd.qid.map(qtypes)
    write("needs_hist", dd.groupby(["m", "qtype"]).size().rename("n").reset_index().to_dict("records"))
    per_method, _ = reasoning_ablation(df)
    if per_method is not None:
        write("reasoning", [{"method": m, "reader": reader, "n": r["n"], "acc": r[col],
                             **({"wins": r["reasoning wins"], "losses": r["reasoning losses"]} if reader == "reasoning" else {})}
                            for m, r in per_method.iterrows()
                            for reader, col in [("short answer", "short answer acc"), ("reasoning", "reasoning acc")]])
    rows = questions(df)
    for r in rows:
        r["qtype"] = qtypes.get(r["qid"])
    write("questions", rows)


if __name__ == "__main__":
    main()
