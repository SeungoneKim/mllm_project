"""Step 6 (CPU): the result tables as JSON row files, for the results dashboard.

  python export.py      # after selection.py and analyze.py of each version; writes out/dashboard/*.json

Every file is a JSON array of flat row objects. The v0 tables are computed from the same per-question table as
analyze.py, so the dashboard shows exactly the numbers in out/v0/analysis/summary.md; the versions_* tables are read
from out/<version>/ (retrieval.csv, sensitivity.csv, analysis/metrics.json, analysis/per_question.csv) for every
version that has been analysed.
"""
import json

import numpy as np
import pandas as pd

from analyze import (MAIN_K, METHODS, decomposition_diagnostics, failure_table, missed_by_form, paired, per_question,
                     main_table, reasoning_ablation)
from common import OUT, VERSION, VOUT, load_elements, load_qa, read_jsonl, test_questions
from versions import VERSIONS

DASH = OUT / "dashboard"
BREAKDOWNS = ["stratum", "needs", "qtype", "gap", "n_gold", "combo", "answer_format", "doc_type"]
EXPLORER = ["topk", "coverage", "gold"]
EXPLORER_CONTROLS = ["v2_page"]  # controls whose picks and facts the question explorer shows too


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
    return pd.read_csv(VOUT / "retrieval.csv").to_dict("records")


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


def analysed():
    return [v for v in VERSIONS if (OUT / v / "analysis" / "metrics.json").exists()]


def first_version(rows, key):
    """Rows of methods shared by several versions are kept once, from the earliest version (same pages)."""
    seen, out = set(), []
    for r in rows:
        k = tuple(r[c] for c in key)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def versions_tables():
    """Method results of every analysed version, one row per (group, method, k), tagged with the version."""
    acc, ret, pair, sens = [], [], [], []
    for v in analysed():
        m = json.loads((OUT / v / "analysis" / "metrics.json").read_text())
        ren = VERSIONS[v].get("rename", {})
        tag = lambda r: {"version": v, **r, **({"method": ren[r["method"]]} if r.get("method") in ren else {})}
        # a VLM run on multi-page questions only: its control rows are the fallback, not results
        keep = lambda r: not (r.get("method") in ren and "control" not in VERSIONS[v].get("agg_sets", "control")
                              and r.get("group") == "control")
        acc += [tag(r) for r in m["main"] if keep(r)]
        ret += [tag(r) for r in pd.read_csv(OUT / v / "retrieval.csv").to_dict("records") if keep(r)]
        for key, rows in m.items():
            if key == "paired" or (key.startswith("paired_") and key not in ("paired_k3", "paired_vs_v2_evidence")):
                base = "topk" if key == "paired" else key[len("paired_"):]
                pair += [{"base": base, **tag(r)} for r in rows if keep(r)]
            if key == "paired_k3":
                pair += [{"base": "topk_k3", **tag(r)} for r in rows if keep(r)]
            if key == "paired_vs_v2_evidence":  # a control's own method against v2's evidence
                main = ren.get(VERSIONS[v]["main"], VERSIONS[v]["main"])
                pair += [{"base": "v2_evidence" if r["k"] == 5 else "v2_evidence_k3", "version": v, "method": main, **r}
                         for r in rows if keep({**r, "method": VERSIONS[v]["main"]})]
        if v != "v0" and not ren:
            sens += [{"version": v, **r} for r in pd.read_csv(OUT / v / "sensitivity.csv").to_dict("records")]
    return (first_version(acc, ["group", "method", "k"]), first_version(ret, ["group", "method", "k"]),
            first_version(pair, ["group", "base", "method"]), sens)


def version_questions(rows):
    """Adds the picks and answers of the later versions' own methods at k = 5 (and the VLM's facts of v2) to the
    question rows."""
    by_q = {r["qid"]: r for r in rows}
    for v in analysed():
        if v == "v0" or (VERSIONS[v].get("rename") and v not in EXPLORER_CONTROLS):
            continue
        main = VERSIONS[v]["main"]
        col = VERSIONS[v].get("rename", {}).get(main, main)
        pq = pd.read_csv(OUT / v / "analysis" / "per_question.csv")
        pq = pq[(pq.method == main) & (pq.k == MAIN_K)]
        for r in pq.itertuples():
            if r.qid in by_q:
                pages = json.loads(r.pages) if isinstance(r.pages, str) else r.pages
                by_q[r.qid][f"{col}_pages"] = " ".join(map(str, sorted(pages)))
                by_q[r.qid][f"{col}_retrieval"] = r.outcome
                by_q[r.qid][f"{col}_pred"] = None if pd.isna(r.pred) else r.pred
                by_q[r.qid][f"{col}_correct"] = None if pd.isna(r.correct) else "right" if r.correct > 0 else "wrong"
    qs = test_questions(load_qa()).set_index("qid")
    els = {}
    for v, field in [("v2", "facts"), *[(c, f"facts_{c}") for c in EXPLORER_CONTROLS]]:
        us = OUT / v / "unit_scores.pkl"
        if not us.exists():
            continue
        for qid, u in pd.read_pickle(us).items():
            if qid not in by_q:
                continue
            d = qs.loc[qid].doc_id
            els.setdefault(d, load_elements(d))
            by_q[qid][field] = " | ".join(
                f"p{'+'.join(str(els[d][i]['page']) for i in f['evidence'])}: {f['fact']}" for f in u["facts"])
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
    write("sensitivity", pd.read_csv(VOUT / "sensitivity.csv").to_dict("records"))
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
    write("questions", version_questions(rows))
    acc, ret, pair, sens = versions_tables()
    write("versions_accuracy", acc)
    write("versions_retrieval", ret)
    write("versions_paired", pair)
    write("versions_sensitivity", sens)
    metrics = {v: json.loads((OUT / v / "analysis" / "metrics.json").read_text()) for v in analysed()}
    agg = [{"version": v, **m["aggregation"]} for v, m in metrics.items() if "aggregation" in m]
    if agg:
        write("aggregation", agg)
    if "need_forms" in metrics.get("v1", {}):
        write("need_forms", [metrics["v1"]["need_forms"]])


if __name__ == "__main__":
    main()
