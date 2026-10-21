"""Step 5 (CPU): reader accuracy for every method, paired comparisons, and the failure analysis.

  python analyze.py     # reads out/selections.jsonl, out/reader/answers_*.jsonl, out/decomp.jsonl, out/scores.pkl
                        # writes out/analysis/{summary.md, metrics.json, per_question.csv, examples.md}

Answers are scored with idea5/score.py (the rule-based part of MMLongBench-Doc's scorer); a question counts as correct
when its score is above 0, as in idea5. Retrieval outcomes per question: complete (every gold page selected), partial
(some), none. A wrong answer is retrieval-attributable when the same reader answers correctly from the gold pages.
"""
import importlib.util
import json
import math

import numpy as np
import pandas as pd

from common import ANNOTATIONS, IDEA23, OUT, load_qa, read_jsonl, test_questions

ANALYSIS = OUT / "analysis"
MAIN_K = 5
METHODS = ["topk", "hybrid", "mmr", "decomp_rrf", "coverage", "coverage_beam", "gold"]
GROUPS = ["multi", "same_mod", "cross_multi", "control"]

_spec = importlib.util.spec_from_file_location("idea5_score", IDEA23.parent / "idea5" / "score.py")
idea5_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(idea5_score)


def mcnemar(b, c):
    """Two-sided exact McNemar p-value for b wins against c losses."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def outcome(pages, gold):
    s, g = set(pages), set(gold)
    return "complete" if g <= s else "partial" if s & g else "none"


def gap_bucket(gold):
    g = max(np.diff(sorted(gold)))
    return "adjacent" if g == 1 else "gap 2-4" if g <= 4 else "gap >=5"


def per_question():
    """One row per (question, method, k) with retrieval outcome and, when the reader has run, the answer."""
    qs = test_questions(load_qa()).set_index("qid")
    answers = {(r["qid"], tuple(r["pages"])): r["pred"]
               for p in sorted((OUT / "reader").glob("answers_*.jsonl")) for r in read_jsonl(p)}
    n_needs = {r["qid"]: len(r["needs"]) for r in read_jsonl(OUT / "decomp.jsonl")}
    rows = []
    for r in read_jsonl(OUT / "selections.jsonl"):
        if r["method"] not in METHODS:
            continue
        q = qs.loc[r["qid"]]
        pred = answers.get((r["qid"], tuple(sorted(r["pages"]))))
        sc = None if pred is None else idea5_score.score(q.answer, pred, q.answer_format)
        rows.append({"qid": r["qid"], "set": r["set"], "stratum": r["stratum"], "method": r["method"],
                     "k": r["k"] if r["k"] is not None else 0, "pages": r["pages"], "gold": q.gold,
                     "outcome": outcome(r["pages"], q.gold), "recall": len(set(r["pages"]) & set(q.gold)) / q.n_gold,
                     "pred": pred, "score": sc, "correct": None if sc is None else float(sc > 0),
                     "combo": q.combo, "doc_type": q.doc_type, "answer_format": q.answer_format,
                     "n_gold": q.n_gold, "gap": gap_bucket(q.gold) if q.n_gold > 1 else "single",
                     "needs": min(n_needs.get(r["qid"], 1), 3), "qtype": qtype(q)})
    return pd.DataFrame(rows)


def qtype(q):
    """Coarse question type from the wording and the answer format."""
    text = q.question.lower()
    if "how many" in text:
        return "count"
    if q.answer_format == "List":
        return "list"
    if any(w in text for w in ("difference", "ratio", "sum ", "total", "average", "compare", "higher", "lower",
                               "more than", "less than", "increase", "decrease", "percentage change", "growth")):
        return "compare/compute"
    return "lookup"


def by_group(df):
    """Rows for every reporting group: the two sets, and the two strata of multi."""
    return pd.concat([df.assign(group=df.set), df[df.set == "multi"].assign(group=lambda d: d.stratum)])


def main_table(df):
    g = by_group(df)
    g = g[(g.k.isin([3, MAIN_K])) | (g.method == "gold")]
    agg = g.groupby(["group", "method", "k"]).agg(n=("qid", "size"), answered=("correct", "count"),
                                                  complete=("outcome", lambda o: 100 * (o == "complete").mean()),
                                                  recall=("recall", lambda x: 100 * x.mean()),
                                                  acc=("correct", lambda x: 100 * x.mean()),
                                                  score=("score", lambda x: 100 * x.mean()))
    return agg.round(1).reset_index()


def paired(df, k=MAIN_K, base="topk"):
    """Wins and losses of each method against top-k on the same questions (answers and complete retrieval)."""
    out = []
    for group in GROUPS:
        g = by_group(df)
        g = g[(g.group == group) & ((g.k == k) | (g.method == "gold"))]
        b = g[g.method == base].set_index("qid")
        for m in METHODS:
            if m == base:
                continue
            x = g[g.method == m].set_index("qid").reindex(b.index)
            row = {"group": group, "method": m, "k": k}
            for name, col in [("answer", "correct"), ("complete", None)]:
                if col:
                    ok = x[col].notna() & b[col].notna()
                    xm, bm = x[col][ok] > 0, b[col][ok] > 0
                else:
                    xm, bm = x.outcome == "complete", b.outcome == "complete"
                w, l = int((xm & ~bm).sum()), int((~xm & bm).sum())
                row.update({f"{name}_wins": w, f"{name}_losses": l, f"{name}_p": round(mcnemar(w, l), 4)})
            out.append(row)
    return pd.DataFrame(out)


def failure_table(df, method, k=MAIN_K):
    """Retrieval outcome x answer for one method on multi, and whether the gold pages fix the wrong answers."""
    m = df[(df.set == "multi") & (df.method == method) & (df.k == k) & df.correct.notna()].set_index("qid")
    gold = df[(df.set == "multi") & (df.method == "gold") & df.correct.notna()].set_index("qid").correct
    m = m[m.index.isin(gold.index)]
    m["gold_correct"] = gold.reindex(m.index)
    t = m.groupby("outcome").agg(n=("correct", "size"), correct=("correct", "sum"),
                                 wrong_fixed_by_gold=("gold_correct", lambda s: 0))
    for o in t.index:
        sub = m[(m.outcome == o) & (m.correct == 0)]
        t.loc[o, "wrong_fixed_by_gold"] = int((sub.gold_correct == 1).sum())
    t["wrong"] = t.n - t.correct
    return t[["n", "correct", "wrong", "wrong_fixed_by_gold"]].astype(int).reindex(["complete", "partial", "none"])


def breakdown(df, col, methods=("topk", "coverage", "gold"), k=MAIN_K, min_n=5):
    """Complete-evidence recall and accuracy per value of `col`; values with fewer than min_n questions -> other."""
    m = df[(df.set == "multi") & df.method.isin(methods) & ((df.k == k) | (df.method == "gold"))].copy()
    counts = m[m.method == methods[0]][col].value_counts()
    m[col] = m[col].where(m[col].map(counts) >= min_n, "other")
    t = m.groupby([col, "method"]).agg(complete=("outcome", lambda o: 100 * (o == "complete").mean()),
                                       acc=("correct", lambda x: 100 * x.mean())).round(1).unstack("method")
    cols = {"n": m[m.method == methods[0]].groupby(col).size()}
    for meth in methods:
        if meth != "gold":
            cols[f"{meth} complete"] = t[("complete", meth)]
        cols[f"{meth} acc"] = t[("acc", meth)]
    return pd.DataFrame(cols).sort_values("n", ascending=False)


def missed_by_form(df, methods=("topk", "coverage"), k=MAIN_K):
    """Per gold page of a multi-page question: the form of its evidence (out/page_tags.jsonl), its ColQwen rank, and
    whether each method selected it. Also, for cross-modal questions that a method retrieves only in part, which
    form the found and the missed pages hold."""
    tags = {(r["qid"], r["page"]): r["form"] for r in read_jsonl(OUT / "page_tags.jsonl")}
    if not tags:
        return None, None
    sq = pd.read_pickle(OUT / "scores.pkl")["q"]
    m = df[(df.set == "multi") & (df.k == k)]
    sel = {(r.qid, r.method): set(r.pages) for r in m.itertuples()}
    rows = []
    for r in m[m.method == methods[0]].itertuples():
        rank = np.empty(len(sq[r.qid]), int)
        rank[np.argsort(-sq[r.qid], kind="stable")] = np.arange(1, len(rank) + 1)
        for p in r.gold:
            rows.append({"qid": r.qid, "stratum": r.stratum, "form": tags.get((r.qid, p)) or "untagged",
                         "rank": rank[p - 1], **{f"{x} selected": p in sel[(r.qid, x)] for x in methods}})
    t = pd.DataFrame(rows)
    agg = {"pages": ("qid", "size"), "median ColQwen rank": ("rank", "median")}
    agg.update({f"{x} missed %": (f"{x} selected", lambda s: 100 * (1 - s.mean())) for x in methods})
    by_form = t.groupby(["stratum", "form"]).agg(**agg).round(1)
    cross = t[t.stratum == "cross_multi"]
    pairs = []
    for qid, g in cross.groupby("qid"):
        hit = g[f"{methods[0]} selected"]
        if hit.any() and not hit.all():
            pairs += [(f, mf) for f in g[hit].form for mf in g[~hit].form]
    pair_tab = pd.Series(pairs).value_counts().rename_axis("found -> missed").to_frame("pairs") if pairs else None
    return by_form, pair_tab


def reasoning_ablation(df):
    """The short-answer reader against the same reader asked to reason first (reader.py --reason, out/reader_cot),
    on multi-page questions at k = 5. Returns (per method, per retrieval outcome of top-k) or (None, None)."""
    cot = {(r["qid"], tuple(r["pages"])): r["pred"]
           for p in sorted((OUT / "reader_cot").glob("answers_*.jsonl")) for r in read_jsonl(p)}
    if not cot:
        return None, None
    qa = load_qa().set_index("qid")
    m = df[(df.set == "multi") & df.method.isin(["topk", "coverage", "gold"]) & ((df.k == MAIN_K) | (df.method == "gold"))]
    m = m.assign(correct_cot=[None if (r.qid, tuple(sorted(r.pages))) not in cot else
                              float(idea5_score.score(qa.loc[r.qid].answer, cot[(r.qid, tuple(sorted(r.pages)))],
                                                      qa.loc[r.qid].answer_format) > 0) for r in m.itertuples()])
    m = m[m.correct_cot.notna()]
    per_method = m.groupby("method").apply(lambda g: pd.Series({
        "n": len(g), "short answer acc": 100 * g.correct.mean(), "reasoning acc": 100 * g.correct_cot.mean(),
        "reasoning wins": int(((g.correct_cot > 0) & (g.correct == 0)).sum()),
        "reasoning losses": int(((g.correct_cot == 0) & (g.correct > 0)).sum())}), include_groups=False).round(1)
    t = m[m.method == "topk"]
    by_outcome = t.groupby("outcome").agg(n=("qid", "size"), short=("correct", lambda c: 100 * c.mean()),
                                          reasoning=("correct_cot", lambda c: 100 * c.mean())).round(1)
    return per_method.reindex(["topk", "coverage", "gold"]), by_outcome.reindex(["complete", "partial", "none"])


def decomposition_diagnostics(qs):
    """How the needs relate to the gold pages: does some need rank each gold page first in the document?"""
    sc = pd.read_pickle(OUT / "scores.pkl")
    decomp = {r["qid"]: r for r in read_jsonl(OUT / "decomp.jsonl")}
    rows = []
    for r in qs[qs.set == "multi"].itertuples():
        d = decomp.get(r.qid)
        if d is None or r.qid not in sc["need"]:
            continue
        need_top = {int(np.argmax(s)) + 1 for s in sc["need"][r.qid]}
        q_top = int(np.argmax(sc["q"][r.qid])) + 1
        rows.append({"qid": r.qid, "m": len(d["needs"]), "parsed": d["parsed"],
                     "gold_hit_by_needs": len(need_top & set(r.gold)) / r.n_gold,
                     "all_gold_top1_of_some_need": set(r.gold) <= need_top,
                     "distinct_need_tops": len(need_top), "q_top_in_gold": q_top in r.gold})
    return pd.DataFrame(rows)


def examples(df, qs, decomp, n=4):
    """Questions where coverage and top-k differ in retrieval, plus reader failures with complete retrieval."""
    sub = df[(df.set == "multi") & ((df.k == MAIN_K) | (df.method == "gold"))]
    w, a, pages, preds = (sub.pivot(index="qid", columns="method", values=c)
                          for c in ["outcome", "correct", "pages", "pred"])
    cases = {
        "coverage completes, top-k does not": w[(w.coverage == "complete") & (w.topk != "complete")].index,
        "top-k completes, coverage does not": w[(w.topk == "complete") & (w.coverage != "complete")].index,
        "top-k complete but wrong, gold also wrong (reader failure)":
            a[(a.get("topk") == 0) & (a.get("gold") == 0)].index.intersection(w[w.topk == "complete"].index),
    }
    qa = qs.set_index("qid")
    lines = ["# Examples (k = 5)", ""]
    for title, idx in cases.items():
        lines += [f"## {title} ({len(idx)} questions)", ""]
        for qid in list(idx)[:n]:
            q = qa.loc[qid]
            lines += [f"- **qid {qid}** ({q.doc_id}, {q.combo}): {q.question.strip()}",
                      f"  - answer: {q.answer} | gold pages {q.gold}",
                      f"  - needs: {decomp.get(qid, {}).get('needs')}"]
            for m in ["topk", "coverage", "gold"]:
                if m in pages.columns:
                    lines.append(f"  - {m}: pages {pages.loc[qid, m]} -> {preds.loc[qid, m] if m in preds.columns else None}")
        lines.append("")
    return "\n".join(lines)


def md(df):
    return df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    qs = test_questions(load_qa())
    df = per_question()
    df.drop(columns=["gold"]).to_csv(ANALYSIS / "per_question.csv", index=False)
    have_answers = df.correct.notna().any()
    print(f"{df.qid.nunique()} questions, {len(df)} (question, method, k) rows, "
          f"{int(df.correct.notna().sum())} with a reader answer")

    out, lines = {}, ["# Idea 2+3 baseline results", ""]
    tab = main_table(df)
    out["main"] = tab.to_dict("records")
    for group in GROUPS:
        t = tab[tab.group == group].set_index(["method", "k"]).drop(columns="group")
        t = t.reindex([(m, k) for m in METHODS for k in ([0] if m == "gold" else [3, MAIN_K])]).dropna(how="all")
        lines += [f"## {group}", "", md(t), ""]

    if have_answers:
        pt = paired(df)
        out["paired"] = pt.to_dict("records")
        lines += ["## Paired against top-k (k = 5): wins / losses, exact McNemar p", "", md(pt.set_index(["group", "method"])), ""]
        for m in ["topk", "coverage"]:
            ft = failure_table(df, m)
            out[f"failure_{m}"] = ft.reset_index().to_dict("records")
            lines += [f"## Failure decomposition, {m}, multi, k = 5", "", md(ft), ""]
        for col in ["stratum", "needs", "qtype", "gap", "n_gold", "combo", "answer_format", "doc_type"]:
            t = breakdown(df, col)
            out[f"by_{col}"] = json.loads(t.to_json())
            lines += [f"## By {col} (multi, k = 5; complete-evidence recall and accuracy, %)", "", md(t), ""]
        ann = pd.read_csv(ANNOTATIONS)[["qid", "category"]]
        t = breakdown(df.merge(ann, on="qid"), "category", min_n=1)
        lines += ["## By Analysis 2 category (annotated questions in multi only)", "", md(t), ""]

    per_method, by_outcome = reasoning_ablation(df) if have_answers else (None, None)
    if per_method is not None:
        out["reasoning"] = per_method.reset_index().to_dict("records")
        out["reasoning_by_outcome"] = by_outcome.reset_index().to_dict("records")
        lines += ["## Ablation: reader asked to reason before answering (multi, k = 5)", "", md(per_method), "",
                  "Top-k, by retrieval outcome (accuracy %):", "", md(by_outcome), ""]

    by_form, pairs = missed_by_form(df)
    if by_form is not None:
        out["missed_by_form"] = by_form.reset_index().to_dict("records")
        lines += ["## Gold pages by evidence form (multi, k = 5): how often each method misses them", "", md(by_form), ""]
        if pairs is not None:
            out["cross_partial_pairs"] = pairs.reset_index().to_dict("records")
            lines += ["## Cross-modal questions retrieved in part by top-k: form of found -> missed page", "",
                      md(pairs), ""]

    dd = decomposition_diagnostics(qs)
    if len(dd):
        out["decomposition"] = {
            "n": len(dd), "unparsed": int((~dd.parsed).sum()), "needs_hist": dd.m.value_counts().sort_index().to_dict(),
            "gold_pages_ranked_first_by_some_need_pct": round(100 * dd.gold_hit_by_needs.mean(), 1),
            "all_gold_ranked_first_by_needs_pct": round(100 * dd.all_gold_top1_of_some_need.mean(), 1),
            "question_top1_in_gold_pct": round(100 * dd.q_top_in_gold.mean(), 1),
        }
        lines += ["## Decomposition diagnostics (multi)", "", "```", json.dumps(out["decomposition"], indent=1), "```", ""]
    decomp = {r["qid"]: r for r in read_jsonl(OUT / "decomp.jsonl")}
    (ANALYSIS / "examples.md").write_text(examples(df, qs, decomp))
    (ANALYSIS / "metrics.json").write_text(json.dumps(out, indent=1, default=str))
    (ANALYSIS / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
