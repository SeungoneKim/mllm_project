"""Existing complementary cases in MMLongBench-Doc (report section "Existing complementary cases").

Usage: python analysis.py DATA_DIR

1. Figure 2 counts: multi-source questions within one modality (panel a) and across modalities (panel b),
   with how many pages their evidence spans. Asserts that the totals match the figure in the report.
2. Retrieval check: for the populations of the three sampled strata, how often the ColQwen2.5 top-k holds
   all, some or none of the evidence pages (scores from idea1/scores.pkl, no GPU needed).
3. Manual categorization: summary of out/annotations.csv (redundant / partition / composition / locator)
   and the LaTeX table used in the report.

Writes out/summary.json and out/table_existing.tex, and prints every number quoted in the report.
"""
import json
import sys

import numpy as np
import pandas as pd

from common import BAD_DOCS, IDEA1_SCORES, OUT, load_qa, multi_source_groups, span_bucket

KS = (1, 3, 5)
STRATA = {
    "same_mod": "Same modality, $\\geq$2 pages",
    "cross_same": "Cross-modality, 1 page",
    "cross_multi": "Cross-modality, $\\geq$2 pages",
}
CATEGORIES = ["partition", "composition", "redundant", "locator"]


def pct(x):
    return round(100 * float(x), 1)


def figure2(df):
    same, cross = multi_source_groups(df)
    assert len(same) == 227 and len(cross) == 234, (len(same), len(cross))
    spans = ["same page", "2 pages", "3-4 pages", ">=5 pages"]
    tab = lambda g, key: (g.assign(span=g.n_listed.map(span_bucket)).groupby(key).span.value_counts()
                          .unstack(fill_value=0).reindex(columns=spans, fill_value=0))
    a, b = tab(same, "combo"), tab(cross, "combo")
    b["total"] = b.sum(axis=1)
    b = b.sort_values("total", ascending=False)
    print("Figure 2a (one modality, >= 2 listed pages): n =", len(same))
    print(a.assign(total=a.sum(axis=1)).sort_values("total", ascending=False).to_string(), "\n")
    print("Figure 2b (>= 2 modalities, >= 1 listed page): n =", len(cross),
          "| two modalities", int((cross.n_mods == 2).sum()), "| three or more", int((cross.n_mods >= 3).sum()))
    print(b.to_string(), "\n")

    two = cross[cross.n_mods == 2]
    top3 = ["Figure + Text", "Chart + Text", "Table + Text"]
    out = {
        "n_same_modality": len(same),
        "n_cross_modality": len(cross),
        "n_cross_two": int((cross.n_mods == 2).sum()),
        "cross_same_page": int(cross.same_page.sum()),
        "cross_same_page_pct": pct(cross.same_page.mean()),
        "figure_text_same_page": f"{int(b.loc['Figure + Text', 'same page'])}/{int(b.loc['Figure + Text', 'total'])}",
        "same_page_pct_by_combo": {c: pct(b.loc[c, "same page"] / b.loc[c, "total"]) for c in b.index[:4]},
        "text_plus_fig_chart_table_share_of_two": pct(two.combo.isin(top3).mean()),
        "chart_table": int(b.loc["Chart + Table", "total"]),
        "chart_figure": int(b.loc["Chart + Figure", "total"]),
        "same_mod_by_modality": a.sum(axis=1).sort_values(ascending=False).astype(int).to_dict(),
        "same_mod_2_pages_pct": pct((same.n_listed == 2).mean()),
        "table_2_pages": f"{int(a.loc['Table', '2 pages'])}/{int(a.loc['Table'].sum())}",
    }
    print("Figure 2 summary:", json.dumps(out, indent=1), "\n")
    return out


def ranks(scores, pages):
    """1-indexed ColQwen2.5 rank of each listed evidence page."""
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), int)
    rank[order] = np.arange(1, len(scores) + 1)
    return np.array([rank[p - 1] for p in sorted(set(pages))])


def hits(g, k):
    """Share of questions whose top-k holds all, some but not all, or none of the evidence pages."""
    return {
        f"all@{k}": pct(np.mean([(x <= k).all() for x in g.ranks])),
        f"some_not_all@{k}": pct(np.mean([(x <= k).any() and not (x <= k).all() for x in g.ranks])),
        f"none@{k}": pct(np.mean([not (x <= k).any() for x in g.ranks])),
    }


def retrieval(df):
    """ColQwen2.5 ranks of the evidence pages, per stratum.

    Pool: answerable questions whose listed pages exist in the PDF, outside BAD_DOCS. The multi-page strata
    keep questions with at most 5 distinct evidence pages, so that every question can be fully covered by the
    top 5; the 2-page subsets are the cleanest comparison with single-page questions.
    """
    scores = pd.read_pickle(IDEA1_SCORES)["colqwen"]
    ok = df[df.answerable & df.pages_in_pdf & (df.n_listed >= 1) & ~df.doc_id.isin(BAD_DOCS)].copy()
    ok["ranks"] = [ranks(scores[q], p) for q, p in zip(ok.qid, ok.pages)]
    same, cross = multi_source_groups(ok)
    pools = {"same_mod": same[same.n_distinct >= 2], "cross_same": cross[cross.same_page],
             "cross_multi": cross[cross.n_distinct >= 2]}
    out = {"pool": {k: len(v) for k, v in pools.items()}}
    groups = {
        "single_page": ok[(ok.n_distinct == 1) & (ok.n_mods == 1)],  # one page and one modality
        "same_mod": pools["same_mod"][pools["same_mod"].n_distinct <= 5],
        "cross_same": pools["cross_same"],
        "cross_multi": pools["cross_multi"][pools["cross_multi"].n_distinct <= 5],
        "same_mod_2p": pools["same_mod"][pools["same_mod"].n_distinct == 2],
        "cross_multi_2p": pools["cross_multi"][pools["cross_multi"].n_distinct == 2],
    }
    print("ColQwen2.5 retrieval, answerable questions with valid labels (excluding", len(BAD_DOCS),
          "mismatched PDFs); pools", out["pool"])
    for name, g in groups.items():
        r = {"n": len(g)}
        for k in KS:
            r.update(hits(g, k))
        r["median_worst_rank"] = float(np.median([x.max() for x in g.ranks]))
        out[name] = r
        print(f"  {name:14s}", r)

    # Where do multi-page evidence pages sit, and is a page missed by the top 5 next to a page it found?
    multi = pd.concat([pools["same_mod"], pools["cross_multi"]])
    gaps = multi.pages.map(lambda l: max(np.diff(sorted(set(l)))))
    out["multi_page_all_adjacent_pct"] = pct((gaps == 1).mean())
    out["multi_page_gap_ge5_pct"] = pct((gaps >= 5).mean())
    missed_next_to_found, missed = 0, 0
    for row in multi[multi.n_distinct <= 5].itertuples():
        pages = np.array(sorted(set(row.pages)))
        found, lost = pages[row.ranks <= 5], pages[row.ranks > 5]
        if len(found) and len(lost):
            missed += len(lost)
            missed_next_to_found += sum(np.abs(found - p).min() == 1 for p in lost)
    out["missed_pages"] = missed
    out["missed_next_to_found_pct"] = pct(missed_next_to_found / missed)
    print("  multi-page pool: all pages adjacent", out["multi_page_all_adjacent_pct"], "% | largest gap >= 5 pages",
          out["multi_page_gap_ge5_pct"], "% | partial hits @5: missed pages next to a found one",
          f"{missed_next_to_found}/{missed} = {out['missed_next_to_found_pct']}%")
    return out, ok.set_index("qid").ranks


def annotations(rank_of):
    ann = pd.read_csv(OUT / "annotations.csv")
    sample = pd.read_csv(OUT / "sample.csv")
    assert list(ann.qid) == list(sample.qid), "annotations.csv must follow the order of sample.csv"
    assert set(ann.category) <= set(CATEGORIES), set(ann.category)
    tab = pd.crosstab(ann.category, ann.stratum).reindex(index=CATEGORIES, columns=list(STRATA), fill_value=0)
    print("\nManual categorization (n = %d)\n" % len(ann), tab.to_string())

    comp = ann[ann.category.isin(["partition", "composition"])]
    multi = comp[comp.stratum != "cross_same"]
    out = {
        "table": tab.to_dict(),
        "complementary_by_stratum": comp.groupby("stratum").size().to_dict(),
        "multi_page_complementary": len(multi),
        "one_page_suffices": int((multi.one_page_suffices == "yes").sum()),
        "label_issues": ann[ann.label_issue.notna()].qid.tolist(),
    }
    # For complementary multi-page items whose pages are all needed, does the top-5 hold all of them?
    strict = multi[multi.one_page_suffices == "no"]
    hit = [(rank_of[q] <= 5).all() for q in strict.qid]
    out["strict_multi_page"] = len(strict)
    out["strict_all_in_top5"] = int(np.sum(hit))
    print("complementary:", out["complementary_by_stratum"], "| multi-page complementary", len(multi),
          "| one page suffices", out["one_page_suffices"], "| all needed pages in ColQwen top-5:",
          f"{int(np.sum(hit))}/{len(strict)}", "| label issues", out["label_issues"])
    return out, tab


def latex_table(tab, ret, path):
    names = {"partition": "Partition", "composition": "Composition", "redundant": "Redundant",
             "locator": "Locator only"}
    cols = list(STRATA)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Existing multi-source questions. Top: categories of 10 sampled questions per group. "
        r"Bottom: answerable questions with valid page labels (pool) and, for those with at most five evidence "
        r"pages, the share whose evidence pages ColQwen2.5 ranks in its top 5, all of them or only some "
        f"(single-page, single-modality questions: {ret['single_page']['all@5']:.0f}\\%)." + "}",
        r"  \label{tab:existing-categories}",
        r"  \begin{tabular}{@{}lccc@{}}",
        r"    \toprule",
        r"    & \multicolumn{1}{c}{Same mod.} & \multicolumn{2}{c}{Cross-modality} \\",
        r"    \cmidrule(lr){2-2} \cmidrule(l){3-4}",
        r"    & $\geq$2 pages & 1 page & $\geq$2 pages \\",
        r"    \midrule",
    ]
    for c in CATEGORIES:
        lines.append(f"    {names[c]} & " + " & ".join(str(int(tab.loc[c, s])) for s in cols) + r" \\")
    lines += [r"    \midrule", "    Pool & " + " & ".join(f"{ret['pool'][s]}" for s in cols) + r" \\"]
    lines.append(r"    All in top-5 (\%) & " + " & ".join(f"{ret[s]['all@5']:.0f}" for s in cols) + r" \\")
    lines.append(r"    Some, not all (\%) & " + " & ".join(
        "--" if s == "cross_same" else f"{ret[s]['some_not_all@5']:.0f}" for s in cols) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))
    print("\nwrote", path)


def main():
    df = load_qa(sys.argv[1] if len(sys.argv) > 1 else "data")
    fig = figure2(df)
    ret, rank_of = retrieval(df)
    ann, tab = annotations(rank_of)
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"figure2": fig, "retrieval": ret, "annotations": ann}, indent=1,
                                                 default=int))
    latex_table(tab, ret, OUT / "table_existing.tex")


if __name__ == "__main__":
    main()
