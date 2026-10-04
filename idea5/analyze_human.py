"""Step 3d: human simulation analysis (CPU).

  python analyze_human.py   # reads out/human/{items.csv, desc_scores.jsonl}, exports/*.json, vlm_descriptions.jsonl
                            # writes out/human/{summary.md, metrics.json, rows.csv, fig_*.pdf}

Questions the step answers, per describer (each annotator, the untrained VLM) and description kind:
  Self-retrieval   Does the description rank its own page first among the target and its 3 look-alikes (chance 25%),
                   and among all pages of the document? This is the training reward of Idea 5.
  Question words   What share of the real question's content words does the description contain? The question was
                   never shown, so this measures whether the details that tell a page apart are the ones questions use.
  Cues             Which cues annotators used to tell the target apart.
Generic and distinct descriptions are compared pairwise per item (Wilcoxon signed-rank on rank_all; seeded bootstrap of
the change in reciprocal rank).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import EXPORTS, OUT, SEED
from cues import content_words
from score_desc import load_descriptions

HUMAN = OUT / "human"
CUE_NAMES = {"title": "Section or statement title", "caption": "Table or figure caption",
             "labels": "Row or column labels", "period": "Dates or time periods", "numbers": "Specific numbers",
             "entities": "Names (people, products, places)", "pagelabel": "Printed page number",
             "visual": "Colours, pictures, visual style", "layout": "Position or layout", "other": "Other"}
C1, GREY, INK, INK2, GRID = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"


def qcov(question, text):
    q = set(content_words(question))
    return len(q & set(content_words(text))) / len(q) if q else float("nan")


def boot_mean_ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED)
    b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def load():
    items = pd.read_csv(HUMAN / "items.csv")
    items = items[items["group"] != "practice"]
    sc = pd.DataFrame([json.loads(l) for l in (HUMAN / "desc_scores.jsonl").open()])
    desc = load_descriptions()
    rows = sc.merge(desc, on=["source", "who", "item_id", "kind"]).merge(
        items[["item_id", "group", "question", "doc_type", "best_rank", "cause"]], on="item_id")
    rows["top1_sib"] = rows["rank_sib"] == 1
    rows["top1_all"] = rows["rank_all"] == 1
    rows["rr_all"] = 1 / rows["rank_all"]
    rows["qcov"] = [qcov(q, t) for q, t in zip(rows["question"], rows["text"])]
    rows["words"] = rows["text"].str.split().str.len()
    rows["describer"] = np.where(rows["source"] == "human", "human: " + rows["who"].astype(str), "VLM: " +
                                 rows["who"].astype(str).str.split("/").str[-1])
    return items, rows


def paired(rows, describer):
    d = rows[rows["describer"] == describer].pivot_table(index="item_id", columns="kind",
                                                          values=["rank_all", "rr_all", "qcov"], aggfunc="first")
    d = d.dropna()
    if len(d) == 0:
        return None
    diff_rr = d[("rr_all", "distinct")] - d[("rr_all", "generic")]
    a, b = d[("rank_all", "distinct")], d[("rank_all", "generic")]
    p = float(wilcoxon(a, b).pvalue) if (a != b).any() else 1.0
    return {"n": int(len(d)), "rr_change": float(diff_rr.mean()), "rr_change_ci": boot_mean_ci(diff_rr),
            "better": int((a < b).sum()), "same": int((a == b).sum()), "worse": int((a > b).sum()), "wilcoxon_p": p,
            "qcov_change": float((d[("qcov", "distinct")] - d[("qcov", "generic")]).mean())}


def cues_table():
    rows = []
    for p in sorted(EXPORTS.glob("annotations_*.json")):
        if "SYNTHETIC" in p.name and "SYNTHETIC" not in str(OUT):
            continue
        data = json.loads(p.read_text())
        for it in data["items"]:
            if it.get("practice") or not it.get("done"):
                continue
            rows.append({"who": data["annotator"], "item_id": it["item_id"], "cues": it.get("cues") or [],
                         "confidence": it.get("confidence"), "t_generic_s": it.get("t_generic_s"),
                         "t_distinct_s": it.get("t_distinct_s"), "notes": it.get("notes", "")})
    return pd.DataFrame(rows)


def style(ax, grid_axis):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_selfret(rows):
    import matplotlib.pyplot as plt
    ds = sorted(rows["describer"].unique(), key=lambda s: (not s.startswith("human"), s))
    x = np.arange(len(ds))
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(ds), 2.4))
    for k, (kind, col, lab) in enumerate([("generic", GREY, "page alone"), ("distinct", C1, "next to look-alikes")]):
        v = [rows[(rows["describer"] == d) & (rows["kind"] == kind)]["top1_sib"].mean() for d in ds]
        xs = x + (k - 0.5) * 0.38
        ax.bar(xs, v, width=0.36, color=col, label=lab)
        for xi, vi in zip(xs, v):
            ax.text(xi, (0 if np.isnan(vi) else vi) + 0.02, f"{vi:.0%}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(0.25, color=INK, linewidth=1, linestyle=(0, (2, 1.5)))
    ax.text(x[0] - 0.45, 0.235, "chance", fontsize=7, color=INK2, ha="left", va="top")
    ax.set_xticks(x, [d.replace(": ", ":\n") for d in ds])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("target ranked 1st of 4", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7, loc="upper left", title="description written", title_fontsize=7)
    style(ax, "y")
    fig.tight_layout()
    fig.savefig(HUMAN / "fig_selfretrieval.pdf")
    plt.close(fig)


def fig_cues(cues):
    import matplotlib.pyplot as plt
    if cues.empty:
        return
    freq = pd.Series({k: cues["cues"].map(lambda c: k in c).mean() for k in CUE_NAMES}).sort_values()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    y = np.arange(len(freq))
    ax.barh(y, freq.values, height=0.6, color=C1)
    for yi, v in zip(y, freq.values):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [CUE_NAMES[k] for k in freq.index])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(f"share of descriptions (n = {len(cues)})", fontsize=8, color=INK2)
    style(ax, "x")
    fig.tight_layout()
    fig.savefig(HUMAN / "fig_cues.pdf")
    plt.close(fig)


def main():
    items, rows = load()
    cues = cues_table()
    rows.drop(columns=["text"]).to_csv(HUMAN / "rows.csv", index=False)
    m = {"n_items": int(len(items)), "groups": items["group"].value_counts().to_dict(), "by_describer": {},
         "paired_generic_vs_distinct": {}}
    for (d, kind), g in rows.groupby(["describer", "kind"]):
        m["by_describer"].setdefault(d, {})[kind] = {
            "n": int(len(g)), "top1_of_4": float(g["top1_sib"].mean()), "top1_all": float(g["top1_all"].mean()),
            "median_rank_all": float(g["rank_all"].median()), "mrr_all": float(g["rr_all"].mean()),
            "question_word_coverage": float(g["qcov"].mean()), "median_words": float(g["words"].median())}
    for d in rows["describer"].unique():
        m["paired_generic_vs_distinct"][d] = paired(rows, d)
    hum = rows[rows["source"] == "human"]
    if not hum.empty and (rows["source"] == "vlm").any():
        h = hum.groupby(["item_id", "kind"])["rr_all"].mean().unstack()
        v = rows[rows["source"] == "vlm"].groupby(["item_id", "kind"])["rr_all"].mean().unstack()
        j = h.join(v, lsuffix="_h", rsuffix="_v", how="inner").dropna()
        if len(j):
            diff = j["distinct_h"] - j["distinct_v"]
            m["human_minus_vlm_distinct_rr"] = {"n": int(len(j)), "mean": float(diff.mean()), "ci": boot_mean_ci(diff)}
    if not cues.empty:
        m["cue_share"] = {k: float(cues["cues"].map(lambda c: k in c).mean()) for k in CUE_NAMES}
        m["median_seconds"] = {"generic": float(cues["t_generic_s"].median()), "distinct": float(cues["t_distinct_s"].median())}
        sib = hum[hum["kind"] == "distinct"].merge(cues, left_on=["who", "item_id"], right_on=["who", "item_id"])
        m["top1_of_4_by_confidence"] = {str(c): {"n": int(len(g)), "top1_of_4": float(g["top1_sib"].mean())}
                                        for c, g in sib.groupby("confidence")}
    (HUMAN / "metrics.json").write_text(json.dumps(m, indent=1))
    fig_selfret(rows)
    fig_cues(cues)
    summary(items, rows, cues, m)
    print((HUMAN / "summary.md").read_text()[:2500])


def summary(items, rows, cues, m):
    L = ["# Idea 5, Step 3: human simulation of the page describer", "",
         f"{m['n_items']} target pages (groups: {m['groups']}; A look-alike misses, B other misses, C hits with a close "
         "look-alike). Each describer wrote a description of the page alone (generic) and one next to its 3 most "
         "similar pages (distinct). Scores: ColQwen with the description as the query, within the document.", "",
         "| describer | kind | n | target 1st of 4 | target 1st of all | median rank | MRR | question words | words |",
         "|---|---|---|---|---|---|---|---|---|"]
    for d, kinds in m["by_describer"].items():
        for kind, v in kinds.items():
            L.append(f"| {d} | {kind} | {v['n']} | {v['top1_of_4']:.0%} | {v['top1_all']:.0%} | {v['median_rank_all']:.0f} "
                     f"| {v['mrr_all']:.3f} | {v['question_word_coverage']:.0%} | {v['median_words']:.0f} |")
    L += ["", "Chance for 'target 1st of 4' is 25%. 'question words': share of the real question's content words in "
          "the description (the question was never shown).", "", "## Distinct against generic (paired by item)", "",
          "| describer | n | change in reciprocal rank [95% CI] | better / same / worse | Wilcoxon p | change in question words |",
          "|---|---|---|---|---|---|"]
    for d, v in m["paired_generic_vs_distinct"].items():
        if v:
            L.append(f"| {d} | {v['n']} | {v['rr_change']:+.3f} [{v['rr_change_ci'][0]:+.3f}, {v['rr_change_ci'][1]:+.3f}] "
                     f"| {v['better']} / {v['same']} / {v['worse']} | {v['wilcoxon_p']:.3f} | {v['qcov_change']:+.0%} |")
    if "human_minus_vlm_distinct_rr" in m:
        h = m["human_minus_vlm_distinct_rr"]
        L += ["", f"Human minus VLM, distinct descriptions, reciprocal rank: {h['mean']:+.3f} "
              f"[{h['ci'][0]:+.3f}, {h['ci'][1]:+.3f}] over {h['n']} items. A positive gap is the room that training "
              "the describer with the self-retrieval reward would have to close."]
    if "cue_share" in m:
        L += ["", "## Cues used to tell the target apart (human)", "", "| cue | share |", "|---|---|"]
        for k, v in sorted(m["cue_share"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {CUE_NAMES[k]} | {v:.0%} |")
        L += ["", f"Median seconds per item: generic {m['median_seconds']['generic']:.0f}, distinct "
              f"{m['median_seconds']['distinct']:.0f}.",
              "Target 1st of 4 by confidence: " + ", ".join(f"{c}: {v['top1_of_4']:.0%} (n={v['n']})"
                                                         for c, v in m["top1_of_4_by_confidence"].items())]
    L += ["", "## Examples", ""]
    show = rows.sort_values(["item_id", "describer", "kind"])
    for item_id in list(show["item_id"].unique())[:4]:
        it = items[items["item_id"] == item_id].iloc[0]
        L.append(f"**{item_id}** ({it['doc_type']}, page {it['target']} of {it['n_pages']}, group {it['group']}). "
                 f"Real question: {it['question']}")
        L.append("")
        for r in show[show["item_id"] == item_id].itertuples():
            L.append(f"- {r.describer}, {r.kind} (1st of 4: {'yes' if r.top1_sib else 'no'}, rank {r.rank_all} of "
                     f"{r.n_pages}): {r.text}")
        L.append("")
    if not cues.empty and cues["notes"].astype(str).str.strip().any():
        L += ["## Annotator notes", ""] + [f"- {r.who} {r.item_id}: {r.notes}" for r in cues.itertuples()
                                           if str(r.notes).strip()]
    (HUMAN / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
