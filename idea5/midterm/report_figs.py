"""Figures for the report, from out/eval/ (run evaluate.py first).

  python report_figs.py [--qids 30,20] [--k 64]   # writes out/eval/fig_report_main.pdf and fig_report_examples.pdf

fig_report_main      (a) accuracy against the per-page budget K, all 829 questions, with the two references;
                     (b) where each selector's kept tokens fall at K = 64 (pages with a text layer)
fig_report_examples  one row per question: the gold page with SparseVLM's kept tokens (blue) and the residual
                     selector's (aqua)
"""
import argparse
import ast
import json
import textwrap

import pandas as pd
from PIL import Image

from shared import OUT, PAGES, PATCH, TEST

BLUE, AQUA, INK, INK2 = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e"


def main_fig(k=64):
    import matplotlib.pyplot as plt
    import numpy as np
    from conds import LABELS
    short = dict(LABELS, comp="residual (ours)", svlmnr="SparseVLM, no recycling", full="text + full images",
                 text="text layers only", down="downscaled pages")
    from evaluate import GRID, NEUTRAL, SLOT
    from shared import BUDGETS
    m = json.loads((OUT / "eval" / "metrics.json").read_text())
    acc, where, base = m["accuracy"], m["where_kept"], m["page_base"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.45), gridspec_kw={"width_ratios": [1, 1.1]})
    for c, ls, col in [("full", (0, (5, 2)), INK), ("text", (0, (2, 1.5)), INK2)]:
        a1.axhline(acc[c]["all"] * 100, color=col, linewidth=1, linestyle=ls, label=short[c])
    order = ["down", "comp", "ink", "svlm", "svlmnr", "fastv", "rand"]
    for meth in order:
        a1.plot(BUDGETS, [acc[f"{meth}_{b}"]["all"] * 100 for b in BUDGETS], color=SLOT[meth], linewidth=2,
                marker="o", markersize=3.5, linestyle="--" if meth == "svlmnr" else "-", label=short[meth])
    a1.set_xscale("log", base=2)
    a1.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    a1.set_xlabel("visual tokens kept per page (K)", fontsize=7, color=INK2)
    a1.set_ylabel("accuracy (%), 829 questions", fontsize=7, color=INK2)
    a1.set_title("(a) accuracy against the token budget", fontsize=7.5, color=INK)
    a1.legend(frameon=False, fontsize=5.3, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=2.0,
              columnspacing=0.8)
    a1.set_ylim(25, 52)
    rows = [("page (all tokens)", [base["words"], base["other_ink"], base["blank"]])]
    for meth in ["svlm", "svlmnr", "fastv", "rand", "ink", "comp"]:
        w = where[f"{meth}_{k}"]
        rows.append((short[meth], [w["words"], w["other_ink"], w["blank"]]))
    parts = [("on text-layer words", SLOT["fastv"]), ("other ink", SLOT["comp"]), ("blank", NEUTRAL)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, vals) in zip(y, rows):
        left = 0.0
        for (pname, col), v in zip(parts, vals):
            a2.barh(yi, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=1,
                    label=pname if yi == y[0] else None)
            if v >= 0.12:
                a2.text(left + v / 2, yi, f"{v * 100:.0f}", ha="center", va="center", fontsize=5.5, color=INK)
            left += v
    a2.set_yticks(y, [r[0] for r in rows], fontsize=5.8)
    a2.set_xlim(0, 1)
    a2.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    a2.set_title(f"(b) where the kept tokens fall, K = {k}", fontsize=7.5, color=INK)
    a2.legend(frameon=False, fontsize=5.3, ncol=3, loc="upper center", bbox_to_anchor=(0.4, -0.24))
    for ax, g in ((a1, "y"), (a2, "x")):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=6)
        ax.grid(axis=g, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "eval" / "fig_report_main.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "eval" / "fig_report_main.pdf")


def main(qids, k):
    import matplotlib.pyplot as plt
    items = {it["qid"]: it for it in map(json.loads, (TEST / "items.jsonl").open())}
    df = pd.read_csv(OUT / "eval" / "scored.csv")
    df = df[df["qid"].isin(qids) & df["cond"].isin([f"svlm_{k}", f"comp_{k}"])].set_index(["qid", "cond"])
    aspects = []
    for q in qids:
        w, h = Image.open(PAGES / f"{items[q]['page_keys'][0]}.png").size
        aspects.append(h / w)
    heights = [1.6 * a_ + 0.75 for a_ in aspects]  # two images of width 1.6 in per row, plus titles and caption
    fig = plt.figure(figsize=(3.3, sum(heights)), layout="constrained")
    for sub, q in zip(fig.subfigures(len(qids), 1, height_ratios=heights), qids):
        it = items[q]
        assert it["n_gold"] == 1, "single-page questions only"
        img = Image.open(PAGES / f"{it['page_keys'][0]}.png").convert("RGB")
        cols = img.size[0] // PATCH
        axes = sub.subplots(1, 2)
        for ax, (cond, col, name) in zip(axes, [(f"svlm_{k}", BLUE, "SparseVLM"), (f"comp_{k}", AQUA, "residual (ours)")]):
            r = df.loc[(q, cond)]
            ax.imshow(img)
            for t in ast.literal_eval(r["kept"]):
                rr, cc = divmod(int(t), cols)
                ax.add_patch(plt.Rectangle((cc * PATCH, rr * PATCH), PATCH, PATCH, facecolor=col, alpha=0.55,
                                           edgecolor="none"))
            mark = "right" if r["correct"] == 1 else "wrong"
            ax.set_title(f"{name}, K={k}\n\"{str(r['pred'])[:30]}\" ({mark})", fontsize=5.5, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_color("#a3a29c")
        sub.supxlabel(textwrap.fill(f"Q: {it['question']} Gold: {it['answer']}", 80), fontsize=5.2, color=INK2)
    fig.savefig(OUT / "eval" / "fig_report_examples.pdf")
    plt.close(fig)
    print("wrote", OUT / "eval" / "fig_report_examples.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="30,20")
    ap.add_argument("--k", type=int, default=64)
    a = ap.parse_args()
    main_fig(a.k)
    main([int(q) for q in a.qids.split(",")], a.k)
