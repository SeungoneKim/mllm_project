"""Step 3 (CPU): score the reader's answers, compare the token selectors and analyse their failures.

  python evaluate.py     # reads out/test/, out/reader/answers_*.jsonl; writes out/eval/

Accuracy uses the analysis's rule-based MMLongBench-Doc scorer (correct = score > 0). Question groups:
  all; single- and multi-page; each evidence type (questions with that single type); visual (any chart, figure or
  layout evidence); text layer (every gold page has one) / no text layer (at least one gold page has none);
  image-needed (text only wrong, text + full image right) and text-sufficient (text only right). The last two are defined by the two reference conditions, so selectors are compared on them
  but the reference conditions themselves are not.
Selection diagnostics, from each kept token's 28 x 28 block (out/test/tokinfo.npz): the share of kept tokens on
text-layer words (txt > 0.5), on other ink (ink >= 0.02), and blank; and the share of the K x N blocks with the most
ink outside the text layer (the comp selection) that the selector also keeps. Indices run over the gold pages'
visual tokens page after page, as in the reader.
Response to the image, against text only (T) and a selector's input (S), as in the analysis: recovered (T wrong,
S right) and interference (T right, S wrong).
Confidence intervals: 2,000 bootstrap resamples of questions, seed 555.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from conds import LABELS, MODEL_METHODS, SELECTORS, conditions, parse
from shared import BUDGETS, OUT, PAGES, SEED, TEST, TYPES, score

READER = Path(os.environ.get("IDEA5_MT_READER", OUT / "reader"))
EVAL = Path(os.environ.get("IDEA5_MT_EVAL", OUT / "eval"))
N_BOOT = 2000
TXT_ON, INK_ON = 0.5, 0.02
VISUAL = {"chart", "figure", "layout"}
# validated categorical slots (reference palette, light), neutral, ink, grid
SLOT = {"svlm": "#2a78d6", "svlmnr": "#2a78d6", "fastv": "#eb6834", "comp": "#1baf7a", "ink": "#eda100",
        "down": "#e87ba4", "rand": "#a3a29c"}
NEUTRAL, INK, INK2, GRID = "#c3c2b7", "#0b0b0b", "#52514e", "#e6e5e0"


# ---------- data ----------

def load():
    items = pd.DataFrame([json.loads(l) for l in (TEST / "items.jsonl").open()])
    rows = [json.loads(l) for p in sorted(READER.glob("answers_*.jsonl")) for l in p.open()]
    ans = pd.DataFrame(rows).drop_duplicates(["qid", "cond"], keep="last")
    df = ans.merge(items[["qid", "answer", "answer_format", "ev_types", "single_type", "no_text_layer", "doc_type",
                          "page_keys", "n_gold"]], on="qid")
    df["score"] = [score(a, p, f) for a, p, f in zip(df["answer"], df["pred"], df["answer_format"])]
    df["correct"] = (df["score"] > 0).astype(float)
    parsed = df["cond"].map(parse)
    df["method"] = [m if m else ("down" if v.startswith("down") else None) for v, m, _ in parsed]
    df["k"] = [k for _, _, k in parsed]
    return items, df


def groups(items, acc):
    """Question groups: name -> qids."""
    ev = items.set_index("qid")["ev_types"]
    single = items.set_index("qid")["single_type"]
    nt = items.set_index("qid")["no_text_layer"]
    n_gold = items.set_index("qid")["n_gold"]
    g = {"all": list(items["qid"]), "single-page": [q for q in items["qid"] if n_gold[q] == 1],
         "multi-page": [q for q in items["qid"] if n_gold[q] > 1]}
    for t in TYPES:
        g[t] = [q for q in items["qid"] if single[q] and ev[q] == [t]]
    g["visual"] = [q for q in items["qid"] if VISUAL & set(ev[q])]
    g["text layer"] = [q for q in items["qid"] if not nt[q]]
    g["no text layer"] = [q for q in items["qid"] if nt[q]]
    if {"text", "full"} <= set(acc.columns):
        both = acc[["text", "full"]].dropna()
        g["image-needed"] = list(both.index[(both["text"] == 0) & (both["full"] == 1)])
        g["text-sufficient"] = list(both.index[both["text"] == 1])
    return g


def boot(f, idx, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    idx = np.asarray(idx)
    if len(idx) < 2:
        return [float("nan")] * 2
    b = [f(idx[rng.integers(len(idx), size=len(idx))]) for _ in range(n)]
    return [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


# ---------- tables ----------

def accuracy_table(df, acc, g):
    conds = [c for c in conditions() if c in acc.columns]
    tokens = df.groupby("cond")[["vis_per_layer", "input_tokens", "seq_after"]].mean()
    out = {}
    for c in conds:
        r = {name: float(acc.loc[acc.index.intersection(q), c].mean()) if len(q) else float("nan")
             for name, q in g.items()}
        r["ci_all"] = boot(lambda ix: acc.loc[ix, c].mean(), acc[c].dropna().index)
        r["vis_per_layer"] = float(tokens.loc[c, "vis_per_layer"])
        r["seq_after"] = float(tokens.loc[c, "seq_after"])
        out[c] = r
    return out


def paired(acc, a, b, qids):
    q = acc.index.intersection(qids)
    d = (acc.loc[q, a] - acc.loc[q, b]).dropna()
    return {"n": int(len(d)), "diff": float(d.mean()), "ci": boot(lambda ix: d.loc[ix].mean(), d.index),
            "better": int((d > 0).sum()), "same": int((d == 0).sum()), "worse": int((d < 0).sum())}


def comparisons(acc, g):
    pairs = [("svlm", "text"), ("svlm", "down"), ("svlm", "rand"), ("svlm", "fastv"), ("svlm", "svlmnr"),
             ("comp", "svlm"), ("comp", "ink"), ("comp", "down"), ("full", "svlm")]
    out = {}
    for k in BUDGETS:
        for a, b in pairs:
            ca = a if a in ("text", "full") else f"{a}_{k}"
            cb = b if b in ("text", "full") else f"{b}_{k}"
            if ca in acc and cb in acc:
                for name in ("all", "multi-page", "visual", "image-needed"):
                    if name in g:
                        out[f"{ca} - {cb} ({name})"] = paired(acc, ca, cb, g[name])
    return out


def response(acc, g):
    """Recovered / interference counts of every condition against text only."""
    out = {}
    if "text" not in acc:
        return out
    for c in acc.columns:
        if c == "text":
            continue
        both = acc[["text", c]].dropna()
        out[c] = {"recovered": int(((both["text"] == 0) & (both[c] == 1)).sum()),
                  "interference": int(((both["text"] == 1) & (both[c] == 0)).sum()), "n": int(len(both))}
    return out


def where_kept(df, tok):
    """Where each selector's kept tokens fall, on pages with a text layer."""
    rows = []
    sel = df[df["kept"].notna() & ~df["no_text_layer"]]
    base = {}
    for r in sel.itertuples():
        ink = np.concatenate([tok[f"{key}_ink"].astype(np.float32).ravel() for key in r.page_keys])
        txt = np.concatenate([tok[f"{key}_txt"].astype(np.float32).ravel() for key in r.page_keys])
        kept = np.asarray(r.kept, dtype=int)
        on_words = txt[kept] > TXT_ON
        blank = ~on_words & (ink[kept] < INK_ON)
        k = int(r.k) * int(r.n_gold)
        comp = np.argsort(-(ink * (1 - txt)), kind="stable")[:k]
        rows.append({"cond": r.cond, "method": r.method, "k": int(r.k), "words": on_words.mean(),
                     "other_ink": (~on_words & ~blank).mean(), "blank": blank.mean(),
                     "residual_kept": len(np.intersect1d(kept, comp)) / max(k, 1), "correct": r.correct,
                     "qid": r.qid})
        for key in r.page_keys:
            if key not in base:
                pi, pt = tok[f"{key}_ink"].astype(np.float32).ravel(), tok[f"{key}_txt"].astype(np.float32).ravel()
                w = pt > TXT_ON
                base[key] = (w.mean(), (~w & (pi >= INK_ON)).mean(), (~w & (pi < INK_ON)).mean())
    w = pd.DataFrame(rows)
    page_base = np.mean(list(base.values()), axis=0) if base else [np.nan] * 3
    return w, {"words": float(page_base[0]), "other_ink": float(page_base[1]), "blank": float(page_base[2])}


# ---------- figures ----------

def style(ax, grid_axis="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_budget(tab):
    import matplotlib.pyplot as plt
    panels = [("all", "all questions"), ("visual", "chart, figure or layout evidence")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, (g, title) in zip(axes, panels):
        for c, lab, ls in [("text", "text only", (0, (2, 1.5))), ("full", "text + full image", (0, (5, 2)))]:
            if c in tab:
                ax.axhline(tab[c][g], color=INK2 if c == "text" else INK, linewidth=1, linestyle=ls, label=lab)
        for m in SELECTORS:
            ys = [tab.get(f"{m}_{k}", {}).get(g, np.nan) for k in BUDGETS]
            if np.all(np.isnan(ys)):
                continue
            ax.plot(BUDGETS, ys, color=SLOT[m], linewidth=2, marker="o", markersize=4,
                    linestyle="--" if m == "svlmnr" else "-", label=LABELS[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(k) for k in BUDGETS])
        ax.set_xlabel("visual tokens kept per page (K)", fontsize=7, color=INK2)
        ax.set_title(title, fontsize=8, color=INK)
        style(ax)
    axes[0].set_ylabel("accuracy", fontsize=7, color=INK2)
    axes[1].legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(EVAL / "fig_budget.pdf")
    plt.close(fig)


def fig_where(w, base):
    import matplotlib.pyplot as plt
    if w.empty:
        return
    order = [m for m in ("svlm", "svlmnr", "fastv", "rand", "ink", "comp") if m in set(w["method"])]
    parts = [("words", "#eb6834", "on text-layer words"), ("other_ink", "#1baf7a", "other ink"),
             ("blank", NEUTRAL, "blank")]
    ks = [k for k in (BUDGETS[0], BUDGETS[-1]) if k in set(w["k"])]
    fig, axes = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 2.0), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        labels = ["page (all tokens)"] + [LABELS[m] for m in order]
        vals = [[base[p] for p, _, _ in parts]] + [
            [w[(w["method"] == m) & (w["k"] == k)][p].mean() for p, _, _ in parts] for m in order]
        y = np.arange(len(labels))[::-1]
        for yi, v in zip(y, vals):
            left = 0.0
            for (p, col, lab), x in zip(parts, v):
                ax.barh(yi, x, left=left, height=0.62, color=col, edgecolor="white", linewidth=1,
                        label=lab if yi == y[0] else None)
                if x >= 0.12:
                    ax.text(left + x / 2, yi, f"{x * 100:.0f}", ha="center", va="center", fontsize=6, color=INK)
                left += x
        ax.set_yticks(y, labels, fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_title(f"K = {k}", fontsize=8, color=INK)
        style(ax, "x")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(EVAL / "fig_where.pdf")
    plt.close(fig)


def fig_examples(df, items, g, k=BUDGETS[0], n=2):
    """Single-page image-needed questions where SparseVLM and the residual selector disagree, with the kept tokens."""
    import matplotlib.pyplot as plt
    import textwrap
    from PIL import Image
    a, b = f"svlm_{k}", f"comp_{k}"
    single = set(g.get("image-needed", [])) & set(g["single-page"])
    sub = df[df["cond"].isin([a, b]) & df["qid"].isin(single)]
    wide = sub.pivot(index="qid", columns="cond", values="correct").dropna()
    picks = list(wide.index[(wide[a] == 0) & (wide[b] == 1)][:n]) + list(wide.index[(wide[a] == 1) & (wide[b] == 0)][:1])
    if not picks:
        return []
    kept = sub.set_index(["qid", "cond"])["kept"]
    it = items.set_index("qid")
    fig, axes = plt.subplots(len(picks), 2, figsize=(6.6, 3.9 * len(picks)), squeeze=False)
    for row, q in zip(axes, picks):
        img = Image.open(PAGES / f"{it.loc[q, 'page_keys'][0]}.png").convert("RGB")
        cols = img.size[0] // 28
        for ax, c, col in [(row[0], a, "#2a78d6"), (row[1], b, "#1baf7a")]:
            ax.imshow(img)
            for j in kept[(q, c)]:
                r_, c_ = divmod(int(j), cols)
                ax.add_patch(plt.Rectangle((c_ * 28, r_ * 28), 28, 28, facecolor=col, alpha=0.45, edgecolor="none"))
            ok = "right" if wide.loc[q, c] == 1 else "wrong"
            short = {"svlm": "SparseVLM", "comp": "residual (ours, untrained)"}[c.rsplit("_", 1)[0]]
            ax.set_title(f"{short}, K={k}: {ok}", fontsize=7, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].text(0, -0.08, textwrap.fill(f"Q: {it.loc[q, 'question']}  (answer: {it.loc[q, 'answer']})", 110),
                    transform=row[0].transAxes, fontsize=6, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(EVAL / "fig_examples.pdf")
    plt.close(fig)
    return [int(q) for q in picks]


# ---------- summary ----------

def pct(x):
    return "" if x is None or np.isnan(x) else f"{x * 100:.0f}"


def summary(items, df, tab, g, comp, resp, w, base, examples):
    L = ["# Idea 5 midterm: text layer + K selected visual tokens per page", "",
         f"{len(items)} questions ({sum(items['n_gold'] == 1)} single-page; {sum(items['no_text_layer'])} with a gold "
         "page without a text layer). Reader: Qwen2.5-VL-7B on the gold pages, greedy. Accuracy in % (rule-based "
         "scorer, correct = score > 0). K is per page; the selectors choose K x N tokens across a question's N pages.", "",
         "Group sizes: " + ", ".join(f"{k} {len(v)}" for k, v in g.items()), ""]
    cols = ["all", "single-page", "multi-page", "visual", "text layer", "no text layer", "image-needed",
            "text-sufficient"] + list(TYPES)
    cols = [c for c in cols if c in g]
    L += ["| condition | " + " | ".join(cols) + " | 95% CI (all) | visual tokens / layer | sequence |",
          "|---|" + "---|" * (len(cols) + 3)]
    for c, r in tab.items():
        L.append(f"| {c} | " + " | ".join(pct(r[x]) for x in cols)
                 + f" | [{pct(r['ci_all'][0])}, {pct(r['ci_all'][1])}] | {r['vis_per_layer']:.0f} | {r['seq_after']:.0f} |")
    L += ["", "## Paired differences (percentage points)", "",
          "| comparison | n | diff [95% CI] | better / same / worse |", "|---|---|---|---|"]
    for name, v in comp.items():
        L.append(f"| {name} | {v['n']} | {v['diff'] * 100:+.1f} [{v['ci'][0] * 100:+.1f}, {v['ci'][1] * 100:+.1f}] "
                 f"| {v['better']} / {v['same']} / {v['worse']} |")
    L += ["", "## Response to the image, against text only", "", "| condition | recovered | interference | n |",
          "|---|---|---|---|"]
    for c, v in resp.items():
        L.append(f"| {c} | {v['recovered']} | {v['interference']} | {v['n']} |")
    if not w.empty:
        L += ["", "## Where the kept tokens fall (pages with a text layer)", "",
              f"All tokens of these pages: {pct(base['words'])}% on text-layer words, {pct(base['other_ink'])}% "
              f"other ink, {pct(base['blank'])}% blank.", "",
              "| condition | on words | other ink | blank | residual kept | residual kept, right / wrong (image-needed) |",
              "|---|---|---|---|---|---|"]
        imgq = set(g.get("image-needed", []))
        for c, s in w.groupby("cond", sort=False):
            si = s[s["qid"].isin(imgq)]
            L.append(f"| {c} | {pct(s['words'].mean())} | {pct(s['other_ink'].mean())} | {pct(s['blank'].mean())} "
                     f"| {pct(s['residual_kept'].mean())} | {pct(si[si['correct'] == 1]['residual_kept'].mean())} / "
                     f"{pct(si[si['correct'] == 0]['residual_kept'].mean())} |")
    if examples:
        it = items.set_index("qid")
        L += ["", "## Examples in fig_examples.pdf", ""]
        for q in examples:
            preds = df[(df["qid"] == q) & df["cond"].isin(["text", "full", f"svlm_{BUDGETS[0]}", f"comp_{BUDGETS[0]}"])]
            L.append(f"- qid {q} ({it.loc[q, 'doc_type']}, {', '.join(it.loc[q, 'ev_types'])}): {it.loc[q, 'question']} "
                     f"Answer: {it.loc[q, 'answer']}. " + "; ".join(f"{r.cond}: {r.pred}" for r in preds.itertuples()))
    (EVAL / "summary.md").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    EVAL.mkdir(parents=True, exist_ok=True)
    items, df = load()
    expected = len(items) * len(conditions())
    if len(df) < expected:
        print(f"WARNING: {expected - len(df)} of {expected} (question, condition) rows missing")
    acc = df.pivot(index="qid", columns="cond", values="correct")
    g = groups(items, acc)
    tab = accuracy_table(df, acc, g)
    comp = comparisons(acc, g)
    resp = response(acc, g)
    tok = np.load(TEST / "tokinfo.npz")
    w, base = where_kept(df, tok)
    fig_budget(tab)
    fig_where(w, base)
    examples = fig_examples(df, items, g)
    df.drop(columns=["answer"]).to_csv(EVAL / "scored.csv", index=False)
    (EVAL / "metrics.json").write_text(json.dumps({
        "groups": {k: len(v) for k, v in g.items()}, "accuracy": tab, "paired": comp, "response": resp,
        "where_kept": w.drop(columns=["qid"]).groupby("cond").mean(numeric_only=True).to_dict(orient="index")
        if not w.empty else {}, "page_base": base, "examples": examples}, indent=1))
    print(summary(items, df, tab, g, comp, resp, w, base, examples)[:6000])


if __name__ == "__main__":
    main()
