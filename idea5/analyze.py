"""Part B, step 3 (CPU): score the reader's answers and test the hypotheses.

  python analyze.py     # reads out/sample/items.jsonl and out/reader/*.jsonl, writes out/analysis/

Hypotheses (README.md):
  H1  text alone (text layer T, or the VLM's transcription X) falls short of text + image on chart, figure and layout
      questions, but not on text questions.
  H2  interaction responses differ by evidence type (from the correctness of T, I_hi and TI_hi per question):
      equivalence (all right), text-dominant (T and TI right, I wrong), image-dominant (I and TI right, T wrong),
      emergence (only TI right), interference (T or I right, TI wrong), none (all wrong).
      Regression per type, as in the lecture: y = w0 + w1 I + w2 T + w3 I*T over the cells Q, I_hi, T, TI_hi.
  H3  modulation: lowering the image to 256 tokens costs less with the text layer than without it,
      and the residual: text + words-masked image (TM_lo) is as good as text + image (TI_lo).
Confidence intervals: 2,000 bootstrap resamples of questions, seed 555.
"""
import json

import numpy as np
import pandas as pd

from common import OUT, SEED, TYPES
from score import score

READER = OUT / "reader"
ANA = OUT / "analysis"
CONDS = ["Q", "T", "X", "I_hi", "I_lo", "TI_hi", "TI_lo", "M_lo", "TM_lo"]
LABELS = {"Q": "question only", "T": "text layer", "X": "VLM transcription", "I_hi": "image (1280)",
          "I_lo": "image (256)", "TI_hi": "text + image (1280)", "TI_lo": "text + image (256)",
          "M_lo": "masked image (256)", "TM_lo": "text + masked image (256)"}
CLASSES = ["equivalence", "text-dominant", "image-dominant", "emergence", "interference", "none"]
N_BOOT = 2000
# validated categorical slots 1-5 (reference palette, light), neutral for "none"; ink and grid tokens
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
NEUTRAL, INK, INK2, GRID = "#c3c2b7", "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load():
    items = pd.DataFrame([json.loads(l) for l in (OUT / "sample" / "items.jsonl").open()])
    rows = [json.loads(l) for p in sorted(READER.glob("answers_*.jsonl")) for l in p.open()]
    ans = pd.DataFrame(rows).drop_duplicates(["qid", "cond"], keep="last")
    df = ans.merge(items[["qid", "type", "answer", "answer_format", "no_text_layer", "doc_type"]], on="qid")
    df["score"] = [score(a, p, f) for a, p, f in zip(df["answer"], df["pred"], df["answer_format"])]
    df["correct"] = (df["score"] > 0).astype(float)
    return items, df


def wide(df, col):
    return df.pivot(index="qid", columns="cond", values=col)


def boot(f, idx, n=N_BOOT):
    """Percentile CI of statistic f over bootstrap resamples of the question index."""
    rng = np.random.default_rng(SEED)
    idx = np.asarray(idx)
    b = [f(idx[rng.integers(len(idx), size=len(idx))]) for _ in range(n)]
    return [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


def accuracy_table(df, items):
    acc = wide(df, "correct")
    out = {}
    for t in TYPES + ["all"]:
        q = items["qid"] if t == "all" else items.loc[items["type"] == t, "qid"]
        a = acc.loc[acc.index.intersection(q)]
        out[t] = {c: {"acc": float(a[c].mean()),
                      "ci": boot(lambda ix: a.loc[ix, c].mean(), a.index) if t == "all" else None}
                  for c in CONDS if c in a}
        out[t]["n"] = int(len(a))
    return out


def classes(acc):
    t, i, ti = acc["T"], acc["I_hi"], acc["TI_hi"]
    lab = np.select([(t == 1) & (i == 1) & (ti == 1), (t == 1) & (i == 0) & (ti == 1), (t == 0) & (i == 1) & (ti == 1),
                     (t == 0) & (i == 0) & (ti == 1), ((t == 1) | (i == 1)) & (ti == 0)], CLASSES[:5], "none")
    return pd.Series(lab, index=acc.index)


def regression(y):
    """Coefficients of y = w0 + w1 I + w2 T + w3 I*T on the four cells (exact in a balanced 2x2 within questions)."""
    q, i, t, ti = (y[c].mean() for c in ["Q", "I_hi", "T", "TI_hi"])
    return {"w0": q, "w1_image": i - q, "w2_text": t - q, "w3_interaction": ti - i - t + q}


def regression_table(df, items):
    out = {}
    for col in ["correct", "lp_mean"]:
        y = wide(df, col)
        out[col] = {}
        for t in TYPES + ["all"]:
            q = items["qid"] if t == "all" else items.loc[items["type"] == t, "qid"]
            yy = y.loc[y.index.intersection(q)]
            est = regression(yy)
            out[col][t] = {k: {"est": float(v), "ci": boot(lambda ix, k=k: regression(yy.loc[ix])[k], yy.index)}
                           for k, v in est.items()}
    return out


def paired(acc, a, b, idx):
    """Mean of (a - b) over questions idx, with CI and better/same/worse counts."""
    d = (acc.loc[idx, a] - acc.loc[idx, b])
    return {"n": int(len(d)), "diff": float(d.mean()), "ci": boot(lambda ix: d.loc[ix].mean(), d.index),
            "better": int((d > 0).sum()), "same": int((d == 0).sum()), "worse": int((d < 0).sum())}


def hypotheses(df, items):
    acc = wide(df, "correct")
    has_text = items.loc[~items["no_text_layer"], "qid"]
    h = {"H1": {}, "H3": {}}
    for t in TYPES + ["all"]:
        q = acc.index.intersection(items["qid"] if t == "all" else items.loc[items["type"] == t, "qid"])
        h["H1"][t] = {"T_vs_TI_hi": paired(acc, "T", "TI_hi", q), "X_vs_TI_hi": paired(acc, "X", "TI_hi", q)}
    q = acc.index.intersection(has_text)
    res_drop_no_text = acc.loc[q, "I_hi"] - acc.loc[q, "I_lo"]
    res_drop_text = acc.loc[q, "TI_hi"] - acc.loc[q, "TI_lo"]
    gap = res_drop_no_text - res_drop_text
    h["H3"] = {"n_with_text_layer": int(len(q)),
               "drop_without_text": {"diff": float(res_drop_no_text.mean()), "ci": boot(lambda ix: res_drop_no_text.loc[ix].mean(), q)},
               "drop_with_text": {"diff": float(res_drop_text.mean()), "ci": boot(lambda ix: res_drop_text.loc[ix].mean(), q)},
               "modulation": {"diff": float(gap.mean()), "ci": boot(lambda ix: gap.loc[ix].mean(), q)},
               "TM_lo_vs_TI_lo": paired(acc, "TM_lo", "TI_lo", q), "TM_lo_vs_T": paired(acc, "TM_lo", "T", q),
               "M_lo_vs_I_lo": paired(acc, "M_lo", "I_lo", q)}
    nt = acc.index.intersection(items.loc[items["no_text_layer"], "qid"])
    h["no_text_layer"] = {"n": int(len(nt)), **{c: float(acc.loc[nt, c].mean()) for c in CONDS}}
    return h


# ---------- figures ----------

def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)


def fig_accuracy(tab):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cols = TYPES + ["all"]
    M = np.array([[tab[t][c]["acc"] for t in cols] for c in CONDS])
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            ax.text(c, r, f"{M[r, c] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[r, c] >= 0.55 else INK)
    ax.set_xticks(range(len(cols)), [f"{t}\n{tab[t]['n']}" for t in cols], fontsize=5.8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(CONDS)), [LABELS[c] for c in CONDS], fontsize=6.5)
    ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CONDS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.axvline(len(cols) - 1.5, color="white", linewidth=3)
    fig.tight_layout()
    fig.savefig(ANA / "fig_accuracy.pdf")
    plt.close(fig)


def fig_classes(cls, items):
    import matplotlib.pyplot as plt
    cols = TYPES + ["all"]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    y = np.arange(len(cols))[::-1]
    for t, yi in zip(cols, y):
        q = cls.index.intersection(items["qid"] if t == "all" else items.loc[items["type"] == t, "qid"])
        share = cls.loc[q].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
        left = 0.0
        for k, c in enumerate(CLASSES):
            ax.barh(yi, share[c], left=left, height=0.62, color=(SLOTS + [NEUTRAL])[k], edgecolor="white",
                    linewidth=1.0, label=c if t == cols[0] else None)
            if share[c] >= 0.12:
                ax.text(left + share[c] / 2, yi, f"{share[c] * 100:.0f}", ha="center", va="center", fontsize=6,
                        color=INK)
            left += share[c]
    ax.set_yticks(y, [f"{t} ({len(cls.index.intersection(items['qid'] if t == 'all' else items.loc[items['type'] == t, 'qid']))})"
                      for t in cols], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    style(ax)
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              handlelength=1.0, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(ANA / "fig_classes.pdf")
    plt.close(fig)


# ---------- summary ----------

def pct(x):
    return f"{x * 100:+.0f}" if x is not None else ""


def ci(c):
    return f"[{c[0] * 100:+.0f}, {c[1] * 100:+.0f}]"


def summary(items, df, tab, reg, hyp, cls, tok):
    L = ["# Part B: what does the page image add beyond the text layer?", "",
         f"{len(items)} questions (gold page only; {int(items['no_text_layer'].sum())} pages have no text layer). "
         "Reader: Qwen2.5-VL-7B, greedy short answers. Accuracy in % (MMLongBench-Doc rules; correct = score > 0).", "",
         "| condition | " + " | ".join(f"{t} (n={tab[t]['n']})" for t in TYPES + ["all"]) + " | 95% CI (all) | input tokens |",
         "|---|" + "---|" * (len(TYPES) + 3)]
    for c in CONDS:
        L.append(f"| {LABELS[c]} | " + " | ".join(f"{tab[t][c]['acc'] * 100:.0f}" for t in TYPES + ["all"])
                 + f" | [{tab['all'][c]['ci'][0] * 100:.0f}, {tab['all'][c]['ci'][1] * 100:.0f}] | {tok[c]:.0f} |")
    L += ["", "## H1: text alone against text + image (1280), paired, percentage points", "",
          "| type | text layer - text+image [CI] | b/s/w | transcription - text+image [CI] | b/s/w |", "|---|---|---|---|---|"]
    for t in TYPES + ["all"]:
        a, b = hyp["H1"][t]["T_vs_TI_hi"], hyp["H1"][t]["X_vs_TI_hi"]
        L.append(f"| {t} | {pct(a['diff'])} {ci(a['ci'])} | {a['better']}/{a['same']}/{a['worse']} | "
                 f"{pct(b['diff'])} {ci(b['ci'])} | {b['better']}/{b['same']}/{b['worse']} |")
    L += ["", "## H2: interaction responses (from T, I_hi, TI_hi)", "",
          "| type | n | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * len(CLASSES)]
    for t in TYPES + ["all"]:
        q = cls.index.intersection(items["qid"] if t == "all" else items.loc[items["type"] == t, "qid"])
        vc = cls.loc[q].value_counts().reindex(CLASSES, fill_value=0)
        L.append(f"| {t} | {len(q)} | " + " | ".join(str(int(v)) for v in vc.values) + " |")
    for col, name in [("correct", "accuracy, points"), ("lp_mean", "mean log-prob of the reference answer")]:
        L += ["", f"Regression y = w0 + w1 I + w2 T + w3 I*T ({name}), estimate [95% CI]:", "",
              "| type | w0 | w1 image | w2 text | w3 interaction |", "|---|---|---|---|---|"]
        for t in TYPES + ["all"]:
            r = reg[col][t]
            f = (lambda v: f"{v['est'] * 100:+.0f} {ci(v['ci'])}") if col == "correct" else \
                (lambda v: f"{v['est']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")
            L.append(f"| {t} | " + " | ".join(f(r[k]) for k in ["w0", "w1_image", "w2_text", "w3_interaction"]) + " |")
    h3 = hyp["H3"]
    L += ["", f"## H3 (pages with a text layer, n = {h3['n_with_text_layer']}), percentage points", "",
          f"- Accuracy lost going from 1280 to 256 image tokens, without text: {pct(h3['drop_without_text']['diff'])} "
          f"{ci(h3['drop_without_text']['ci'])}; with the text layer: {pct(h3['drop_with_text']['diff'])} "
          f"{ci(h3['drop_with_text']['ci'])}; difference (modulation): {pct(h3['modulation']['diff'])} {ci(h3['modulation']['ci'])}."]
    for k, lab in [("TM_lo_vs_TI_lo", "text + masked image - text + image (256)"),
                   ("TM_lo_vs_T", "text + masked image - text layer alone"),
                   ("M_lo_vs_I_lo", "masked image - image (256)")]:
        v = h3[k]
        L.append(f"- {lab}: {pct(v['diff'])} {ci(v['ci'])}, better/same/worse {v['better']}/{v['same']}/{v['worse']}.")
    nt = hyp["no_text_layer"]
    L += ["", f"Pages without a text layer (n = {nt['n']}): " + ", ".join(f"{LABELS[c]} {nt[c] * 100:.0f}" for c in CONDS) + "."]
    L += ["", "Reference-answer tokens aligned with the tokenized prompt: "
          f"{df['aligned'].mean():.0%} of rows (log-prob rows that are not aligned are still scored)."]
    (ANA / "summary.md").write_text("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    ANA.mkdir(parents=True, exist_ok=True)
    items, df = load()
    missing = len(items) * len(CONDS) - len(df)
    if missing:
        print(f"WARNING: {missing} (question, condition) rows missing")
    df.drop(columns=["answer"]).to_csv(ANA / "answers_scored.csv", index=False)
    tab = accuracy_table(df, items)
    reg = regression_table(df, items)
    hyp = hypotheses(df, items)
    cls = classes(wide(df, "correct"))
    tok = df.groupby("cond")["input_tokens"].mean()
    (ANA / "metrics.json").write_text(json.dumps({"accuracy": tab, "regression": reg, "hypotheses": hyp,
                                                  "classes": cls.value_counts().to_dict(),
                                                  "input_tokens": tok.to_dict()}, indent=1))
    fig_accuracy(tab)
    fig_classes(cls, items)
    print(summary(items, df, tab, reg, hyp, cls, tok))


if __name__ == "__main__":
    main()
