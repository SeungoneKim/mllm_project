"""Step 4: Part A analysis.

  python analyze_A.py annotations_*.json                 # writes out/partA/
  python analyze_A.py --allow-synthetic tests/*SYNTHETIC*.json   # writes tests/out_SYNTHETIC/ only

Practice items are ignored. Files whose name or annotator contains SYNTHETIC are refused
unless --allow-synthetic is given, and synthetic results never go to out/partA/.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import cohen_kappa_score

from common import IDEA4, MIN_TEXT_WORDS, OUT, doc_texts, n_words

GOLD_TYPES = ["chart", "table", "figure", "text"]          # gold types in the sample
PRED_TYPES = ["chart", "table", "figure", "text", "layout"]  # form choices
SPEC_WORD_FIELDS = ["title", "x_label", "y_label", "categories", "series", "caption", "columns", "rows",
                    "heading", "passage", "description"]
STEPS = ["step1", "step2", "step3"]

# Figure style: one column, 8 pt, colour-blind-safe slots 1 and 2 of the reference palette.
C_STEP1, C_STEP2 = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------- loading ----------

def is_synthetic(path, data):
    return "SYNTHETIC" in Path(path).name or "SYNTHETIC" in str(data.get("annotator", ""))


def load(paths, allow_synthetic):
    sample = pd.read_csv(OUT / "sample.csv").set_index("item_id")
    rows, synthetic = [], False
    names = set()
    for p in paths:
        data = json.loads(Path(p).read_text())
        if is_synthetic(p, data):
            if not allow_synthetic:
                sys.exit(f"refusing synthetic file {p} (use --allow-synthetic for tests)")
            synthetic = True
        ann = data["annotator"]
        if ann in names:
            sys.exit(f"two files for annotator {ann}: pass only the latest export")
        names.add(ann)
        for it in data["items"]:
            if it.get("practice") or it["item_id"].startswith("P"):
                continue
            g = sample.loc[it["item_id"]]
            r = {"annotator": ann, "item_id": it["item_id"], "doc_id": g["doc_id"], "gold_type": g["gold_type"],
                 "gold_page": int(g["gold_page"]), "question": g["question"],
                 "inventory_source": data.get("inventory_source", "")}
            for s in STEPS:
                st = it.get(s)
                r[f"{s}_done"] = st is not None
                r[f"{s}_t"] = (st["t_close"] - st["t_open"]) / 1000 if st else np.nan
            for s in ["step1", "step2"]:
                st = it.get(s) or {}
                r[f"{s}_type"] = st.get("type")
                r[f"{s}_render"] = st.get("render")
                r[f"{s}_spec"] = st.get("spec")
                r[f"{s}_correct"] = st.get("type") == g["gold_type"] if st else None
            s1, s2, s3 = it.get("step1") or {}, it.get("step2") or {}, it.get("step3") or {}
            r.update({"cues": s1.get("cues"), "confidence": s1.get("confidence"), "changed": s2.get("changed"),
                      "subtype_match": s3.get("subtype_match"), "label_ok": s3.get("label_ok"),
                      "bbox": s3.get("bbox"), "no_clear_region": s3.get("no_clear_region"), "notes": s3.get("notes")})
            rows.append(r)
    return pd.DataFrame(rows), sample, synthetic


# ---------- statistics ----------

def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan, np.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


def acc_table(df, col):
    out = {}
    for t in GOLD_TYPES + ["all"]:
        g = df if t == "all" else df[df["gold_type"] == t]
        k, n = int(g[col].sum()), len(g)
        p, lo, hi = wilson(k, n)
        out[t] = {"k": k, "n": n, "acc": p, "lo": lo, "hi": hi}
    return out


def mcnemar(df):
    b = int((df["step1_correct"] & ~df["step2_correct"]).sum())  # broken
    c = int((~df["step1_correct"] & df["step2_correct"]).sum())  # fixed
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"s1_right_s2_wrong": b, "s1_wrong_s2_right": c, "p_exact": p}


def confusion(df, col):
    m = pd.crosstab(pd.Categorical(df["gold_type"], GOLD_TYPES), pd.Categorical(df[col], PRED_TYPES), dropna=False)
    m.index.name, m.columns.name = "gold", "predicted"
    return m


# ---------- word overlap ----------

STOP = set(ENGLISH_STOP_WORDS)


def tokens(text):
    out = set()
    for w in re.split(r"[^0-9a-z]+", str(text).lower()):
        if w in STOP:
            continue
        if re.fullmatch(r"\d{4}", w) or len(re.findall(r"[a-z]", w)) >= 3:
            out.add(w)
    return out


def spec_words(spec):
    parts = []
    for k in SPEC_WORD_FIELDS:
        v = spec.get(k, "")
        parts.extend(v if isinstance(v, list) else [v])
    return tokens(" ".join(map(str, parts)))


def page_token_sets(docs):
    """Per document, a list of token sets per page: text layer, or OCR from Step 2 when the layer is thin."""
    ocr = {}
    inv = OUT / "inventory" / "pages.jsonl"
    if inv.exists():
        for line in inv.open():
            r = json.loads(line)
            if r.get("ocr_text"):
                ocr[(r["doc_id"], r["page"])] = r["ocr_text"]
    sets, cov = {}, {"thin_pages": 0, "thin_with_ocr": 0}
    for d in docs:
        pages = []
        for i, t in enumerate(doc_texts(d)):
            if n_words(t) < MIN_TEXT_WORDS:
                cov["thin_pages"] += 1
                if (d, i + 1) in ocr:
                    cov["thin_with_ocr"] += 1
                    t = t + "\n" + ocr[(d, i + 1)]
            pages.append(tokens(t))
        sets[d] = pages
    return sets, cov


def overlap(df, step):
    docs = sorted(df["doc_id"].unique())
    psets, cov = page_token_sets(docs)
    rows = []
    for _, r in df[df[f"{step}_done"]].iterrows():
        words = spec_words(r[f"{step}_spec"])
        qw = tokens(r["question"])
        pages = psets[r["doc_id"]]
        gi = r["gold_page"] - 1
        rec = {"annotator": r["annotator"], "item_id": r["item_id"], "gold_type": r["gold_type"]}
        for name, w in [("all", words), ("question", words & qw), ("added", words - qw)]:
            rec[f"n_{name}"] = len(w)
            if not w:
                continue
            hits = np.array([len(w & p) / len(w) for p in pages])
            others = np.delete(hits, gi)
            rec[f"gold_{name}"] = hits[gi]
            rec[f"other_{name}"] = others.mean()
            rec[f"top_{name}"] = bool(hits[gi] > others.max())          # strictly highest
            rec[f"top_tie_{name}"] = bool(hits[gi] > 0 and hits[gi] >= others.max())  # highest, ties allowed, at least one hit
        rows.append(rec)
    o = pd.DataFrame(rows)
    summ = {}
    for name in ["all", "question", "added"]:
        if f"gold_{name}" not in o:
            continue
        g = o.dropna(subset=[f"gold_{name}"])
        diff = g[f"gold_{name}"] - g[f"other_{name}"]
        p = wilcoxon(diff).pvalue if len(g) > 0 and (diff != 0).any() else np.nan
        summ[name] = {"n": len(g), "gold_hit": g[f"gold_{name}"].mean(), "other_hit": g[f"other_{name}"].mean(),
                      "wilcoxon_p": p, "share_gold_top": g[f"top_{name}"].mean(),
                      "share_gold_top_ties": g[f"top_tie_{name}"].mean(), "median_words": g[f"n_{name}"].median()}
    cov["gold_pages_thin"] = int(sum(n_words(doc_texts(d)[p - 1]) < MIN_TEXT_WORDS
                                     for d, p in df[["doc_id", "gold_page"]].drop_duplicates().values))
    return summ, cov


# ---------- main ----------

def analyse(df, sample):
    # Majority baseline over the items in the form (build_form.py may use a subset of the sample).
    form_items = OUT / "form_items.txt"
    ids = form_items.read_text().split() if form_items.exists() else list(sample.index)
    real = sample.loc[[i for i in ids if not sample.loc[i, "practice"]]]
    base_type = real["gold_type"].value_counts()
    baseline = base_type.max() / len(real)
    d1 = df[df["step1_done"]].copy()
    d12 = df[df["step1_done"] & df["step2_done"]].copy()
    for c in ["step1_correct", "step2_correct"]:
        d1[c] = d1[c].astype(bool)
        d12[c] = d12[c].astype(bool)

    m = {"n_annotators": df["annotator"].nunique(), "annotators": sorted(df["annotator"].unique()),
         "inventory_source": sorted(df["inventory_source"].unique()),
         "n_rows": len(df), "n_step1": len(d1), "n_step2": len(d12), "n_step3": int(df["step3_done"].sum()),
         "majority_baseline": {"type": base_type.idxmax(), "acc": baseline}}
    m["accuracy_step1"] = acc_table(d1, "step1_correct")
    m["accuracy_step2"] = acc_table(d12, "step2_correct")
    m["per_annotator"] = {a: {"step1": acc_table(g, "step1_correct")["all"],
                              "step2": acc_table(g[g["step2_done"]], "step2_correct")["all"],
                              "mcnemar": mcnemar(g[g["step2_done"]])}
                          for a, g in d1.groupby("annotator")}
    m["mcnemar_pooled"] = mcnemar(d12)
    type_changed = d12["step1_type"] != d12["step2_type"]
    m["changes"] = {"any_change": int(d12["changed"].fillna(False).astype(bool).sum()),
                    "type_changed": int(type_changed.sum()),
                    "render_changed": int((d12["step1_render"] != d12["step2_render"]).sum()),
                    "fixed": m["mcnemar_pooled"]["s1_wrong_s2_right"],
                    "broke": m["mcnemar_pooled"]["s1_right_s2_wrong"]}
    m["confusion_step1"] = confusion(d1, "step1_type").to_dict("index")
    m["confusion_step2"] = confusion(d12, "step2_type").to_dict("index")
    m["render_rate"] = {s: {t: float(g[f"{s}_render"].astype(bool).mean()) for t, g in dd.groupby("gold_type")}
                        for s, dd in [("step1", d1), ("step2", d12)]}
    d3 = df[df["step3_done"]]
    # Subtype match only means something when the final (step 2) type is right.
    ch = d3[(d3["gold_type"] == "chart") & (d3["step2_type"] == "chart")]["subtype_match"].value_counts().to_dict()
    nch = sum(ch.values())
    m["subtype_match_chart"] = {"counts": ch, "n": nch,
                                "yes_rate": ch.get("yes", 0) / nch if nch else np.nan,
                                "yes_or_partly_rate": (ch.get("yes", 0) + ch.get("partly", 0)) / nch if nch else np.nan}
    m["label_not_ok"] = d3[d3["label_ok"] == "no"][["annotator", "item_id", "gold_type", "notes"]].to_dict("records")
    m["label_unsure"] = int((d3["label_ok"] == "unsure").sum())
    m["no_clear_region"] = int(d3["no_clear_region"].fillna(False).astype(bool).sum())

    kap = {}
    for a, b in itertools.combinations(sorted(d1["annotator"].unique()), 2):
        x = d1[d1["annotator"] == a].set_index("item_id")
        y = d1[d1["annotator"] == b].set_index("item_id")
        common = x.index.intersection(y.index)
        if len(common) < 2:
            continue
        kap[f"{a} vs {b}"] = {"n": len(common),
                              "type_step1": cohen_kappa_score(x.loc[common, "step1_type"], y.loc[common, "step1_type"]),
                              "render_step1": cohen_kappa_score(x.loc[common, "step1_render"].astype(str),
                                                                y.loc[common, "step1_render"].astype(str)),
                              "type_agreement": float((x.loc[common, "step1_type"] == y.loc[common, "step1_type"]).mean())}
    m["kappa"] = kap
    m["accuracy_by_confidence"] = {int(c): {"n": len(g), "acc": float(g["step1_correct"].mean())}
                                   for c, g in d1.groupby("confidence")}
    m["median_seconds"] = {s: float(df[f"{s}_t"].median()) for s in STEPS}
    for s in ["step1", "step2"]:
        summ, cov = overlap(df, s)
        m[f"word_overlap_{s}"] = summ
    m["ocr_coverage"] = cov
    return m, d1


def fmt_acc(a):
    return f"{a['acc']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}] ({a['k']}/{a['n']})"


def write_summary(m, path):
    L = ["# Part A summary", "",
         f"Annotators: {', '.join(m['annotators'])}. Inventory source: {', '.join(m['inventory_source'])}.",
         f"Rows (annotator x item): {m['n_rows']}; step 1 done {m['n_step1']}, step 2 {m['n_step2']}, step 3 {m['n_step3']}.",
         f"Majority baseline: {m['majority_baseline']['acc']:.2f} (always '{m['majority_baseline']['type']}').", "",
         "## Type accuracy (pooled over annotators; Wilson 95% intervals)", "",
         "| gold type | step 1 | step 2 |", "|---|---|---|"]
    for t in GOLD_TYPES + ["all"]:
        L.append(f"| {t} | {fmt_acc(m['accuracy_step1'][t])} | {fmt_acc(m['accuracy_step2'][t])} |")
    L += ["", "Pooled intervals treat annotator-item pairs as independent. Per annotator:", ""]
    for a, v in m["per_annotator"].items():
        L.append(f"- {a}: step 1 {fmt_acc(v['step1'])}, step 2 {fmt_acc(v['step2'])}, "
                 f"McNemar exact p = {v['mcnemar']['p_exact']:.3f}")
    c = m["changes"]
    L += ["", "## Changes after the inventory", "",
          f"Any change: {c['any_change']}. Type changed: {c['type_changed']}. Render changed: {c['render_changed']}.",
          f"Fixed an error: {c['fixed']}. Broke a correct answer: {c['broke']}. "
          f"Pooled McNemar exact p = {m['mcnemar_pooled']['p_exact']:.3f}.", ""]
    for s in ["step1", "step2"]:
        L += [f"## Confusion matrix, {s} (rows gold, columns predicted)", "",
              "| gold | " + " | ".join(PRED_TYPES) + " |", "|---" * (len(PRED_TYPES) + 1) + "|"]
        for g in GOLD_TYPES:
            L.append(f"| {g} | " + " | ".join(str(m[f'confusion_{s}'][g][p]) for p in PRED_TYPES) + " |")
        L.append("")
    L += ["## Render rate per gold type", "", "| gold | step 1 | step 2 |", "|---|---|---|"]
    for t in GOLD_TYPES:
        r1, r2 = m["render_rate"]["step1"].get(t, np.nan), m["render_rate"]["step2"].get(t, np.nan)
        L.append(f"| {t} | {r1:.2f} | {r2:.2f} |")
    s = m["subtype_match_chart"]
    L += ["", f"## Subtype match (chart items typed chart at step 2, n = {s['n']})", "",
          f"Counts: {s['counts']}. Yes: {s['yes_rate']:.2f}. Yes or partly: {s['yes_or_partly_rate']:.2f}.", "",
          f"## Labels", "", f"label_ok = no: {len(m['label_not_ok'])}. Unsure: {m['label_unsure']}. "
          f"No clear region: {m['no_clear_region']}.", ""]
    for r in m["label_not_ok"]:
        L.append(f"- {r['annotator']} {r['item_id']} ({r['gold_type']}): {r['notes']}")
    L += ["", "## Agreement (step 1)", ""]
    if not m["kappa"]:
        L.append("One annotator only.")
    for k, v in m["kappa"].items():
        L.append(f"- {k} (n = {v['n']}): kappa type {v['type_step1']:.2f}, kappa render {v['render_step1']:.2f}, "
                 f"raw type agreement {v['type_agreement']:.2f}")
    L += ["", "## Accuracy by confidence (step 1)", "", "| confidence | n | accuracy |", "|---|---|---|"]
    for k, v in m["accuracy_by_confidence"].items():
        L.append(f"| {k} | {v['n']} | {v['acc']:.2f} |")
    t = m["median_seconds"]
    L += ["", f"Median seconds per step: step 1 {t['step1']:.0f}, step 2 {t['step2']:.0f}, step 3 {t['step3']:.0f}.", ""]
    for s in ["step1", "step2"]:
        L += [f"## Word overlap, {s} specification", "",
              "| words | n items | gold hit | other pages hit | Wilcoxon p | gold strictly top | gold top (ties) | median words |",
              "|---|---|---|---|---|---|---|---|"]
        for name, v in m[f"word_overlap_{s}"].items():
            L.append(f"| {name} | {v['n']} | {v['gold_hit']:.2f} | {v['other_hit']:.2f} | {v['wilcoxon_p']:.3g} | "
                     f"{v['share_gold_top']:.2f} | {v['share_gold_top_ties']:.2f} | {v['median_words']:.0f} |")
        L.append("")
    cov = m["ocr_coverage"]
    L.append(f"OCR coverage: {cov['thin_with_ocr']} of {cov['thin_pages']} thin pages (fewer than {MIN_TEXT_WORDS} "
             f"text-layer words) in these documents have OCR text. Thin gold pages: {cov['gold_pages_thin']}.")
    path.write_text("\n".join(L) + "\n")


def figures(m, out):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

    def acc_panel(ax):
        cats = GOLD_TYPES + ["all"]
        x = np.arange(len(cats))
        w = 0.36
        for j, (s, col, lab) in enumerate([("step1", C_STEP1, "Question only"), ("step2", C_STEP2, "With inventory")]):
            a = m[f"accuracy_{s}"]
            y = np.array([a[c]["acc"] for c in cats])
            err = np.array([[a[c]["acc"] - a[c]["lo"] for c in cats], [a[c]["hi"] - a[c]["acc"] for c in cats]])
            ax.bar(x + (j - 0.5) * w, y, w * 0.92, color=col, label=lab, zorder=2)
            ax.errorbar(x + (j - 0.5) * w, y, yerr=err, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
        ax.axhline(m["majority_baseline"]["acc"], color=INK2, ls="--", lw=0.8, zorder=1, label="Majority")
        ax.set_xticks(x, ["Chart", "Table", "Figure", "Text", "All"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Type accuracy")
        ax.yaxis.grid(True, color="#e6e6e3", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", ncols=3, bbox_to_anchor=(0, 1.14), handlelength=1.5, columnspacing=1.0)

    def conf_panel(ax):
        from matplotlib.colors import LinearSegmentedColormap
        cm = pd.DataFrame(m["confusion_step1"]).T.loc[GOLD_TYPES, PRED_TYPES].values
        cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1), aspect="auto")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                dark = cm[i, j] > 0.55 * cm.max()
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if dark else INK)
        ax.set_xticks(range(len(PRED_TYPES)), ["Chart", "Table", "Figure", "Text", "Layout"])
        ax.set_yticks(range(len(GOLD_TYPES)), ["Chart", "Table", "Figure", "Text"])
        ax.set_xlabel("Predicted at step 1")
        ax.set_ylabel("Gold type")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig, ax = plt.subplots(figsize=(3.25, 2.1), layout="constrained")
    acc_panel(ax)
    fig.savefig(out / "fig_partA_accuracy.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.25, 2.1), layout="constrained")
    conf_panel(ax)
    fig.savefig(out / "fig_partA_confusion.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.1), layout="constrained")
    acc_panel(axs[0])
    conf_panel(axs[1])
    for ax, lab in zip(axs, "AB"):
        ax.text(-0.14, 1.08, lab, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
    fig.savefig(out / "fig_partA.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df, sample, synthetic = load(args.files, args.allow_synthetic)
    out = Path(args.out) if args.out else (IDEA4 / "tests" / "out_SYNTHETIC" if synthetic else OUT / "partA")
    if synthetic and out.resolve() == (OUT / "partA").resolve():
        sys.exit("synthetic results may not be written to out/partA/")
    out.mkdir(parents=True, exist_ok=True)

    m, d1 = analyse(df, sample)
    m["synthetic"] = synthetic
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    write_summary(m, out / "summary.md")
    long = df.drop(columns=["step1_spec", "step2_spec"]).assign(
        step1_spec=df["step1_spec"].map(json.dumps), step2_spec=df["step2_spec"].map(json.dumps))
    long.to_csv(out / "items_long.csv", index=False)
    d1[["annotator", "item_id", "gold_type", "step1_type", "step1_correct", "confidence", "cues", "question"]] \
        .to_csv(out / "cues.csv", index=False)
    figures(m, out)
    print((out / "summary.md").read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
