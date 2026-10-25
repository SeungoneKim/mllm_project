"""Part A (CPU): is the evidence of each question in the text layer of its gold pages at all?

  python coverage.py     # writes out/coverage/{questions.csv, summary.md, fig_coverage.pdf}

For every eligible question (answerable, evidence pages inside the PDF), over its gold pages:
  text layer     words in the PDF text layer; a page with fewer than 20 has no usable text layer
  raster         share of the page area covered by embedded raster images (photos, screenshots, raster charts)
  vector         number of vector drawing paths (vector charts, table rules, diagrams)
  answer in text the reference answer can be found in the gold pages' text layer: every number of the answer
                 for answers with numbers, otherwise at least 80% of its content words. A proxy: counts and derived
                 values may be missing from the text even when every input to them is there.
"""
import json
import re

import numpy as np
import pandas as pd
import pymupdf

from common import DOC_DIR, MIN_TEXT_WORDS, OUT, SEED, TYPES, eligible, load_qa, n_words, text_layer

COV = OUT / "coverage"
RASTER_HEAVY = 0.2  # a page whose raster images cover at least 20% of its area
STOP = set("""the and for with from that this are was were has have had its their they them than then into over
under about after before between each any all both some such other only also not can could would will may might
what which who when where why how many much more most less least very per""".split())


def numbers(text):
    """Numbers in a string, commas removed, trailing .0 dropped (so 1,358,000 and 1358000 match)."""
    out = set()
    for m in re.findall(r"\d+(?:\.\d+)?", (text or "").replace(",", "")):
        out.add(m[:-2] if m.endswith(".0") else m)
    return out


def words(text):
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", (text or "").lower()) if w not in STOP}


def answer_in_text(answer, text):
    a_num = numbers(answer)
    if a_num:
        return float(a_num <= numbers(text))
    a_w = words(answer)
    if not a_w:
        return float("nan")
    return float(len(a_w & words(text)) / len(a_w) >= 0.8)


def page_stats(page):
    area = abs(page.rect)
    raster = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        raster += abs(r) if not r.is_empty else 0.0
    text = text_layer(page)
    return {"words": n_words(text), "raster_frac": min(1.0, raster / area) if area else 0.0,
            "vector_paths": len(page.get_cdrawings()), "text": text}


def build():
    qa = eligible(load_qa())
    rows = []
    for d, qs in qa.groupby("doc_id"):
        with pymupdf.open(DOC_DIR / d) as doc:
            needed = sorted({p for ps in qs["ev_pages"] for p in ps})
            stats = {p: page_stats(doc[p - 1]) for p in needed}
        for _, q in qs.iterrows():
            ps = [stats[p] for p in sorted(set(q["ev_pages"]))]
            text = "\n".join(s["text"] for s in ps)
            rows.append({
                "qid": q["qid"], "doc_id": d, "doc_type": q["doc_type"], "n_gold": len(ps),
                "ev_types": ",".join(q["ev_types"]), "single_type": len(q["ev_types"]) == 1,
                "answer_format": q["answer_format"], "question": q["question"], "answer": q["answer"],
                "min_words": min(s["words"] for s in ps), "no_text_layer": any(s["words"] < MIN_TEXT_WORDS for s in ps),
                "max_raster_frac": max(s["raster_frac"] for s in ps),
                "raster_heavy": any(s["raster_frac"] >= RASTER_HEAVY for s in ps),
                "vector_paths": sum(s["vector_paths"] for s in ps),
                "answer_in_text": answer_in_text(q["answer"], text),
                "counting": bool(re.match(r"\s*how many", q["question"], re.I)),
            })
    return pd.DataFrame(rows).sort_values("qid").reset_index(drop=True)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(SEED)
    b = x[rng.integers(len(x), size=(n, len(x)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def by_type(df):
    """One row per evidence type (questions with that single type), plus all questions."""
    groups = [(t, df[df["single_type"] & (df["ev_types"] == t)]) for t in TYPES] + [("all", df)]
    rows = []
    for t, g in groups:
        ait = g.loc[~g["counting"], "answer_in_text"]
        rows.append({"type": t, "n": len(g), "no_text_layer": g["no_text_layer"].mean(),
                     "raster_heavy": g["raster_heavy"].mean(), "answer_in_text": ait.mean(),
                     "answer_in_text_ci": boot_ci(ait), "n_answer_in_text": int(ait.notna().sum()),
                     "counting": g["counting"].mean()})
    return pd.DataFrame(rows)


def figure(tab):
    import matplotlib.pyplot as plt
    C1, C2, C3, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e6e5e0"
    tab = tab.set_index("type")
    order = TYPES + ["all"]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    bars = [("answer_in_text", C1, "answer found in text layer"), ("raster_heavy", C2, "gold page is >=20% raster"),
            ("no_text_layer", C3, "gold page has no text layer")]
    for k, (col, c, lab) in enumerate(bars):
        ax.barh(y + (1 - k) * 0.27, tab.loc[order, col], height=0.25, color=c, label=lab)
    ax.set_yticks(y, [f"{t} ({int(tab.loc[t, 'n'])})" for t in order], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"], fontsize=7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=1)
    fig.tight_layout()
    fig.savefig(COV / "fig_coverage.pdf")
    plt.close(fig)


def main():
    COV.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(COV / "questions.csv", index=False)
    tab = by_type(df)
    figure(tab)
    L = ["# Part A: is the evidence in the text layer?", "",
         f"{len(df)} eligible questions over {df['doc_id'].nunique()} documents. Evidence-type rows hold questions "
         f"with that single type; 'all' holds every question. 'answer in text' excludes counting questions "
         f"('How many ...'), whose answer is rarely written on the page.", "",
         "| type | n | answer in text [95% CI] | raster-heavy gold page | no text layer | counting |", "|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        L.append(f"| {r.type} | {r.n} | {r.answer_in_text:.0%} [{r.answer_in_text_ci[0]:.0%}, {r.answer_in_text_ci[1]:.0%}] "
                 f"(n={r.n_answer_in_text}) | {r.raster_heavy:.0%} | {r.no_text_layer:.0%} | {r.counting:.0%} |")
    (COV / "summary.md").write_text("\n".join(L) + "\n")
    (COV / "metrics.json").write_text(json.dumps(tab.to_dict(orient="records"), indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
