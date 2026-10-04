"""Part B analysis: gold-page rank within the document for every query variant.

  python analyze_B.py    # reads out/partB/scores.jsonl, writes out/partB/{metrics.json, summary.md, ranks.csv, figures}

Fusion: z(question) + w * z(image), with z-scores over the pages of the document. w = 1 is the main result;
other w only as a curve (no tuning set). Paired comparisons against the question-as-text baseline use a
seeded bootstrap of the mean change in reciprocal rank and a Wilcoxon signed-rank test on ranks.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import IDEA4, OUT, SEED

PDIR = IDEA4 / "proposer"
OUTB = OUT / "partB"
TYPES = ["chart", "table", "text", "figure"]
LABEL = {"q_text": "Question (text)", "spec_text": "Spec as text", "code_img": "Spec drawn (code)",
         "gen_img": "Spec generated (Qwen-Image)", "fuse_code": "Question + code drawing",
         "fuse_gen": "Question + generated image", "fuse_spec_text": "Question + spec as text",
         "qimg": "Question as image", "oracle": "Oracle crop (real region)",
         "spec_text_h": "Human spec as text", "code_img_h": "Human spec drawn (code)",
         "gen_img_h": "Human spec generated", "fuse_code_h": "Question + human code drawing",
         "fuse_gen_h": "Question + human generated image"}
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"


def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def rank_of(scores, gold):
    s = np.asarray(scores)
    return int(1 + (s > s[gold]).sum())


def load():
    ids = pd.read_csv(PDIR / "id_map.csv").set_index("pid")
    specs = {json.loads(l)["pid"]: json.loads(l) for l in (PDIR / "specs.jsonl").open()}
    sc = {}
    for line in (OUTB / "scores.jsonl").open():
        r = json.loads(line)
        sc[(r["pid"], r["variant"])] = np.array(r["scores"])
    return ids, specs, sc


def fused(sc, pid, img_variant, w=1.0):
    a, b = sc.get((pid, "q_text")), sc.get((pid, img_variant))
    if a is None or b is None:
        return None
    return z(a) + w * z(b)


def build_ranks(ids, specs, sc):
    rows = []
    form = set((OUT / "form_items.txt").read_text().split()) if (OUT / "form_items.txt").exists() else set()
    for pid, r in ids.iterrows():
        gold = int(r["gold_page"]) - 1
        base = {"pid": pid, "item_id": r["item_id"] if isinstance(r["item_id"], str) else "",
                "gold_type": r["gold_type"], "n_pages": len(sc[(pid, "q_text")]),
                "prop_type": specs[pid]["type"], "prop_render": specs[pid]["render"]}
        base["part_a"] = base["item_id"] in form and base["item_id"].startswith("A")
        variants = {v: sc[(pid, v)] for (p, v) in sc if p == pid}
        for name, src in [("fuse_code", "code_img"), ("fuse_gen", "gen_img"), ("fuse_spec_text", "spec_text"),
                          ("fuse_code_h", "code_img_h"), ("fuse_gen_h", "gen_img_h")]:
            f = fused(sc, pid, src)
            if f is not None:
                variants[name] = f
        for v, s in variants.items():
            rows.append({**base, "variant": v, "rank": rank_of(s, gold)})
    d = pd.DataFrame(rows)
    d["rr"] = 1 / d["rank"]
    d["r1"] = d["rank"] == 1
    d["r5"] = d["rank"] <= 5
    return d


def table(d, variants, subset_name):
    out = []
    for v in variants:
        g = d[d["variant"] == v]
        if g.empty:
            continue
        for t in TYPES + ["all"]:
            h = g if t == "all" else g[g["gold_type"] == t]
            if h.empty:
                continue
            out.append({"subset": subset_name, "variant": v, "type": t, "n": len(h), "R@1": h["r1"].mean(),
                        "R@5": h["r5"].mean(), "MRR": h["rr"].mean(), "median_rank": h["rank"].median()})
    return pd.DataFrame(out)


def paired(d, v, base="q_text", n_boot=2000):
    a = d[d["variant"] == base].set_index("pid")
    b = d[d["variant"] == v].set_index("pid")
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return None
    diff = (b.loc[common, "rr"] - a.loc[common, "rr"]).values
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(n_boot)]
    rk = (b.loc[common, "rank"] - a.loc[common, "rank"]).values
    p = wilcoxon(rk).pvalue if (rk != 0).any() else 1.0
    return {"n": len(common), "d_mrr": float(diff.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "better": int((rk < 0).sum()), "worse": int((rk > 0).sum()),
            "same": int((rk == 0).sum()), "wilcoxon_p": float(p)}


def w_curve(ids, sc, src, pids):
    ws = np.round(np.arange(0, 2.01, 0.25), 2)
    res = []
    for w in ws:
        rr = []
        for pid in pids:
            f = fused(sc, pid, src, w)
            if f is not None:
                rr.append(1 / rank_of(f, int(ids.loc[pid, "gold_page"]) - 1))
        res.append(float(np.mean(rr)) if rr else np.nan)
    return dict(zip(map(float, ws), res))


def figure(d, path):
    """Change in MRR against the question-as-text baseline, all 144 questions (bootstrap 95% CI),
    with per-type means as small markers."""
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})
    order = ["spec_text", "code_img", "gen_img", "qimg", "fuse_spec_text", "fuse_code", "fuse_gen"]
    short = {"spec_text": "Spec as text", "code_img": "Code drawing", "gen_img": "Generated image",
             "qimg": "Question as image", "fuse_spec_text": "Q + spec as text", "fuse_code": "Q + code drawing",
             "fuse_gen": "Q + generated image"}
    fig, ax = plt.subplots(figsize=(3.25, 2.3), layout="constrained")
    y = np.arange(len(order))
    for i, v in enumerate(order):
        p = paired(d, v)
        ax.errorbar(p["d_mrr"], i, xerr=[[p["d_mrr"] - p["lo"]], [p["hi"] - p["d_mrr"]]], fmt="o", color=INK,
                    ms=4, elinewidth=1, capsize=2, zorder=3, label="All (95% CI)" if i == 0 else None)
        for t, c, mk in [("chart", C1, "^"), ("table", C2, "s"), ("text", C3, "D")]:
            g = d[d["gold_type"] == t]
            pt = paired(g, v)
            ax.plot(pt["d_mrr"], i + 0.28, mk, color=c, ms=3.5, zorder=2, label=t.capitalize() if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8, ls="--")
    ax.axhline(3.5, color="#d0d0cc", lw=0.6)
    ax.set_yticks(y, [short[v] for v in order])
    ax.invert_yaxis()
    ax.set_xlabel("Change in MRR vs. question as text")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=4, loc="lower left", bbox_to_anchor=(-0.6, 1.0), handletextpad=0.3,
              columnspacing=0.8)
    fig.savefig(path)
    plt.close(fig)


def gallery(d, ids, path, pids):
    """Question, code drawing, generated image and gold page for a few items."""
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import textwrap
    from PIL import Image
    from common import page_image_path
    fig, axs = plt.subplots(len(pids), 4, figsize=(6.75, 1.75 * len(pids)), layout="constrained")
    for i, pid in enumerate(pids):
        r = ids.loc[pid]
        cells = [None, IDEA4 / "renders" / "code" / f"{pid}.png", IDEA4 / "renders" / "gen" / f"{pid}.png",
                 page_image_path(r["doc_id"], int(r["gold_page"]))]
        for j, c in enumerate(cells):
            ax = axs[i, j]
            ax.axis("off")
            if c is None:
                ranks = d[d["pid"] == pid].set_index("variant")["rank"]
                txt = textwrap.fill(r["question"], 34) + f"\n\n{r['gold_type']} | ranks: text {ranks.get('q_text')}, " \
                      f"code {ranks.get('code_img')}, gen {ranks.get('gen_img')}, fused {ranks.get('fuse_gen')}"
                ax.text(0, 1, txt, va="top", fontsize=6.5, transform=ax.transAxes)
            else:
                ax.imshow(Image.open(c).convert("RGB"))
            if i == 0:
                ax.set_title(["Question", "Code drawing", "Qwen-Image", "Gold page"][j], fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main():
    ids, specs, sc = load()
    d = build_ranks(ids, specs, sc)
    d.to_csv(OUTB / "ranks.csv", index=False)
    main_v = ["q_text", "spec_text", "code_img", "gen_img", "fuse_spec_text", "fuse_code", "fuse_gen", "qimg"]
    human_v = ["q_text", "spec_text", "code_img", "gen_img", "fuse_code", "fuse_gen",
               "spec_text_h", "code_img_h", "gen_img_h", "fuse_code_h", "fuse_gen_h", "oracle"]
    subsets = {"all pool questions": d, "proposer drew (render = true)": d[d["prop_render"]],
               "proposer type correct": d[d["prop_type"] == d["gold_type"]], "Part A items": d[d["part_a"]]}
    tabs = [table(g, human_v if k == "Part A items" else main_v, k) for k, g in subsets.items()]
    T = pd.concat(tabs)
    m = {"n_questions": int(d["pid"].nunique()), "median_pages": float(d.drop_duplicates("pid")["n_pages"].median()),
         "random_mrr": float(np.mean([np.mean(1 / np.arange(1, n + 1)) for n in d.drop_duplicates("pid")["n_pages"]])),
         "table": T.to_dict("records"), "paired": {}, "w_curve": {}}
    for k, g in subsets.items():
        m["paired"][k] = {v: paired(g, v) for v in (human_v if k == "Part A items" else main_v) if v != "q_text"}
    m["head_to_head"] = {f"{a} vs {b}": paired(d, a, base=b) for a, b in
                         [("fuse_code", "fuse_spec_text"), ("fuse_gen", "fuse_spec_text"), ("gen_img", "code_img"),
                          ("code_img", "spec_text"), ("fuse_gen", "fuse_code")]}
    allp = list(d["pid"].unique())
    for src in ["code_img", "gen_img"]:
        m["w_curve"][src] = w_curve(ids, sc, src, allp)
    (OUTB / "metrics.json").write_text(json.dumps(m, indent=1, default=float))

    L = ["# Part B summary", "",
         f"{m['n_questions']} questions, retrieval within the document (median {m['median_pages']:.0f} pages). "
         f"Random-ranking MRR: {m['random_mrr']:.3f}.", "Retriever: ColQwen2.5 (vidore/colqwen2.5-v0.2). "
         "Proposer: Claude Opus 5.5 (fresh context, question + inventory only). Generator: Qwen-Image-2.1.", ""]
    for k in subsets:
        t = T[T["subset"] == k]
        L += [f"## {k}", "", "| variant | type | n | R@1 | R@5 | MRR | median rank |", "|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {LABEL.get(r['variant'], r['variant'])} | {r['type']} | {r['n']} | {r['R@1']:.2f} | "
                     f"{r['R@5']:.2f} | {r['MRR']:.3f} | {r['median_rank']:.0f} |")
        L += ["", "Paired against the question-as-text baseline (all types):", "",
              "| variant | n | change in MRR [95% CI] | better / same / worse | Wilcoxon p |", "|---|---|---|---|---|"]
        for v, p in m["paired"][k].items():
            if p:
                L.append(f"| {LABEL.get(v, v)} | {p['n']} | {p['d_mrr']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] | "
                         f"{p['better']} / {p['same']} / {p['worse']} | {p['wilcoxon_p']:.3g} |")
        L.append("")
    L += ["## Head to head (all questions; change in MRR of the first minus the second)", "",
          "| comparison | n | change in MRR [95% CI] | better / same / worse | Wilcoxon p |", "|---|---|---|---|---|"]
    for k, p in m["head_to_head"].items():
        a, b = k.split(" vs ")
        L.append(f"| {LABEL[a]} vs {LABEL[b]} | {p['n']} | {p['d_mrr']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] | "
                 f"{p['better']} / {p['same']} / {p['worse']} | {p['wilcoxon_p']:.3g} |")
    L += ["", "## Fusion weight curve (all questions, MRR)", ""]
    for src, c in m["w_curve"].items():
        L.append(f"- {LABEL[src]}: " + ", ".join(f"w={w:g}: {v:.3f}" for w, v in c.items()))
    (OUTB / "summary.md").write_text("\n".join(L) + "\n")

    figure(d, OUTB / "fig_partB_delta.pdf")
    # Gallery: the chart and table items where adding the code drawing moved the gold page up the most, plus the
    # item where it moved it down the most. Chosen by rank change only.
    w = d.pivot_table(index=["pid", "gold_type"], columns="variant", values="rank").reset_index()
    w["gain"] = w["q_text"] - w["fuse_code"]
    ct = w[w["gold_type"].isin(["chart", "table"])]
    ex = ct.sort_values("gain", ascending=False)["pid"].tolist()[:2] + ct.sort_values("gain")["pid"].tolist()[:1]
    gallery(d, ids, OUTB / "fig_partB_gallery.pdf", ex)
    print("\n".join(L))


if __name__ == "__main__":
    main()
