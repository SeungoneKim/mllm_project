"""Step 2: why does page-image retrieval miss? Attribution of every ColQwen miss (CPU).

  python attribute.py     # reads out/scan/, writes out/attr/{questions.csv, metrics.json, summary.md, fig_*.pdf}

A question is a miss when no gold page is in the top 5 (common.TOP_K). For each question, g is its best-ranked gold
page and w the top-ranked non-gold page (the "winner"). Measures:

  look-alike    w is among the 3 pages most similar to g (page-to-page ColQwen similarity, non-gold pages only).
                Chance level: 3 / (number of non-gold pages).
  nn_z          similarity of g to its most similar other page, z-scored within the document (how close g's
                nearest look-alike is, relative to how alike the document's pages are in general).
  page ref      the question cites a page or slide ("page 9", "the second page"); label_mismatch: none of the
                cited numbers is a gold page index (printed page labels, or wrong labels).
  vocabulary    share of the question's content words found in the text layer of g (and of w). A vocabulary gap
                is coverage below 0.5 on a page with a text layer (>= 20 words).

Single cause per miss, first match in this order: page ref, look-alike, no text layer, vocabulary gap, other
(no text layer comes before the vocabulary gap because coverage needs a text layer).
Multi-label rates are reported too, so the order hides nothing.

Pre-registered hypotheses (README.md):
  H1  At least 25% of misses are look-alike misses, and the rate is above chance.
  H2  With document length and evidence type held fixed, a closer nearest look-alike (nn_z) raises the odds of a miss.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import (IDEA4, MIN_TEXT_WORDS, OUT, SCAN, SEED, TOP_K, TYPES, doc_stem, doc_texts, eligible,
                    load_qa, n_words, page_image_path)
from cues import content_words, coverage, element_refs, page_refs

ATTR = OUT / "attr"
N_NEIGHBOURS = 3
VOCAB_GAP = 0.5
CAUSES = ["page ref", "look-alike", "no text layer", "vocabulary gap", "other"]  # also the matching order
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3 (same as idea4 figures)
GREY, INK, INK2, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"


# ---------- per-question measures ----------

def load_scores():
    rows = [json.loads(l) for l in (SCAN / "scores.jsonl").open()]
    return {r["qid"]: np.asarray(r["scores"], float) for r in rows}


def rank_of(s, page):
    """1-indexed rank of a 1-indexed page; ties count in the page's favour (same as idea4)."""
    return int(1 + (s > s[page - 1]).sum())


def measure(q, s, sim, texts):
    n = len(s)
    gold = sorted(set(q["ev_pages"]))
    ranks = {p: rank_of(s, p) for p in gold}
    g = min(gold, key=lambda p: (ranks[p], p))
    best = ranks[g]
    order = np.argsort(-s, kind="stable") + 1
    non_gold = np.array([p for p in range(1, n + 1) if p not in gold])
    if len(non_gold) == 0:
        return None  # every page is gold: nothing can beat it
    w = int(next(p for p in order if p not in gold))

    # look-alike: neighbours of g among the non-gold pages, by page-to-page similarity
    sg = sim[g - 1]
    nbr = non_gold[np.argsort(-sg[non_gold - 1], kind="stable")]
    w_nn_rank = int(np.where(nbr == w)[0][0]) + 1
    off = sim[~np.eye(n, dtype=bool)]
    others = np.delete(sg, g - 1)
    nn1 = float(others.max()) if n > 1 else float("nan")
    nn_z = (nn1 - off.mean()) / (off.std() + 1e-9) if n > 2 else float("nan")

    refs = page_refs(q["question"], n)
    qw = content_words(q["question"])
    g_words = n_words(texts[g - 1])
    return {
        "best_rank": best, "best_gold": g, "rr": 1 / best, "r1": best == 1, "r5": best <= TOP_K,
        "all_gold_top5": all(r <= TOP_K for r in ranks.values()), "miss": best > TOP_K,
        "winner": w, "winner_nn_rank": w_nn_rank, "n_non_gold": len(non_gold),
        "lookalike": w_nn_rank <= N_NEIGHBOURS,
        "lookalike_chance": min(N_NEIGHBOURS, len(non_gold)) / max(len(non_gold), 1),
        "sim_gw": float(sg[w - 1]), "nn1_sim": nn1, "nn_z": float(nn_z),
        "neighbours": json.dumps([int(p) for p in nbr[:N_NEIGHBOURS]]),
        "cited_pages": json.dumps(refs), "cites_page": bool(refs),
        "label_mismatch": bool(refs) and not (set(refs) & set(gold)),
        "cites_element": bool(element_refs(q["question"])),
        "q_words": len(set(qw)), "gold_text_words": g_words, "no_text_layer": g_words < MIN_TEXT_WORDS,
        "cov_gold": coverage(qw, texts[g - 1]), "cov_winner": coverage(qw, texts[w - 1]),
    }


def cause_of(r):
    if not r["miss"]:
        return ""
    if r["cites_page"]:
        return "page ref"
    if r["lookalike"]:
        return "look-alike"
    if r["no_text_layer"]:
        return "no text layer"
    if r["cov_gold"] < VOCAB_GAP:
        return "vocabulary gap"
    return "other"


def build():
    qa = eligible(load_qa())
    scores = load_scores()
    qa = qa[qa["qid"].isin(scores)]
    rows = []
    for d, qs in qa.groupby("doc_id"):
        sim = np.load(SCAN / "pagesim" / f"{doc_stem(d)}.npy")
        texts = doc_texts(d)
        for _, q in qs.iterrows():
            s = scores[q["qid"]]
            assert len(s) == len(texts) == sim.shape[0], (d, len(s), len(texts), sim.shape)
            r = measure(q, s, sim, texts)
            if r is None:
                continue
            rows.append({"qid": q["qid"], "doc_id": d, "doc_type": q["doc_type"], "n_pages": len(s),
                         "n_gold": len(set(q["ev_pages"])), "ev_types": ",".join(q["ev_types"]),
                         "question": q["question"], **r})
    df = pd.DataFrame(rows)
    df["single_page"] = df["n_gold"] == 1
    df["vocab_gap"] = ~df["no_text_layer"] & (df["cov_gold"] < VOCAB_GAP)
    df["cause"] = df.apply(cause_of, axis=1)
    return df.sort_values("qid").reset_index(drop=True)


# ---------- statistics ----------

def boot_ci(x, stat=np.mean, n=5000, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    b = [stat(x[rng.integers(len(x), size=len(x))]) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def chance_p(observed, probs, n=20000, seed=SEED):
    """One-sided p: P(sum of independent Bernoulli(p_i) >= observed), by simulation."""
    rng = np.random.default_rng(seed)
    draws = (rng.random((n, len(probs))) < np.asarray(probs)).sum(axis=1)
    return float((draws >= observed).mean())


def logit(X, y, names, iters=100, ridge=1e-6):
    """Logistic regression by Newton-Raphson. Returns coefficients, SEs, odds ratios and Wald p-values."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    out = []
    np.seterr(over="ignore")  # separated terms give odds ratios of inf; they are reported as such
    for name, bi, si in zip(["intercept"] + names, b, se):
        out.append({"term": name, "coef": float(bi), "se": float(si), "odds_ratio": float(np.exp(bi)),
                    "or_ci": [float(np.exp(bi - 1.96 * si)), float(np.exp(bi + 1.96 * si))],
                    "p": float(2 * norm.sf(abs(bi / si)))})
    return out


def h2_model(df):
    d = df.dropna(subset=["nn_z"]).copy()
    cols = {"nn_z": d["nn_z"], "log_pages": np.log(d["n_pages"]), "multi_page": (~d["single_page"]).astype(float)}
    for t in TYPES:
        if t != "text":  # text is the reference level; questions can have several evidence types
            cols[f"type_{t}"] = d["ev_types"].str.split(",").map(lambda ts: float(t in ts))
    X = pd.DataFrame(cols)
    X = X.loc[:, X.std() > 0]
    return logit(X.values, d["miss"].values, list(X.columns)), len(d)


def andre_check(df):
    """Our question-as-text ranks against André's (idea4 Part B), on his 144 pool questions."""
    ranks, ids = IDEA4 / "out" / "partB" / "ranks.csv", IDEA4 / "proposer" / "id_map.csv"
    if not ranks.exists() or not ids.exists():
        return None
    a = pd.read_csv(ranks)
    a = a[a["variant"] == "q_text"][["pid", "rank"]].merge(pd.read_csv(ids)[["pid", "qid", "gold_page"]], on="pid")
    m = a.merge(df[["qid", "best_rank", "best_gold"]], on="qid")
    m = m[m["gold_page"] == m["best_gold"]]
    return {"n": int(len(m)), "same_rank": int((m["rank"] == m["best_rank"]).sum()),
            "max_abs_diff": int((m["rank"] - m["best_rank"]).abs().max()) if len(m) else None}


def metrics(df):
    miss = df[df["miss"]]
    hit = df[~df["miss"]]
    m = {"n_questions": int(len(df)), "n_docs": int(df["doc_id"].nunique()), "top_k": TOP_K,
         "R@1": float(df["r1"].mean()), "R@5": float(df["r5"].mean()), "MRR": float(df["rr"].mean()),
         "n_miss": int(len(miss)), "miss_rate": float(df["miss"].mean())}
    m["by_doc_type"] = {t: {"n": int(len(g)), "R@5": float(g["r5"].mean()), "misses": int(g["miss"].sum())}
                        for t, g in df.groupby("doc_type")}
    m["by_pages"] = {k: {"n": int(len(g)), "R@5": float(g["r5"].mean()), "misses": int(g["miss"].sum())}
                     for k, g in df.groupby(pd.cut(df["n_pages"], [0, 20, 50, 100, 10000],
                                                   labels=["<=20", "21-50", "51-100", ">100"]), observed=True)}
    m["single_vs_multi"] = {("single" if k else "multi"): {"n": int(len(g)), "R@5": float(g["r5"].mean())}
                            for k, g in df.groupby("single_page")}
    m["cause_counts"] = {c: int((miss["cause"] == c).sum()) for c in CAUSES}
    flags = {"look-alike": "lookalike", "page ref": "cites_page", "label mismatch": "label_mismatch",
             "vocabulary gap": "vocab_gap", "no text layer": "no_text_layer", "cites element": "cites_element"}
    m["flag_rates"] = {k: {"misses": float(miss[v].mean()), "hits": float(hit[v].mean()),
                           "n_misses": int(miss[v].sum())} for k, v in flags.items()}
    obs = int(miss["lookalike"].sum())
    m["H1"] = {"lookalike_misses": obs, "share": float(miss["lookalike"].mean()) if len(miss) else float("nan"),
               "share_ci": boot_ci(miss["lookalike"].astype(float)),
               "chance_share": float(miss["lookalike_chance"].mean()) if len(miss) else float("nan"),
               "p_vs_chance": chance_p(obs, miss["lookalike_chance"]) if len(miss) else float("nan"),
               "threshold": 0.25}
    h = m["H1"]
    above = len(miss) > 0 and h["p_vs_chance"] < 0.05
    h["verdict"] = ("supported" if above and h["share_ci"][0] >= h["threshold"] else
                    "consistent (point estimate meets the threshold, CI does not)" if above and h["share"] >= h["threshold"]
                    else "not supported")
    m["H2_logit"], m["H2_n"] = h2_model(df)
    m["andre_check"] = andre_check(df)
    return m


# ---------- figures ----------

def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_causes(df, m):
    import matplotlib.pyplot as plt
    miss = df[df["miss"]]
    counts = [(miss["cause"] == c).sum() for c in CAUSES]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    y = np.arange(len(CAUSES))[::-1]
    ax.barh(y, counts, height=0.6, color=C1)
    for yi, c in zip(y, counts):
        ax.text(c + max(counts) * 0.02, yi, f"{c} ({c / max(len(miss), 1):.0%})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, CAUSES)
    ax.set_xlim(0, max(counts) * 1.3 + 1)
    ax.set_xlabel(f"misses (no gold page in top {TOP_K}), n = {len(miss)}", fontsize=8, color=INK2)
    ch = miss.loc[miss["cause"] != "page ref", "lookalike_chance"].sum()  # expected look-alike count by chance
    ax.plot([ch, ch], [y[1] - 0.4, y[1] + 0.4], color=INK, linewidth=1.5, linestyle=(0, (2, 1.5)))
    ax.text(ch, y[1] - 0.45, "expected by chance", fontsize=6.5, color=INK2, ha="left", va="top")
    style(ax)
    fig.tight_layout()
    fig.savefig(ATTR / "fig_causes.pdf")
    plt.close(fig)


def fig_winner(df):
    """Where the winning page sits among g's neighbours, for misses, against chance."""
    import matplotlib.pyplot as plt
    miss = df[df["miss"]]
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 10 ** 6, ">10")]
    obs, exp = [], []
    for lo, hi, _ in bins:
        obs.append(((miss["winner_nn_rank"] >= lo) & (miss["winner_nn_rank"] <= hi)).mean())
        exp.append(np.mean([max(0, min(hi, n) - lo + 1) / n for n in miss["n_non_gold"]]))
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.bar(x - 0.2, obs, width=0.38, color=C1, label="observed")
    ax.bar(x + 0.2, exp, width=0.38, color=GREY, label="chance")
    ax.set_xticks(x, [b[2] for b in bins])
    ax.set_xlabel("winner's rank among the gold page's most similar pages", fontsize=8, color=INK2)
    ax.set_ylabel("share of misses", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ATTR / "fig_winner_rank.pdf")
    plt.close(fig)


def fig_nn(df):
    """Miss rate by quartile of nn_z, split by document length."""
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["nn_z"]).copy()
    d["q"] = pd.qcut(d["nn_z"], 4, labels=["Q1\n(least alike)", "Q2", "Q3", "Q4\n(most alike)"])
    med = int(d["n_pages"].median())
    groups = [(f"<= {med} pages", d["n_pages"] <= med, C1), (f"> {med} pages", d["n_pages"] > med, C2)]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for k, (lab, mask, col) in enumerate(groups):
        g = d[mask].groupby("q", observed=False)["miss"]
        rate, n = g.mean().values, g.size().values
        xs = x + (k - 0.5) * 0.38
        ax.bar(xs, rate, width=0.36, color=col, label=lab)
        for xi, r, ni in zip(xs, rate, n):
            ax.text(xi, (0 if np.isnan(r) else r) + 0.005, f"n={ni}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x, ["Q1\n(least alike)", "Q2", "Q3", "Q4\n(most alike)"])
    ax.set_xlabel("closeness of the gold page's nearest look-alike (nn_z quartile)", fontsize=8, color=INK2)
    ax.set_ylabel(f"miss rate (no gold in top {TOP_K})", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(ATTR / "fig_lookalike_missrate.pdf")
    plt.close(fig)


def fig_gallery(df, per_cause=1):
    """Gold page next to the page that beat it, for the worst miss of each cause."""
    import textwrap

    import matplotlib.pyplot as plt
    from PIL import Image
    miss = df[df["miss"]].sort_values("best_rank", ascending=False)
    picks = pd.concat([miss[miss["cause"] == c].head(per_cause) for c in ["look-alike", "page ref", "vocabulary gap"]])
    picks = picks[[page_image_path(r.doc_id, r.best_gold).exists() and page_image_path(r.doc_id, r.winner).exists()
                   for r in picks.itertuples()]]
    if picks.empty:
        return
    fig = plt.figure(figsize=(6.0, 4.3 * len(picks)), layout="constrained")
    for sub, r in zip(np.atleast_1d(fig.subfigures(len(picks), 1)), picks.itertuples()):
        q = textwrap.fill(f"[{r.cause}] {r.question}", 80)
        sub.suptitle(q, fontsize=8, color=INK, ha="left", x=0.02)
        for ax, p, lab in zip(sub.subplots(1, 2), [r.best_gold, r.winner],
                              [f"gold page {r.best_gold} (rank {r.best_rank})", f"ColQwen top 1: page {r.winner}"]):
            ax.imshow(Image.open(page_image_path(r.doc_id, p)))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GREY)
            ax.set_title(f"{lab} of {r.n_pages}", fontsize=7.5, color=INK2)
    fig.savefig(ATTR / "fig_examples.pdf")
    plt.close(fig)


# ---------- summary ----------

def summary(df, m):
    L = ["# Idea 5, Step 2: why ColQwen misses", ""]
    L.append(f"{m['n_questions']} eligible questions over {m['n_docs']} documents. Retriever: ColQwen2.5 within the "
             f"document. R@1 {m['R@1']:.3f}, R@{TOP_K} {m['R@5']:.3f}, MRR {m['MRR']:.3f}. "
             f"Misses (no gold page in top {TOP_K}): {m['n_miss']} ({m['miss_rate']:.1%}).")
    if m["andre_check"]:
        a = m["andre_check"]
        L.append(f"Check against idea4 Part B (question as text): {a['same_rank']} of {a['n']} questions have the "
                 f"same gold rank (max difference {a['max_abs_diff']}).")
    L += ["", "## Retrieval by document type", "", "| document type | n | R@5 | misses |", "|---|---|---|---|"]
    for t, v in sorted(m["by_doc_type"].items(), key=lambda kv: kv[1]["R@5"]):
        L.append(f"| {t} | {v['n']} | {v['R@5']:.2f} | {v['misses']} |")
    L += ["", "| pages | n | R@5 | misses |", "|---|---|---|---|"]
    for t, v in m["by_pages"].items():
        L.append(f"| {t} | {v['n']} | {v['R@5']:.2f} | {v['misses']} |")
    L += ["", "## Causes of misses (single label, first match in the order of the table)",
          "", "| cause | misses | share |", "|---|---|---|"]
    for c in CAUSES:
        k = m["cause_counts"][c]
        L.append(f"| {c} | {k} | {k / max(m['n_miss'], 1):.0%} |")
    L += ["", "Cause by document type (misses):", "", "| document type | " + " | ".join(CAUSES) + " |",
          "|---|" + "---|" * len(CAUSES)]
    ct = pd.crosstab(df.loc[df["miss"], "doc_type"], df.loc[df["miss"], "cause"]).reindex(columns=CAUSES, fill_value=0)
    for t, row in ct.iterrows():
        L.append(f"| {t} | " + " | ".join(str(int(v)) for v in row.values) + " |")
    L += ["", "## Flags among misses and hits (multi-label)", "", "| flag | misses | hits |", "|---|---|---|"]
    for k, v in m["flag_rates"].items():
        L.append(f"| {k} | {v['misses']:.0%} ({v['n_misses']}) | {v['hits']:.0%} |")
    L.append("(For hits, look-alike means the top non-gold page is among the gold page's 3 nearest pages.)")
    h = m["H1"]
    L += ["", "## H1: at least 25% of misses are look-alike misses, above chance", "",
          f"Look-alike misses: {h['lookalike_misses']} of {m['n_miss']} = {h['share']:.1%} "
          f"[95% CI {h['share_ci'][0]:.1%}, {h['share_ci'][1]:.1%}]. Chance: {h['chance_share']:.1%}. "
          f"P(chance >= observed) = {h['p_vs_chance']:.4f}. Verdict: {h['verdict']}.",
          "", f"## H2: logistic regression of miss (n = {m['H2_n']})", "",
          "| term | odds ratio [95% CI] | p |", "|---|---|---|"]
    for t in m["H2_logit"]:
        if t["term"] != "intercept":
            L.append(f"| {t['term']} | {t['odds_ratio']:.2f} [{t['or_ci'][0]:.2f}, {t['or_ci'][1]:.2f}] | {t['p']:.4f} |")
    L.append("nn_z is per standard deviation of the document's page-to-page similarity; type terms are against text.")
    L += ["", "## Misses, most severe first", "", "| qid | doc | pages | gold | rank | winner | winner nn rank | cause | question |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in df[df["miss"]].sort_values("best_rank", ascending=False).itertuples():
        q = r.question.replace("|", "/")
        L.append(f"| {r.qid} | {r.doc_id[:30]} | {r.n_pages} | {r.best_gold} | {r.best_rank} | {r.winner} | "
                 f"{r.winner_nn_rank} | {r.cause} | {q[:120]} |")
    (ATTR / "summary.md").write_text("\n".join(L) + "\n")


def main():
    ATTR.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(ATTR / "questions.csv", index=False)
    m = metrics(df)
    (ATTR / "metrics.json").write_text(json.dumps(m, indent=1))
    summary(df, m)
    fig_causes(df, m)
    fig_winner(df)
    fig_nn(df)
    fig_gallery(df)
    print((ATTR / "summary.md").read_text()[:3000])


if __name__ == "__main__":
    main()
