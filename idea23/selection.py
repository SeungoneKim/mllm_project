"""Step 3 (CPU): page selection, and retrieval metrics for every method.

  IDEA23_VERSION=v1 python selection.py   # reads out/scores.pkl, out/decomp.jsonl, idea1/scores.pkl (+ the inputs
                        # of v1/v2 below); writes out/<version>/{selections.jsonl, retrieval.csv, retrieval.md,
                        # sensitivity.csv} and for v2 evidence_choices.jsonl
  IDEA23_VERSION=v2 python selection.py --candidates   # the elements the VLM reads -> out/<version>/candidates.*

A version computes the methods listed in versions.py. v1 adds coverage_mod(_beam), v2 the element-level methods and
evidence (evidence.py); they read out/need_forms.jsonl, the element index, out/element_scores.pkl and, for evidence,
out/<version>/unit_scores.pkl.

Notation (as in the report). A document has pages p_1..p_N, a question q has gold evidence pages G, and a selector
returns a set S of k pages for the reader. s(x, p) is the ColQwen2.5 MaxSim score of query x against page p, and
s^(x, p) its min-max normalisation over the N pages of the document, so every query's scores lie in [0, 1].

Retrievers (rank the whole document):
  topk      top k by s(q, p)                                                    M3DocRAG (Cho et al., 2024)
  bm25      top k by BM25 over the PDF text layer (idea1/scores.pkl)            text channel only
  hybrid    top k by z(s(q, .)) + z(bm25(q, .)), z-scored over the document     image + text channels fused
Rerankers (choose k pages from the candidate pool C = the ColQwen top-20, shared by all of them):
  mmr         greedy max of  a s^(q, p) - (1 - a) max_{p' in S} cos(e_p, e_p'),  a = 0.5    MMR / DF-RAG style
  decomp_rrf  reciprocal-rank fusion (c = 60) of the rankings of q and of each need d_j     Ammann et al., 2025
  coverage    greedy max of  F(S) = sum_j w_j max_{p in S} s^(d_j, p) + lam sum_{p in S} s^(q, p),
              w_j = 1/m, lam = 1/k (both terms in [0, 1])                                  ours, v0
  coverage_beam   the same objective with beam search, B = 3 partial sets                 ours, v0 ablation
  coverage_full   coverage over the whole document instead of C                            ablation (retrieval only)
Oracle:
  gold      the gold evidence pages (reader upper bound)

Defaults are fixed a priori; out/sensitivity.csv reports how the retrieval metrics move with a and lam on the same
test questions, for discussion only (no setting is chosen from it).
"""
import json

import numpy as np
import pandas as pd

from common import IDEA1_SCORES, OUT, VERSION, VOUT, load_elements, load_qa, read_jsonl, test_questions
from evidence import ANCHOR_PAGES, GAMMA, Elements, coverage_with_units, need_gates, page_forms
from versions import VERSIONS

KS = (1, 3, 5, 10)
POOL = 20
MMR_ALPHA = 0.5
RRF_C = 60
LAMBDA_K = 1.0  # lam = LAMBDA_K / k
BEAM = 3
RETRIEVERS = ["topk", "bm25", "hybrid"]
RERANKERS = ["mmr", "decomp_rrf", "coverage", "coverage_beam", "coverage_full"]


def minmax(x):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def ranking(score):
    """0-indexed pages, best first; ties keep page order."""
    return np.argsort(-np.asarray(score, float), kind="stable")


def rrf(scores, c=RRF_C, cand=None):
    """Reciprocal-rank fusion of several score vectors; ranks are taken within `cand` when given."""
    n = len(scores[0])
    idx = np.arange(n) if cand is None else np.asarray(cand)
    fused = np.zeros(n)
    for s in scores:
        order = idx[ranking(np.asarray(s)[idx])]
        fused[order] += 1.0 / (c + np.arange(1, len(order) + 1))
    if cand is not None:
        mask = np.full(n, -np.inf)
        mask[idx] = 0
        fused = fused + mask
    return fused


def mmr(sq, pooled, k, cand, alpha=MMR_ALPHA):
    sim = pooled @ pooled.T
    sel, left = [], list(cand)
    while left and len(sel) < k:
        red = sim[np.ix_(left, sel)].max(1) if sel else np.zeros(len(left))
        val = alpha * sq[left] - (1 - alpha) * red
        sel.append(left.pop(int(np.argmax(val))))
    return sel


def coverage_greedy(sn, sq, k, cand, lam, init=()):
    """Greedy maximisation of F(S); sn is [m, N] (normalised need scores), sq is [N]. Ties go to the earlier
    candidate, i.e. the higher ColQwen rank. `init` pages are taken first (v2 evidence fills up after its facts)."""
    w = np.full(len(sn), 1.0 / len(sn))
    sel = list(init)
    cov = sn[:, sel].max(1) if sel else np.zeros(len(sn))
    left = [c for c in cand if c not in sel]
    while left and len(sel) < k:
        gain = (w[:, None] * np.maximum(sn[:, left] - cov[:, None], 0)).sum(0) + lam * sq[left]
        p = left.pop(int(np.argmax(gain)))
        sel.append(p)
        cov = np.maximum(cov, sn[:, p])
    return sel


def coverage_beam(sn, sq, k, cand, lam, beam=BEAM):
    """Beam search over partial sets: keep the `beam` best sets of each size (duplicates as sets removed)."""
    w = np.full(len(sn), 1.0 / len(sn))
    beams = [((), np.zeros(len(sn)), 0.0)]  # (pages in order, coverage vector, relevance sum)
    for _ in range(min(k, len(cand))):
        seen = {}
        for pages, cov, rel in beams:
            for c in cand:
                if c in pages:
                    continue
                key = frozenset(pages + (c,))
                ncov, nrel = np.maximum(cov, sn[:, c]), rel + sq[c]
                val = float(w @ ncov + lam * nrel)
                if key not in seen or val > seen[key][0] + 1e-12:
                    seen[key] = (val, pages + (c,), ncov, nrel)
        best = sorted(seen.values(), key=lambda t: -t[0])[:beam]
        beams = [(p, cv, r) for _, p, cv, r in best]
    return list(beams[0][0])


def select_all(row, sq_raw, bm25, sn_raw, pooled, k, alpha=MMR_ALPHA, lam_k=LAMBDA_K):
    """Selections (1-indexed pages, in selection order) of every method for one question and budget k."""
    sq = minmax(sq_raw)
    sn = np.stack([minmax(s) for s in sn_raw]) if sn_raw is not None else sq[None]
    cand = list(ranking(sq_raw)[:POOL])
    lam = lam_k / k
    out = {
        "topk": list(ranking(sq_raw)[:k]),
        "bm25": list(ranking(bm25)[:k]),
        "hybrid": list(ranking(zscore(sq_raw) + zscore(bm25))[:k]),
        "mmr": mmr(sq, pooled, k, cand, alpha),
        "decomp_rrf": list(ranking(rrf([sq_raw, *([] if sn_raw is None else sn_raw)], cand=cand))[:k]),
        "coverage": coverage_greedy(sn, sq, k, cand, lam),
        "coverage_beam": coverage_beam(sn, sq, k, cand, lam),
        "coverage_full": coverage_greedy(sn, sq, k, list(ranking(sq_raw)), lam),
    }
    return {m: [int(p) + 1 for p in s] for m, s in out.items()}


def select_mod(sq_raw, sn_raw, k, gates, lam_k=LAMBDA_K):
    """v1: coverage with each need's page scores multiplied by its form gates g(j, p) (evidence.py)."""
    sq = minmax(sq_raw)
    sn = np.stack([minmax(s) for s in sn_raw]) * gates
    cand, lam = list(ranking(sq_raw)[:POOL]), lam_k / k
    out = {"coverage_mod": coverage_greedy(sn, sq, k, cand, lam), "coverage_mod_beam": coverage_beam(sn, sq, k, cand, lam)}
    return {m: [int(p) + 1 for p in s] for m, s in out.items()}


def select_elements(E, k, gates, units=None, lam_k=LAMBDA_K, evidence=False):
    """v2: page selection from element scores (evidence.Elements), and with `evidence` from the VLM's facts too.
    units = (options per fact, r_need [m, U], r_q [U]), or None for a question the VLM gave no fact for, whose
    evidence selection is then elem_coverage_mod's. -> (selections, indices of the facts covered)."""
    pq, lam, cand = E.page_max(E.q)[0], lam_k / k, E.pool(POOL)
    pn, pn_mod = E.page_max(E.need), E.page_max(gates * E.need)
    out = {"elem_topk": list(ranking(pq)[:k]), "elem_coverage": coverage_greedy(pn, pq, k, cand, lam),
           "elem_coverage_mod": coverage_greedy(pn_mod, pq, k, cand, lam)}
    got = []
    if evidence:
        pages = []
        if units is not None and units[0]:
            opts, r_need, r_q = units
            pages, got = coverage_with_units(opts, r_need, r_q, k, lam)
        out["evidence"] = coverage_greedy(pn_mod, pq, k, cand, lam, init=pages)
    return {m: [int(p) + 1 for p in s] for m, s in out.items()}, got


def metrics(sel, gold):
    s, g = set(sel), set(gold)
    return {"recall": len(s & g) / len(g), "complete": float(g <= s), "any": float(bool(s & g))}


def load_inputs():
    qs = test_questions(load_qa())
    sc = pd.read_pickle(OUT / "scores.pkl")
    bm25 = pd.read_pickle(IDEA1_SCORES)["bm25"]
    decomp = {r["qid"]: r["needs"] for r in read_jsonl(OUT / "decomp.jsonl")}
    return qs, sc, bm25, decomp


def check_against_idea1(qs, sc):
    """Our ColQwen2.5 run should reproduce Analysis 1's scores up to rendering differences."""
    old = pd.read_pickle(IDEA1_SCORES)["colqwen"]
    m = qs[qs.set == "multi"]
    for name, get in [("this run", lambda q: sc["q"][q]), ("idea1/scores.pkl", lambda q: old[q])]:
        c = np.mean([set(g) <= set(ranking(get(q))[:5] + 1) for q, g in zip(m.qid, m.gold)])
        print(f"  complete@5 on multi with {name}: {100 * c:.1f}")
    top1 = np.mean([ranking(sc["q"][q])[0] == ranking(old[q])[0] for q in qs.qid])
    print(f"  same top-1 page as idea1: {100 * top1:.1f}%")


def table(rows, keys):
    df = pd.DataFrame(rows)
    df["group"] = df["set"]
    both = pd.concat([df, df[df.set == "multi"].assign(group=lambda d: d.stratum)])
    agg = both.groupby(["group", *keys])[["recall", "complete"]].mean().mul(100).round(1)
    agg["n"] = both.groupby(["group", *keys]).size()
    return agg.reset_index()


class VersionInputs:
    """What v1 and v2 add to the page scores: need forms, the element index, element scores and the VLM's facts."""

    def __init__(self, qs, sc, decomp, methods, facts=True):
        self.mod = any(m.startswith("coverage_mod") for m in methods)
        self.elem = any(m.startswith("elem_") or m == "evidence" for m in methods)
        self.sc, self.decomp, self.forms, self.els, self.facts = sc, decomp, {}, {}, None
        if self.mod or self.elem:
            self.forms = {r["qid"]: r["forms"] for r in read_jsonl(OUT / "need_forms.jsonl")}
            self.els = {d: load_elements(d) for d in qs.doc_id.unique()}
            print(f"need forms for {len(self.forms)} questions, elements for {len(self.els)} documents "
                  f"({sum(map(len, self.els.values()))} elements)")
        if self.elem:
            self.es = pd.read_pickle(OUT / "element_scores.pkl")
            if facts and "evidence" in methods:
                self.facts = load_facts()

    def need_forms(self, qid, m):
        f = self.forms.get(qid)
        return f if f is not None and len(f) == m else [["any"]] * m

    def elements(self, r):
        sn = self.sc["need"][r.qid]
        return Elements(self.els[r.doc_id], r.doc_pages, self.es["q"][r.qid], self.es["need"][r.qid],
                        self.sc["q"][r.qid], sn)

    def units(self, r, E, gamma=GAMMA):
        """The VLM's facts of one question as (options, r_need, r_q), or None when it has no aggregation."""
        f = self.facts.get(r.qid) if self.facts is not None else None
        if f is None:
            return None
        opts = []
        for fact in f["facts"]:
            o = [{int(E.page[i]) for i in fact["evidence"]}]
            if len(fact["evidence"]) == 1:
                o += [{int(E.page[i])} for i in fact["same_as"]]
            opts.append(o)
        r_need = np.clip(f["r_need"], 0, None)
        for u, fact in enumerate(f["facts"]):  # a fact counts fully for the needs the VLM says it serves
            if fact["needs"]:
                r_need[[j for j in range(len(r_need)) if j not in fact["needs"]], u] *= gamma
        return opts, r_need, np.clip(f["r_q"], 0, None)


def load_facts():
    """{qid: {"facts": [...], "r_need": [m, U], "r_q": [U]}} from aggregate.py and units.py."""
    us = OUT / VERSION / "unit_scores.pkl"
    return pd.read_pickle(us) if us.exists() else None


def write_candidates(qs, vin):
    """v2: the elements shown to the VLM for each question (aggregate.py), and how often they hold the evidence."""
    rows, lines = [], []
    for r in qs.itertuples():
        E = vin.elements(r)
        forms = vin.need_forms(r.qid, len(vin.decomp[r.qid]))
        gates = need_gates(forms, E.form)
        anchor = coverage_greedy(E.page_max(gates * E.need), E.page_max(E.q)[0], ANCHOR_PAGES, E.pool(POOL),
                                 LAMBDA_K / 5)
        idx = E.candidates(forms, anchor)
        pages = sorted({int(E.page[i]) + 1 for i in idx})
        rows.append({"qid": int(r.qid), "set": r.set, "needs": vin.decomp[r.qid], "elements": [int(i) for i in idx],
                     "pages": pages, "complete": set(r.gold) <= set(pages),
                     "page_ceiling": set(r.gold) <= set(ranking(vin.sc["q"][r.qid])[:len(pages)] + 1)})
    VOUT.mkdir(parents=True, exist_ok=True)
    with (VOUT / "candidates.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    df = pd.DataFrame(rows)
    for s, g in df.groupby("set"):
        lines.append(f"{s}: {len(g)} questions, {g.elements.map(len).mean():.1f} candidate elements on "
                     f"{g.pages.map(len).mean():.1f} pages; all gold pages among them {100 * g.complete.mean():.1f}%, "
                     f"in the ColQwen top-n pages for the same n {100 * g.page_ceiling.mean():.1f}%")
    (VOUT / "candidates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(candidates=False):
    qs, sc, bm25, decomp = load_inputs()
    methods = VERSIONS[VERSION]["methods"]
    print(f"version {VERSION} ({VERSIONS[VERSION]['name']}): {', '.join(methods)}")
    print(f"{len(qs)} test questions ({(qs.set == 'multi').sum()} multi, {(qs.set == 'control').sum()} control); "
          f"{sum(q in sc['need'] for q in qs.qid)} with decomposed needs")
    vin = VersionInputs(qs, sc, decomp, methods, facts=not candidates)
    if candidates:
        return write_candidates(qs, vin)
    if vin.elem and "evidence" in methods and vin.facts is None:
        print("no VLM facts yet (units.py): evidence is left out")
        methods = [m for m in methods if m != "evidence"]
    check_against_idea1(qs, sc)

    sel_rows, met_rows, sens_rows, choices = [], [], [], []
    for r in qs.itertuples():
        sq, b = sc["q"][r.qid], np.asarray(bm25[r.qid], float)
        assert len(sq) == len(b) == r.doc_pages, (r.qid, len(sq), len(b), r.doc_pages)
        sn, pooled = sc["need"].get(r.qid), sc["pooled"][r.doc_id]
        base = {"qid": int(r.qid), "set": r.set, "stratum": r.stratum}
        forms = vin.need_forms(r.qid, len(sn)) if vin.mod or vin.elem else None
        gates = need_gates(forms, page_forms(vin.els[r.doc_id], r.doc_pages)) if vin.mod else None
        E = vin.elements(r) if vin.elem else None
        egates = need_gates(forms, E.form) if vin.elem else None
        units = vin.units(r, E) if vin.elem else None
        for k in KS:
            sel = select_all(r, sq, b, sn, pooled, k)
            if vin.mod:
                sel.update(select_mod(sq, sn, k, gates))
            if vin.elem:
                more, got = select_elements(E, k, egates, units, evidence="evidence" in methods)
                sel.update(more)
                if "evidence" in methods:
                    choices.append({"qid": int(r.qid), "k": k, "facts": got, "pages": sel["evidence"],
                                    "fallback": units is None})
            for m in [m for m in sel if m in methods]:
                sel_rows.append({**base, "method": m, "k": k, "pages": sel[m]})
                met_rows.append({**base, "method": m, "k": k, **metrics(sel[m], r.gold)})
        sel_rows.append({**base, "method": "gold", "k": None, "pages": list(r.gold)})
        if r.set == "multi":  # sensitivity of the free constants, retrieval only
            for k in (3, 5):
                if VERSION == "v0":
                    for a in (0.3, 0.5, 0.7, 0.9):
                        pages = [int(p) + 1 for p in mmr(minmax(sq), pooled, k, list(ranking(sq)[:POOL]), a)]
                        sens_rows.append({**base, "method": "mmr", "param": f"alpha={a}", "k": k,
                                          **metrics(pages, r.gold)})
                    for lk in (0.0, 0.5, 1.0, 2.0, 4.0):
                        pages = select_all(r, sq, b, sn, pooled, k, lam_k=lk)["coverage"]
                        sens_rows.append({**base, "method": "coverage", "param": f"lam*k={lk}", "k": k,
                                          **metrics(pages, r.gold)})
                if vin.mod:
                    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
                        pages = select_mod(sq, sn, k, need_gates(forms, page_forms(vin.els[r.doc_id], r.doc_pages),
                                                                 g))["coverage_mod"]
                        sens_rows.append({**base, "method": "coverage_mod", "param": f"gamma={g}", "k": k,
                                          **metrics(pages, r.gold)})
                if vin.elem:
                    for be in (0.0, 0.25, 0.5, 0.75, 1.0):
                        Eb = Elements(E.els, r.doc_pages, vin.es["q"][r.qid], vin.es["need"][r.qid], sq, sn, beta=be)
                        pages = select_elements(Eb, k, egates)[0]["elem_coverage_mod"]
                        sens_rows.append({**base, "method": "elem_coverage_mod", "param": f"beta={be}", "k": k,
                                          **metrics(pages, r.gold)})
                    for g in (0.0, 0.25, 0.75, 1.0):
                        pages = select_elements(E, k, need_gates(forms, E.form, g))[0]["elem_coverage_mod"]
                        sens_rows.append({**base, "method": "elem_coverage_mod", "param": f"gamma={g}", "k": k,
                                          **metrics(pages, r.gold)})

    VOUT.mkdir(parents=True, exist_ok=True)
    with (VOUT / "selections.jsonl").open("w") as f:
        for row in sel_rows:
            f.write(json.dumps(row) + "\n")
    if choices:
        with (VOUT / "evidence_choices.jsonl").open("w") as f:
            for row in choices:
                f.write(json.dumps(row) + "\n")
    ret = table(met_rows, ["method", "k"])
    ret.to_csv(VOUT / "retrieval.csv", index=False)
    sens = table(sens_rows, ["method", "param", "k"])
    sens.to_csv(VOUT / "sensitivity.csv", index=False)

    order = [m for m in methods if m != "gold"]
    lines = ["# Retrieval (recall@k / complete-evidence recall@k, %)", ""]
    for group in ["multi", "same_mod", "cross_multi", "control"]:
        g = ret[ret.group == group]
        n = int(g.n.iloc[0])
        lines += [f"## {group} (n = {n})", "", "| method | " + " | ".join(f"k={k}" for k in KS) + " |",
                  "|---|" + "---|" * len(KS)]
        for m in order:
            cells = [g[(g.method == m) & (g.k == k)].iloc[0] for k in KS]
            lines.append(f"| {m} | " + " | ".join(f"{c.recall:.1f} / {c.complete:.1f}" for c in cells) + " |")
        lines.append("")
    multi = qs[qs.set == "multi"]
    pool = np.mean([set(g) <= set(ranking(sc["q"][q])[:POOL] + 1) for q, g in zip(multi.qid, multi.gold)])
    lines.append(f"Candidate pool ceiling (all gold pages in ColQwen top-{POOL}), multi: {100 * pool:.1f}%")
    if vin.elem:
        epool = np.mean([set(r.gold) <= {int(p) + 1 for p in vin.elements(r).pool(POOL)} for r in multi.itertuples()])
        lines.append(f"Element pool ceiling (all gold pages in the top-{POOL} pages by best element), multi: "
                     f"{100 * epool:.1f}%")
    (VOUT / "retrieval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    s = sens[sens.group == "multi"]
    if len(s):
        print("\nSensitivity (multi, complete-evidence recall %):")
        print(s.pivot_table(index=["method", "param"], columns="k", values="complete").to_string())


if __name__ == "__main__":
    import sys
    main(candidates="--candidates" in sys.argv[1:])
