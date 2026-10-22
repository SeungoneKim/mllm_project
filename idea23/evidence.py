"""Modality-aware and evidence-level selection (v1 and v2), used by selection.py and aggregate.py.

Notation as in selection.py, plus: a document's elements e (segment.py) each lie on one page p(e) and have one form
f(e); F_j is the set of forms need d_j expects (need_forms.py), or "any". s(x, e) is the ColQwen2.5 score of query x
against the crop of e.

v1 (page_mod). A page p has the forms of its elements, f(p). A need is credited in full by a page holding one of its
forms and at a discount gamma otherwise:
    g(j, p) = 1 if F_j = any, f(p) unknown, or F_j and f(p) share a form; gamma otherwise        (gamma = 0.5)
    coverage_mod: coverage (selection.py) with s^(d_j, p) replaced by g(j, p) s^(d_j, p)
v2 (evidence). An element is scored with its own crop and with its page,
    s~(x, e) = beta s^_E(x, e) + (1 - beta) s^(x, p(e))                                          (beta = 0.5)
where s^_E is min-max normalised over the document's elements. A page is as relevant as its best element:
    elem_topk          the k pages with the largest max_{e on p} s~(q, e)
    elem_coverage      coverage with s^(x, p) replaced by max_{e on p} s~(x, e)
    elem_coverage_mod  the same with max_{e on p} g(j, e) s~(d_j, e), g as above with f(e) for f(p)
    evidence           coverage over the atomic facts that the VLM read off the candidate elements (aggregate.py),
                       each fact costing the pages of its elements; the remaining pages by elem_coverage_mod
The VLM sees at most 28 candidate elements per question (Elements.candidates): the best elements of the first 8 pages
of elem_coverage_mod, then per-form retrieval for the question and each need.
All of them choose from the candidate pool C_E, the 20 pages with the largest max_e s~(q, e), and return k pages.
"""
import numpy as np

from common import FORMS

GAMMA = 0.5
BETA = 0.5
MAX_CANDIDATES = 28  # elements shown to the VLM per question
ANCHOR_PAGES = 8     # pages of elem_coverage_mod whose best elements are always candidates
PER_FORM = (3, 1)    # elements per form for a need: of an expected form, of any other form


def minmax(x):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def page_forms(els, n_pages):
    """The set of forms on each page (0-indexed list); a page segmented as a whole (form Page) is unknown (None)."""
    out = [set() for _ in range(n_pages)]
    for e in els:
        out[e["page"] - 1].add(e["form"])
    return [None if (not f or "Page" in f) else f for f in out]


def need_gates(forms, units_forms, gamma=GAMMA):
    """[m, U] multipliers g(j, u) for needs with expected forms `forms` and units (pages or elements) whose forms
    are `units_forms` (a set, a single form, or None for unknown)."""
    g = np.ones((len(forms), len(units_forms)))
    for j, fj in enumerate(forms):
        if "any" in fj:
            continue
        for u, fu in enumerate(units_forms):
            fu = {fu} if isinstance(fu, str) else fu
            if fu is not None and "Page" not in fu and not set(fj) & fu:
                g[j, u] = gamma
    return g


class Elements:
    """The elements of one document with the smoothed scores s~ of a question and its needs."""

    def __init__(self, els, n_pages, eq, en, sq, sn, beta=BETA):
        self.els, self.n = els, n_pages
        self.page = np.array([e["page"] - 1 for e in els])
        self.form = [e["form"] for e in els]
        self.q = beta * minmax(eq) + (1 - beta) * minmax(sq)[self.page]
        self.need = np.stack([beta * minmax(a) + (1 - beta) * minmax(b)[self.page] for a, b in zip(en, sn)])

    def page_max(self, scores):
        """[..., N] the best element score on each page."""
        scores = np.atleast_2d(scores)
        out = np.zeros((scores.shape[0], self.n))
        for i in range(scores.shape[0]):
            np.maximum.at(out[i], self.page, scores[i])
        return out

    def pool(self, size):
        return list(np.argsort(-self.page_max(self.q)[0], kind="stable")[:size])

    def candidates(self, forms, anchor_pages=(), size=MAX_CANDIDATES):
        """The elements shown to the VLM, from two routes. (1) Page anchor: on each of the `anchor_pages` (the first
        pages elem_coverage_mod picks), its best element for the question and for each need. (2) Modality-decomposed
        retrieval: for the question and for each need, the best elements of each form (more of the forms the need
        expects), merged round-robin across the queries, until `size`. -> element indices.
        The anchor was added after the first diagnostic run: route (2) alone held all gold pages of 53.9% of the
        multi-page questions against 60.3% for ColQwen's top pages at the same page count (out/v2/candidates.md)."""
        out = []
        for p in anchor_pages:
            on = np.where(self.page == p)[0]
            for x in [self.q, *self.need]:
                i = int(on[np.argmax(x[on])])
                if i not in out:
                    out.append(i)
        lists = []
        for x, fx in [(self.q, ["any"])] + list(zip(self.need, forms)):
            per = []
            for f in FORMS + ["Page"]:
                idx = [i for i in np.argsort(-x, kind="stable") if self.form[i] == f]
                n = PER_FORM[0] if (f in fx or f == "Page") else PER_FORM[1] if "any" not in fx else 2
                per += idx[:n]
            lists.append(sorted(per, key=lambda i: -x[i]))
        for rank in range(max(map(len, lists))):
            for l in lists:
                if rank < len(l) and l[rank] not in out and len(out) < size:
                    out.append(l[rank])
        return out


def coverage_with_units(units, r_need, r_q, k, lam):
    """Greedy budgeted coverage over facts. A fact u can be bought through any of its options (sets of 0-indexed
    pages: its own elements together, or one redundant copy) and its gain is
        sum_j (1/m) max(0, r(u, d_j) - cov_j) + lam r(u, q),
    divided by the number of new pages it adds (at least 1). Facts whose option already lies inside the chosen pages
    come for free. Stops when nothing fits in k pages. -> (pages in order, indices of the facts covered)."""
    m = r_need.shape[0]
    cov, pages, got = np.zeros(m), [], set()

    def absorb():
        for u, opts in enumerate(units):
            if u not in got and any(o <= set(pages) for o in opts):
                got.add(u)
                np.maximum(cov, r_need[:, u], out=cov)

    while len(pages) < k:
        best = None
        for u, opts in enumerate(units):
            if u in got:
                continue
            gain = (np.maximum(r_need[:, u] - cov, 0)).sum() / m + lam * r_q[u]
            for o in opts:
                new = o - set(pages)
                if not new or len(pages) + len(new) > k:
                    continue
                val = gain / len(new)
                if best is None or val > best[0] + 1e-12:
                    best = (val, sorted(new))
        if best is None or best[0] <= 0:
            break
        pages += best[1]
        absorb()
    return pages, sorted(got)
