import ast, difflib, itertools, re, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = sys.argv[1] if len(sys.argv) > 1 else "data"
df = pd.read_parquet(f"{D}/data/train-00000-of-00001.parquet")
S = pd.read_pickle("scores.pkl")
text, df["img"], df["txt"] = S["text"], S["colqwen"], S["bm25"]
df["unans"] = df.answer_format == "None"
df["gold"] = df.evidence_pages.apply(ast.literal_eval)
ok = pd.Series([all(1 <= p <= len(text[d]) for p in g) for d, g in zip(df.doc_id, df.gold)], index=df.index)
a = df[df.unans | (ok & (df.gold.str.len() > 0))]  # drop answerable qs w/ broken page labels
rich = {d: np.mean([len(t) >= 50 for t in ts]) >= 0.8 for d, ts in text.items()}  # no OCR yet so only docs w/ a text layer
q = a[a.doc_id.map(rich)]
y = q.unans.to_numpy()
best = lambda v: q[v].map(np.max).to_numpy()
r = lambda x: np.round(x, 3)
print(len(a), "qs", a.unans.sum(), "unans |", len(q), "qs in", sum(rich.values()), "text docs | pages w/o text",
      r(np.mean([len(t) < 50 for ts in text.values() for t in ts])))

# H1: can the scores alone tell when to abstain
print("\nH1 auroc best score: colqwen", r(roc_auc_score(y, -best("img"))), "bm25", r(roc_auc_score(y, -best("txt"))))
def feats(s, n):
    t = np.sort(s)[::-1]
    return [t[0], t[0] - t[1], (t[0] - s.mean()) / (s.std() + 1e-9), t[:5].mean(), t[0] / n]
n = q.question.map(lambda s: len(re.findall(r"\w+", s)))
X = np.array([feats(i, k) + feats(t, k) for i, t, k in zip(q.img, q.txt, n)])
print("each feature:", r([max(roc_auc_score(y, x), 1 - roc_auc_score(y, x)) for x in X.T]))
lr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), X, y, groups=q.doc_id,
                       cv=GroupKFold(5), method="decision_function")
print("logreg on all 10, cv by doc:", r(roc_auc_score(y, lr)))

# twins = each unans q + the most similar answerable q on the same doc
W = lambda s: re.findall(r"\w+", s.lower())
pairs, dup = [], 0
for _, g in a.groupby("doc_id"):
    cands = list(g[~g.unans].itertuples())
    for u in g[g.unans].itertuples():
        if not cands: continue
        sim, x = max(((difflib.SequenceMatcher(None, W(u.question), W(x.question)).ratio(), x) for x in cands), key=lambda t: t[0])
        if W(u.question) == W(x.question): dup += 1
        elif sim >= 0.8: pairs.append((u, x))
rd = lambda s1, s2: abs(s1 - s2) / max(s1, s2)
spread = lambda f, v: np.median([rd(*c) for _, g in f.groupby("doc_id") for c in itertools.combinations(g[v].map(np.max), 2) if max(c) > 0])
bu, ba = np.array([u.img.max() for u, x in pairs]), np.array([x.img.max() for u, x in pairs])
tp = [(u, x) for u, x in pairs if rich[u.doc_id]]
print("twins", len(pairs), "+", dup, "identical | same top page", r(np.mean([u.img.argmax() == x.img.argmax() for u, x in pairs])),
      "| top page is evidence for the twin", r(np.mean([u.img.argmax() + 1 in x.gold for u, x in pairs])))
print("colqwen: unans twin lower", r(np.mean(bu < ba)), "median diff", r(np.median([rd(*z) for z in zip(bu, ba)])),
      "vs 2 answerable qs", r(spread(a[~a.unans], "img")))
print("bm25 on", len(tp), "text pairs: median diff", r(np.median([rd(u.txt.max(), x.txt.max()) for u, x in tp])),
      "vs", r(spread(a[~a.unans & a.doc_id.map(rich)], "txt")))

# H2: abstain if a number from the q isnt in the text of the top 5 pages
nums = lambda s: re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)", re.sub(r"(?<=\d),(?=\d{3})", "", s.lower()))
def missing(row):
    t = " ".join(text[row.doc_id][p] for p in np.argsort(-row.img)[:5])
    return [k for k in nums(row.question) if not re.search(rf"(?<![\w.]){re.escape(k)}(?![\w]|\.\d)", t)]
miss = [missing(row) for row in q.itertuples()]
flag = np.array([len(m) > 0 for m in miss])
fa, ca = flag[~y].mean(), flag[y].mean()
at = lambda v: np.mean(best(v)[y] < np.quantile(best(v)[~y], fa))
print("\nH2 rule catches", r(ca), "| false abst", r(fa), "| threshold at that rate: colqwen", r(at("img")), "bm25", r(at("txt")))
print("unans w/ a number", r(np.mean([len(nums(s)) > 0 for s in q.question[y]])), "| text twins, fires on unans only",
      r(np.mean([bool(missing(u)) and not missing(x) for u, x in tp])), "ans only", r(np.mean([bool(missing(x)) and not missing(u) for u, x in tp])))
for row, m in zip(q.itertuples(), miss):
    if m: print("  UNANS" if row.unans else "  ans  ", len(text[row.doc_id]), "pages |", m, "|", " ".join(row.question.split()))

# H3: page budget
rank = lambda s: np.argsort(np.argsort(-s)) + 1
gr = [rank(s)[np.array(g, int) - 1] for s, g in zip(a.img, a.gold)]
isa = ~a.unans.to_numpy()
rec = lambda n: np.array([np.mean(g <= k) for g, k, o in zip(gr, n, isa) if o])
last = np.array([g.max() for g, o in zip(gr, isa) if o])
print("\nH3 last evidence page is 1st for", r(np.mean(last == 1)), "| below 5th for", r(np.mean(last > 5)))
topk = np.array([(k, rec([k] * len(a)).mean()) for k in range(1, 11)])
orc = []
for k in [*range(1, 11), 12, 15, 20, 30, 50]:  # oracle stops at the last gold page inside the top k
    n = [g[g <= k].max() if (g <= k).any() else 0 for g in gr]
    orc.append((np.mean(n), rec(n).mean()))
orc = np.array(orc)
def kept(c):  # nb of pages sparsemax keeps (= its support size)
    out = []
    for s in a.img:
        z = np.sort(c * (s - s.mean()) / (s.std() + 1e-9))[::-1]
        k = np.arange(1, len(z) + 1)
        out.append(k[1 + k * z > np.cumsum(z)][-1])
    return np.array(out)
sp = []
for c in [0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5]:
    n = kept(c)
    sp.append((n.mean(), rec(n).mean(), np.interp(n.mean(), *topk.T)))
sp = np.array(sorted(sp))
print("sparsemax (pages, recall, top-k recall at same pages)\n", r(sp))
n3 = kept(0.75)  # ~3 pages per q
gain = rec(n3) - rec([3] * len(a))
rng = np.random.default_rng(0)
ci = np.percentile([rng.choice(gain, len(gain)).mean() for _ in range(5000)], [2.5, 97.5])
t3, s3, o3 = topk[2, 1], rec(n3).mean(), np.interp(3, *orc.T)
print("at 3 pages: top-3", r(t3), "sparsemax", r(s3), "oracle", r(o3), "| gap closed", r((s3 - t3) / (o3 - t3)), "| gain ci (points)", np.round(100 * ci, 2))

plt.rcParams.update({"font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
                     "legend.frameon": False, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
OR, BL, GR, GY = "#eb6834", "#2a78d6", "#1baf7a", "#a3a29c"
def save(ax, name, **kw):
    ax.set(**kw)
    ax.figure.tight_layout(pad=0.3)
    ax.figure.savefig(name)

ax = plt.subplots(figsize=(2.2, 1.95))[1]
ax.plot([0, 100], [0, 100], ":", lw=1, color=GY)
for v, c, l in [("img", OR, "ColQwen2.5 score"), ("txt", BL, "BM25 score")]:
    fpr, tpr, _ = roc_curve(y, -best(v))
    ax.plot(100 * fpr, 100 * tpr, lw=1.4, color=c, label=l)
ax.plot(100 * fa, 100 * ca, "o", ms=5, color=GR, mec="white", mew=0.8, label="missing-number check")
ax.legend(loc="lower right", handlelength=1.5)
save(ax, "idea1_abstention.pdf", xlim=(0, 100), ylim=(0, 100), xticks=[0, 50, 100], yticks=[0, 50, 100],
     xlabel="answerable abstained (%)", ylabel="unanswerable caught (%)")

ax = plt.subplots(figsize=(2.2, 1.95))[1]
ax.plot([5, 65], [5, 65], ":", lw=1, color=GY)
ax.scatter(ba, bu, s=7, color=OR, edgecolors="white", linewidths=0.3)
save(ax, "idea1_twins.pdf", xlim=(5, 65), ylim=(5, 65), aspect="equal", xticks=[20, 40, 60], yticks=[20, 40, 60],
     xlabel="answerable question", ylabel="unanswerable twin")

ax = plt.subplots(figsize=(2.2, 1.95))[1]
ax.plot(orc[:, 0], 100 * orc[:, 1], "--", lw=1.2, color=GY, label="oracle cut")
ax.plot(topk[:, 0], 100 * topk[:, 1], "o-", lw=1.4, ms=2.5, color=OR, label="fixed top-$k$")
ax.plot(sp[:, 0], 100 * sp[:, 1], "o-", lw=1.4, ms=2.5, color=GR, label="sparsemax (untrained)")
ax.legend(loc="lower right", handlelength=1.5)
save(ax, "idea1_page_budget.pdf", xlim=(0, 10), ylim=(40, 100), xlabel="pages read per question", ylabel="evidence pages found (%)")
