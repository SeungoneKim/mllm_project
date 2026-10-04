"""Part B: draw a specification with code (matplotlib), and draw questions as images.

  python render.py specs [--input proposer/specs.jsonl] [--out renders/code]   # one PNG per spec
  python render.py questions                                                  # renders/question/<pid>.png
  python render.py test                                                       # SYNTHETIC hand-written specs -> tests/renders_SYNTHETIC/

Charts use seeded plausible values when the spec gives none. Tables get placeholder numbers.
Text, figure and layout specs are drawn as a text block with a fixed font, size and canvas.
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import IDEA4, SEED  # noqa: E402

W, H, DPI = 8, 6, 100            # 800 x 600 px canvas for every render
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#222222"


def seeded_rng(key):
    return np.random.default_rng(SEED + sum(map(ord, str(key))))


def numbers(text, labels=()):
    """Non-negative numbers in the values text, after removing the label strings (so '18-29' is not read)."""
    text = str(text or "")
    for lab in sorted(map(str, labels), key=len, reverse=True):
        if lab:
            text = text.replace(lab, " ")
    return [float(x) for x in re.findall(r"(?<![\w.-])\d+(?:\.\d+)?", text)]


def chart_values(spec, n_cat, n_ser, rng, cats=(), sers=()):
    text = str(spec.get("values") or "")
    vals = numbers(text, list(cats) + list(sers))
    if len(vals) == n_cat * n_ser:
        # Category-major when the text is written per category ("North 10/20; South ..."), else series-major.
        pos = [text.find(str(c)) for c in cats]
        by_cat = n_ser > 1 and all(p >= 0 for p in pos)
        a = np.array(vals)
        return a.reshape(n_cat, n_ser).T if by_cat else a.reshape(n_ser, n_cat)
    return rng.uniform(10, 80, size=(n_ser, n_cat)).round()


def draw_chart(ax, spec, rng):
    sub = spec.get("subtype") or "bar"
    cats = spec.get("categories") or [f"Category {i + 1}" for i in range(4)]
    sers = spec.get("series") or [spec.get("y_label") or "Value"]
    cats, sers = cats[:12], sers[:6]
    v = chart_values(spec, len(cats), len(sers), rng, cats, sers)
    x = np.arange(len(cats))
    cols = PALETTE[:len(sers)]
    if sub.startswith("pie"):
        ax.pie(v[0], labels=cats, colors=(PALETTE * 3)[:len(cats)], autopct="%d%%", textprops={"fontsize": 9})
        ax.axis("equal")
    elif sub == "line" or sub == "area":
        for i, s in enumerate(sers):
            ax.plot(x, v[i], color=cols[i], lw=2, marker="o", label=s)
            if sub == "area":
                ax.fill_between(x, v[i], color=cols[i], alpha=0.3)
        ax.set_xticks(x, cats, rotation=30 if len(cats) > 5 else 0, ha="right" if len(cats) > 5 else "center")
    elif sub == "scatter":
        for i, s in enumerate(sers):
            ax.scatter(rng.uniform(0, 100, len(cats)), v[i], color=cols[i], s=40, label=s)
    elif sub == "horizontal bar":
        h = 0.8 / len(sers)
        for i, s in enumerate(sers):
            ax.barh(x + i * h, v[i], h, color=cols[i], label=s)
            for xi, vi in zip(x + i * h, v[i]):
                ax.text(vi + 1, xi, f"{vi:g}", va="center", fontsize=8)
        ax.set_yticks(x + h * (len(sers) - 1) / 2, cats)
        ax.invert_yaxis()
    elif sub == "stacked bar":
        bottom = np.zeros(len(cats))
        for i, s in enumerate(sers):
            ax.bar(x, v[i], 0.6, bottom=bottom, color=cols[i], label=s)
            bottom += v[i]
        ax.set_xticks(x, cats, rotation=30 if len(cats) > 5 else 0, ha="right" if len(cats) > 5 else "center")
    else:  # bar, grouped bar, other
        w = 0.8 / len(sers)
        for i, s in enumerate(sers):
            ax.bar(x + i * w, v[i], w, color=cols[i], label=s)
            for xi, vi in zip(x + i * w, v[i]):
                ax.text(xi, vi + 1, f"{vi:g}", ha="center", fontsize=8)
        ax.set_xticks(x + w * (len(sers) - 1) / 2, cats, rotation=30 if len(cats) > 5 else 0,
                      ha="right" if len(cats) > 5 else "center")
    if not sub.startswith("pie"):
        ax.set_xlabel(spec.get("x_label", ""))
        ax.set_ylabel(spec.get("y_label", ""))
        ax.spines[["top", "right"]].set_visible(False)
        if len(sers) > 1 or spec.get("series"):
            ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title(textwrap.fill(spec.get("title", ""), 70), fontsize=12, loc="left", fontweight="bold")


def draw_table(fig, spec, rng):
    cols = (spec.get("columns") or ["Column 1", "Column 2", "Column 3"])[:8]
    rows = (spec.get("rows") or [f"Row {i + 1}" for i in range(5)])[:14]
    vals = numbers(spec.get("values"))
    n = len(rows) * (len(cols) - 1)
    cells = vals[:n] if len(vals) >= n else list(rng.integers(1, 999, n))
    body = [[r] + [f"{c:g}" for c in cells[i * (len(cols) - 1):(i + 1) * (len(cols) - 1)]] for i, r in enumerate(rows)]
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.82])
    ax.axis("off")
    t = ax.table(cellText=[[textwrap.shorten(str(c), 30) for c in row] for row in body],
                 colLabels=[textwrap.shorten(str(c), 30) for c in cols], loc="upper center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#999999")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eef6")
    fig.text(0.04, 0.93, textwrap.fill(spec.get("caption", ""), 80), fontsize=12, fontweight="bold", va="top")


def draw_text(fig, spec):
    head = spec.get("heading") or spec.get("caption") or ""
    body = spec.get("passage") or spec.get("description") or ""
    y = 0.92
    if head:
        fig.text(0.06, y, textwrap.fill(head, 60), fontsize=16, fontweight="bold", va="top", family="serif")
        y -= 0.08 * (1 + len(head) // 60)
    fig.text(0.06, y, textwrap.fill(body, 78), fontsize=12, va="top", family="serif", linespacing=1.5)


def render_spec(spec, path, key):
    rng = seeded_rng(key)
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    t = spec.get("element")
    if t == "chart":
        ax = fig.add_axes([0.12, 0.14, 0.82, 0.72])
        draw_chart(ax, spec, rng)
    elif t == "table":
        draw_table(fig, spec, rng)
    else:  # text, figure, layout: a text block
        draw_text(fig, spec)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def render_question(q, path):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    fig.text(0.06, 0.9, textwrap.fill(q, 60), fontsize=16, va="top", family="sans-serif")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


SYNTHETIC_SPECS = [  # hand-written test specs, not related to any real item
    {"element": "chart", "subtype": "grouped bar", "title": "SYNTHETIC fruit sales by region", "x_label": "Region",
     "y_label": "Units", "categories": ["North", "South", "East"], "series": ["Apples", "Pears"], "values": ""},
    {"element": "chart", "subtype": "line", "title": "SYNTHETIC rainfall", "x_label": "Month", "y_label": "mm",
     "categories": ["Jan", "Feb", "Mar", "Apr"], "series": ["2020"], "values": "10 20 15 30"},
    {"element": "chart", "subtype": "pie or donut", "title": "SYNTHETIC pets", "categories": ["Cats", "Dogs", "Fish"],
     "series": [], "values": "50 30 20"},
    {"element": "chart", "subtype": "horizontal bar", "title": "SYNTHETIC scores", "categories": ["A", "B"],
     "series": ["Score"], "values": ""},
    {"element": "chart", "subtype": "stacked bar", "title": "SYNTHETIC mix", "categories": ["X", "Y"],
     "series": ["P", "Q"], "values": ""},
    {"element": "table", "caption": "SYNTHETIC Table 1: results", "columns": ["Model", "Acc", "F1"],
     "rows": ["Alpha", "Beta"], "values": ""},
    {"element": "text", "heading": "SYNTHETIC heading", "passage": "SYNTHETIC passage. " * 12},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["specs", "questions", "test"])
    ap.add_argument("--input", default=str(IDEA4 / "proposer" / "specs.jsonl"))
    ap.add_argument("--out", default=str(IDEA4 / "renders" / "code"))
    args = ap.parse_args()
    if args.cmd == "test":
        out = IDEA4 / "tests" / "renders_SYNTHETIC"
        out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(SYNTHETIC_SPECS):
            render_spec(s, out / f"SYNTHETIC_{i}.png", i)
        print("wrote", out)
    elif args.cmd == "specs":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for line in open(args.input):
            r = json.loads(line)
            render_spec(r["spec"], out / f"{r['pid']}.png", r["pid"])
            n += 1
        print(f"wrote {n} renders to {out}")
    else:
        import pandas as pd
        out = IDEA4 / "renders" / "question"
        out.mkdir(parents=True, exist_ok=True)
        ids = pd.read_csv(IDEA4 / "proposer" / "id_map.csv")
        for r in ids.itertuples():
            render_question(r.question, out / f"{r.pid}.png")
        print(f"wrote {len(ids)} question images")


if __name__ == "__main__":
    main()
