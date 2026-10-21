"""Render synthetic document elements (charts, tables, prose) as PNG.

Layout is computed manually in inches so that, when `size_in` is supplied,
matplotlib point sizes map 1:1 onto PDF points: an element stamped into a box
of exactly that size is as legible as native page content.
"""
import io, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2f6f9f", "#c0632b", "#6c8e3f", "#8a5a9b", "#b03a48", "#4a7b8c"]
GRID = "#d9d9d9"; INK = "#1f1f1f"; MUTED = "#5a5a5a"
DPI = 260
M = 0.10                      # outer margin, inches


def _fonts(w_in):
    s = max(0.66, min(1.0, w_in / 6.2))
    return dict(title=round(11.0 * s, 1), tick=round(8.8 * s, 1),
                val=round(8.8 * s, 1), sub=round(8.0 * s, 1),
                note=round(6.4 * s, 1))


def _wrap(text, width_in, fontsize, factor=0.50):
    """Wrap to the number of characters that fit `width_in` at `fontsize`."""
    n = max(14, int((width_in * 72.0) / (factor * fontsize)))
    return textwrap.wrap(str(text).replace("\n", " "), n)


def _header(fig, title, subtitle, W, F, y_top):
    """Draw title/subtitle at the top; return the y (inches from bottom) below it."""
    H = fig.get_figheight()
    y = y_top
    for txt, fs, bold, col in ((title, F["title"], True, INK),
                               (subtitle, F["sub"], False, MUTED)):
        if not txt:
            continue
        lines = _wrap(txt, W, fs, 0.49 if bold else 0.46)
        lh = fs * 1.22 / 72.0
        for ln in lines:
            y -= lh
            fig.text(M / fig.get_figwidth(), y / H, ln, fontsize=fs, color=col,
                     fontweight="bold" if bold else "normal", va="baseline", ha="left")
        y -= lh * 0.28
    return y


def _footer(fig, note, W, F):
    if not note:
        return 0.0
    lines = _wrap(note, W, F["note"], 0.46)[:5]
    lh = F["note"] * 1.25 / 72.0
    H = fig.get_figheight()
    y = M
    for ln in reversed(lines):
        fig.text(M / fig.get_figwidth(), y / H, ln, fontsize=F["note"],
                 color=MUTED, va="baseline", ha="left")
        y += lh
    return (y - M) + lh * 0.3


def _style(ax, F):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a"); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=F["tick"], length=3)
    ax.set_axisbelow(True)


def norm_spec(spec, assume=None):
    """Repair the shapes a VLM transcription can get wrong.

    `assume` forces a kind; otherwise it is taken from the spec, and inferred
    from which fields are present when the spec does not name one.
    """
    spec = dict(spec or {})
    k = assume or spec.get("kind")
    if k not in ("bar", "barh", "line", "pie", "table", "textimg"):
        if spec.get("columns") or spec.get("rows"):
            k = "table"
        elif spec.get("categories") or spec.get("values"):
            k = "bar"
        elif spec.get("body") is not None:
            k = "textimg"
        else:
            k = "bar"
    spec["kind"] = k
    if k == "table":
        rows = [list(r) if isinstance(r, (list, tuple)) else [r]
                for r in (spec.get("rows") or [])]
        cols = [str(c) for c in (spec.get("columns") or [])]
        n = max([len(cols)] + [len(r) for r in rows] or [1])
        if len(cols) < n:                       # unlabelled leading column etc.
            cols = cols + [""] * (n - len(cols))
        spec["columns"] = cols[:n]
        spec["rows"] = [[str(c) for c in (r + [""] * (n - len(r)))[:n]] for r in rows]
    elif k in ("bar", "barh", "line", "pie"):
        cats = [str(c) for c in (spec.get("categories") or [])]
        vals = spec.get("values") or []
        if vals and all(isinstance(v, (list, tuple)) for v in vals):
            # multi-series transcription: keep the series the question needs
            names = spec.get("series_labels") or spec.get("series") or []
            idx = 0
            spec["series_note"] = str(names[idx]) if idx < len(names) else None
            vals = list(vals[idx])
        out = []
        for v in vals:
            if isinstance(v, (list, tuple)):
                v = v[0] if v else 0
            try:
                out.append(float(str(v).replace(",", "").replace("%", "").strip() or 0))
            except Exception:
                out.append(0.0)
        n = min(len(cats), len(out)) if cats and out else 0
        spec["categories"], spec["values"] = cats[:n], out[:n]
        if n == 0:
            spec["kind"] = "textimg"
            spec["body"] = spec.get("body") or spec.get("title") or ""
    return spec


def _fmt(v, f):
    try: return f.format(v)
    except Exception: return str(v)


def _save(fig, size_in, fmt="png"):
    """Serialize the figure. PDF keeps the element vector, which matters when the
    piece is placed in a paper rather than viewed as a thumbnail."""
    buf = io.BytesIO()
    kw = {"facecolor": "white"} if fmt == "png" else {}
    if size_in:                            # keep the exact physical size
        fig.savefig(buf, format=fmt, **kw)
    else:
        fig.savefig(buf, format=fmt, bbox_inches="tight", pad_inches=0.12, **kw)
    plt.close(fig)
    return buf.getvalue()


def _canvas(size_in, default):
    w, h = size_in or default
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig, w, h


def render_chart(spec, size_in=None, fmt="png"):
    spec = norm_spec(spec)
    kind = spec["kind"]
    if kind == "table":
        return render_table(spec, size_in)
    if kind == "textimg":
        return render_text_image(spec.get("title"), spec.get("body", ""), size_in)
    cats = [str(c) for c in spec.get("categories", [])]
    vals = [float(v) for v in spec.get("values", [])]
    vf = spec.get("value_fmt", "{:g}")
    hl = spec.get("highlight")
    show = spec.get("show_values", True)

    fig, W, H = _canvas(size_in, (6.4, 3.1))
    F = _fonts(W)
    avail_w = W - 2 * M
    y_top = _header(fig, spec.get("title"), spec.get("subtitle"), avail_w, F, H - M)
    y_bot = M + _footer(fig, spec.get("note"), avail_w, F)

    tall = H / W > 1.25
    narrow = W < 3.4
    avail_h = max(0.4, y_top - y_bot)
    legend_h = 0.0
    if kind == "pie":
        # square plot pinned under the header; legend (if any) sits beneath it
        if narrow:
            legend_h = min(avail_h * 0.42, len(cats) * F["tick"] * 1.75 / 72 + 0.10)
        d = max(0.5, min(avail_w, avail_h - legend_h))
        ax_rect = [(M + (avail_w - d) / 2) / W, (y_top - d) / H, d / W, d / H]
    else:
        lpad, bpad = 0.60, 0.42
        if kind == "barh" or (kind == "bar" and tall):
            lpad = min(max(1.2, avail_w * 0.45),
                       0.14 + 0.050 * F["tick"] * max((len(c) for c in cats), default=6))
        ax_rect = [(M + lpad) / W, (y_bot + bpad) / H,
                   max(0.12, (avail_w - lpad - 0.10) / W),
                   max(0.12, (avail_h - bpad - 0.06) / H)]
    ax = fig.add_axes(ax_rect)

    colors = [PALETTE[0]] * len(cats)
    if hl:
        for i, c in enumerate(cats):
            if str(hl).lower() in c.lower():
                colors[i] = PALETTE[1]

    if kind == "pie":
        labels = [_fmt(v, vf) for v in vals]          # exact values, never autopct
        if narrow:
            w_, _ = ax.pie(vals, colors=PALETTE[:len(cats)] * 3, startangle=90,
                           wedgeprops={"edgecolor": "white", "linewidth": 1.1})
            fig.legend(w_, [f"{c} \u2013 {l}" for c, l in zip(cats, labels)],
                       loc="upper left", fontsize=F["tick"], frameon=False,
                       handlelength=1.0, handletextpad=0.5, borderaxespad=0,
                       labelspacing=0.45,
                       bbox_to_anchor=(M / W, (y_bot + legend_h) / H))
        else:
            ax.pie(vals, labels=[f"{c}\n{l}" for c, l in zip(cats, labels)],
                   colors=PALETTE[:len(cats)] * 3, startangle=90, labeldistance=1.12,
                   textprops={"fontsize": F["tick"], "color": INK},
                   wedgeprops={"edgecolor": "white", "linewidth": 1.1})
        ax.axis("equal")
    elif kind == "barh" or (kind == "bar" and tall):
        ax.barh(cats[::-1], vals[::-1], color=colors[::-1], height=0.60)
        ax.xaxis.grid(True, color=GRID, lw=0.7)
        if show and vals:
            for i, v in enumerate(vals[::-1]):
                ax.text(v, i, " " + _fmt(v, vf), va="center", ha="left",
                        fontsize=F["val"], color=INK)
            ax.set_xlim(0, max(vals) * 1.26)
        _style(ax, F)
    elif kind == "line":
        ax.plot(cats, vals, marker="o", ms=3.5, lw=1.7, color=PALETTE[0])
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        if show:
            for x, v in zip(cats, vals):
                ax.annotate(_fmt(v, vf), (x, v), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=F["val"], color=INK)
        _style(ax, F)
        if len(cats) > 5:
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    else:
        ax.bar(cats, vals, color=colors, width=0.60)
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        if show and vals:
            for x, v in zip(cats, vals):
                ax.annotate(_fmt(v, vf), (x, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=F["val"], color=INK)
            ax.set_ylim(0, max(vals) * 1.20)
        _style(ax, F)
        if len(cats) > 4 or max((len(c) for c in cats), default=0) > 10:
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    if kind != "pie" and spec.get("ylabel"):
        ax.set_ylabel(str(spec["ylabel"])[:60], fontsize=F["sub"], color=MUTED)

    return _save(fig, size_in, fmt)


def render_table(spec, size_in=None, fmt="png"):
    spec = norm_spec(spec, assume="table")
    cols = [str(c)[:28] for c in spec.get("columns", [])]
    rows = [[str(c)[:28] for c in r] for r in spec.get("rows", [])]
    if not rows and not cols:          # nothing transcribable: fall back to prose
        return render_text_image(spec.get("title"), str(spec.get("note") or ""), size_in)
    if not cols:
        cols = [""] * len(rows[0])
    nr, nc = len(rows), max(1, len(cols))
    fig, W, H = _canvas(size_in, (min(7.2, 1.6 + 1.35 * nc), 0.40 * (nr + 2) + 0.6))
    F = _fonts(W)
    avail_w = W - 2 * M
    y_top = _header(fig, spec.get("title"), spec.get("subtitle"), avail_w, F, H - M)
    y_bot = M + _footer(fig, spec.get("note"), avail_w, F)
    ax = fig.add_axes([M / W, y_bot / H, avail_w / W, max(0.1, (y_top - y_bot) / H)])
    ax.axis("off")
    tb = ax.table(cellText=rows or [[""] * nc], colLabels=cols, loc="center",
                  cellLoc="center", bbox=[0, 0, 1, 1])
    longest = max([len(c) for c in cols] + [len(c) for r in rows for c in r] + [1])
    fs = F["tick"] * min(1.0, 9.5 / max(nc, 1)) * min(1.0, 11.0 / longest)
    tb.auto_set_font_size(False); tb.set_fontsize(max(3.6, round(fs, 1)))
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c4c4c4"); cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#eef2f5"); cell.set_text_props(weight="bold", color=INK)
        elif r % 2 == 0:
            cell.set_facecolor("#fafafa")
    return _save(fig, size_in, fmt)


def render_text_image(title, body, size_in=None, fmt="png"):
    """Prose rendered as a PICTURE of text - invisible to text-only retrievers."""
    probe = _fonts((size_in or (6.2, 1))[0])
    if size_in is None:
        lines = _wrap(body or "", 6.2 - 2 * M, probe["tick"], 0.47)
        h = 2 * M + (probe["title"] * 1.6 / 72 if title else 0) + \
            len(lines) * probe["tick"] * 1.5 / 72 + 0.08
        size_in = (6.2, max(0.5, h))
    fig, W, H = _canvas(size_in, (6.2, 1.0))
    F = _fonts(W)
    avail = W - 2 * M - 0.08
    y = H - M
    if title:
        for ln in _wrap(title, avail, F["title"], 0.49):
            y -= F["title"] * 1.25 / 72
            fig.text((M + 0.08) / W, y / H, ln, fontsize=F["title"],
                     fontweight="bold", color=INK, va="baseline")
        y -= F["title"] * 0.45 / 72
    for ln in _wrap(body or "", avail, F["tick"], 0.47):
        y -= F["tick"] * 1.45 / 72
        if y < M * 0.4:
            break
        fig.text((M + 0.08) / W, y / H, ln, fontsize=F["tick"], color=INK, va="baseline")
    fig.patches.append(plt.Rectangle((M / W, 0.04), 0.008, 0.92,
                                     transform=fig.transFigure,
                                     facecolor=PALETTE[0], edgecolor="none"))
    return _save(fig, size_in, fmt)


def render_element(spec, size_in=None, fmt="png"):
    spec = norm_spec(spec)
    k = spec.get("kind", "bar")
    if k == "table": return render_table(spec, size_in, fmt)
    if k == "textimg":
        return render_text_image(spec.get("title"), spec.get("body", ""), size_in, fmt)
    return render_chart(spec, size_in, fmt)
