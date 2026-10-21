"""Draw a split part by having a VLM write the Matplotlib code for it.

The fixed schema in `render.py` can draw a chart or a table and nothing else, so
a Venn diagram, a flowchart or a labelled schematic loses its visual structure
and survives only as a table of the words it contained. Here a third model is
asked for the drawing code instead, which lifts that restriction.

The code is model-written, so it is never executed in this process: it runs in a
subprocess, in a scratch directory, under a timeout, with a guard that rejects
file, process and network access before it is run at all.
"""
import pathlib, re, subprocess, sys, tempfile, textwrap
import llm

BANNED = re.compile(
    r"\b(?:os|sys|subprocess|shutil|socket|requests|urllib|http|pathlib|glob|"
    r"importlib|ctypes|pickle|marshal|builtins|eval|exec|compile|open|__import__|"
    r"input|exit|quit)\b")

PROMPT = """Write Python that draws ONE piece of document evidence as a figure.

WHAT IT MUST SHOW (reproduce exactly; invent nothing, add nothing):
{content}

SHAPE OF THE ORIGINAL: {hint}

Rules:
- Define exactly one function: `def draw(fig):` taking a Matplotlib Figure and
  drawing into it. Do not create the figure, do not call savefig, do not call show.
- The figure is {w:.2f} x {h:.2f} inches. Keep every mark and label inside it.
- You may use `matplotlib` (pyplot, patches, lines, path, transforms) and `numpy`
  and `math`. Import nothing else. Do not touch files, processes or the network.
- Draw the structure, not a table of words: if the original is a Venn diagram draw
  overlapping shapes, if it is a flow draw boxes and arrows, if it is a schematic
  draw the parts in their arrangement.
- Plain, printable styling: white background, thin dark strokes, readable labels at
  8-10pt, no title unless the content gives one, no legend unless it is needed.
- Show ONLY the content above. Anything the content does not state must not appear.

Return ONLY the code, no prose and no code fences."""

HEAD = ("import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n\n")


def _script(code, w, h, out):
    # built by concatenation, never str.format: generated code contains braces
    # (dicts, f-strings, format specs) that a format template would mangle
    tail = (f"\n\nfig = plt.figure(figsize=({w}, {h}), dpi=200)\n"
            "fig.patch.set_facecolor('white')\n"
            "draw(fig)\n"
            f"fig.savefig({str(out)!r}, facecolor='white')\n")
    return HEAD + code + tail


def _clean(t):
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1)
    return t.strip()


def check(code):
    """Reasons the code must not be run. Empty list means it may be."""
    bad = sorted(set(BANNED.findall(code)))
    out = [f"uses {', '.join(bad)}"] if bad else []
    if "def draw(" not in code:
        out.append("does not define draw(fig)")
    if "savefig" in code or "plt.show" in code:
        out.append("calls savefig or show")
    return out


def run(code, out_path, size_in, timeout=40):
    """Execute in a subprocess. -> (ok, stderr)."""
    w, h = size_in
    out_path = pathlib.Path(out_path).resolve()      # the child runs elsewhere
    out_path.parent.mkdir(parents=True, exist_ok=True)
    script = _script(code, w, h, out_path)
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "draw.py"
        f.write_text(script)
        try:
            p = subprocess.run([sys.executable, str(f)], cwd=d, timeout=timeout,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout}s"
    if p.returncode != 0:
        return False, (p.stderr or "")[-900:]
    return pathlib.Path(out_path).exists(), (p.stderr or "")[-400:]


def draw_part(content, hint, out_path, model, size_in=(3.1, 2.2), attempts=3):
    """Ask for drawing code, run it, and feed failures back. -> dict."""
    w, h = size_in
    prompt = PROMPT.format(content=content, hint=hint or "unspecified", w=w, h=h)
    log = []
    for i in range(attempts):
        code = _clean(llm.ask(prompt, model=model, use_cache=(i == 0)))
        refused = check(code)
        if refused:
            log.append({"attempt": i + 1, "rejected": refused})
            prompt = (PROMPT.format(content=content, hint=hint or "unspecified",
                                    w=w, h=h)
                      + "\n\nYour previous answer was rejected because it "
                      + "; ".join(refused) + ". Fix it.")
            continue
        ok, err = run(code, out_path, size_in)
        log.append({"attempt": i + 1, "ran": ok, "error": err if not ok else None})
        if ok:
            return {"ok": True, "code": code, "attempts": i + 1, "log": log}
        prompt = (PROMPT.format(content=content, hint=hint or "unspecified", w=w, h=h)
                  + "\n\nYour previous code failed with:\n" + err
                  + "\nReturn corrected code.")
    return {"ok": False, "code": code, "attempts": attempts, "log": log}
