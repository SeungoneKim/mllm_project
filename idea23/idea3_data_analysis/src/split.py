"""Split a page's evidence along a seam that already exists in the content.

Instead of forcing every item into one mould (replace the label with an opaque
token), the model first inventories what on the page contributes to the answer,
then proposes where that evidence naturally comes apart. Two kinds of seam:

  partition    each part carries a different portion of the answer
               (a three-item list divided two and one)
  composition  one part cannot be interpreted without the other
               (a value, and the thing that says what the value measures)

One part is then re-rendered in the opposite modality, so the pair is never
readable by a single-modality retriever.
"""
import argparse, collections, json, pathlib, re, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import llm, render, codegen_render, pageio

ROOT = pathlib.Path(__file__).resolve().parent.parent

OPEN_MODEL = "wine-qwen3-vl-235b-a22b"
CLOSED_MODEL = "wine-gemini-3.1-pro-preview"

INVENTORY = """You are looking at page {pg} of a document, and a question that this page
answers.

QUESTION: {q}
GOLD ANSWER: {a}

List every distinct piece of information ON THIS PAGE that contributes to that answer.
A piece is distinct if a reader could point at it separately: one bar of a chart with
its label, one row of a table, one sentence, one labelled part of a figure, a caption.
Do not list page furniture that contributes nothing.

For each piece give:
  "id"         a short handle, e.g. "bar_effective"
  "modality"   one of chart / table / figure / text
  "content"    its content, transcribed faithfully - numbers exactly as printed
  "role"       what it contributes to the answer, in a few words

Then say whether the pieces you listed, taken together, fully determine the gold answer.

JSON keys: items, complete (true/false)"""

SPLIT = """You are preparing a controlled item for a retrieval study.

QUESTION: {q}
GOLD ANSWER: {a}
EVIDENCE ON THE PAGE (JSON): {inv}

Divide this evidence into exactly two parts, A and B, so that:
  - neither part ALONE lets a reader determine the gold answer. Test this honestly:
    read each part as if it were all you had, and ask whether the gold answer could be
    written down from it. If it could, the split has failed;
  - the two parts TOGETHER do;
  - the division follows a seam that is already in the content. Do not invent facts,
    do not anonymise anything, and do not add placeholder labels. Every word in each
    part must come from the evidence above.

Name the kind of seam you used:
  "partition"   - each part carries a different portion of the answer, and the answer
                  needs all portions (a list of three split two and one; a total that
                  needs two addends).
  "composition" - the part holding the answer ALSO holds at least two other items of
                  the same kind that could plausibly answer the question, so that part
                  alone cannot say which of them is the one asked for; the other part
                  does nothing but select among them.
                  A split that puts the answer on one side and its heading, number or
                  caption on the other is NOT composition: the heading constrains
                  nothing, and the answer still reads straight off its own side.
                  If the page offers no competing alternatives of the same kind -
                  a lone list, a single labelled region, one sentence - then
                  composition is impossible here. Use "partition" instead if the answer
                  has parts that can be divided, and otherwise set splittable to false.

Each part must end up in a DIFFERENT modality, so give each a target:
  "text"   -> the part is written as a sentence or short passage
  "visual" -> the part is drawn. Say which form it takes in "form", and supply
              whichever field that form needs:

      "form":"chart"  -> "spec": {{"kind":"bar"|"barh"|"line"|"pie","title":str,
                          "subtitle":str|null,"categories":[str,...],
                          "values":[num,...],"value_fmt":str}}
      "form":"table"  -> "spec": {{"kind":"table","title":str,"columns":[str,...],
                          "rows":[[str,...],...]}}
      "form":"diagram" -> "description": a precise account of what must be drawn and
                          how the parts are arranged (overlapping sets and what sits
                          in the overlap; boxes and the arrows between them; a
                          labelled schematic and where each label attaches).
                          Choose this ONLY when the original evidence is a diagram
                          whose meaning is carried by its arrangement, so that a
                          table of its words would lose what it shows.

If the evidence cannot be divided this way - because it is a single atomic fact that
answers the question on its own - set "splittable" to false and say why.

JSON keys: splittable, reason, split_kind, seam (one sentence naming the seam),
part_a {{target, text, spec}}, part_b {{target, text, spec}}"""


def part_is_empty(part):
    """A part the model left blank. Rendering one silently produces a white box."""
    if part.get("target") == "visual":
        if part.get("form") == "diagram":
            return len(str(part.get("description") or "").strip()) < 10
        sp = part.get("spec") or {}
        if sp.get("kind") == "table":
            return not (sp.get("rows") and any(any(str(c).strip() for c in r)
                                               for r in sp["rows"]))
        if sp.get("kind") == "textimg":
            return not str(sp.get("body") or "").strip()
        return not (sp.get("categories") and sp.get("values"))
    return len(str(part.get("text") or "").strip()) < 2


def _norm_words(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def provenance(part, page_text):
    """Fraction of the part's content words that occur in the page's own text.

    The split is supposed to redistribute what the page says, not paraphrase it.
    A low score means the model wrote new prose, which changes what the item
    tests. Low-scoring parts are kept but flagged, since a faithful transcription
    of a figure legitimately has no matching page text.
    """
    if part.get("target") == "visual":
        if part.get("form") == "diagram":
            return len(str(part.get("description") or "").strip()) < 10
        sp = part.get("spec") or {}
        blob = " ".join(str(x) for x in
                        (sp.get("categories") or []) + [sp.get("title") or ""]
                        + [c for r in (sp.get("rows") or []) for c in r])
    else:
        blob = str(part.get("text") or "")
    words = [w for w in _norm_words(blob).split() if len(w) > 2]
    page = _norm_words(page_text)
    if not words or len(page) < 50:
        # Slides and scanned pages carry no text layer, so there is nothing to
        # check the part against. That is undefined, not zero.
        return None
    return round(sum(1 for w in words if w in page) / len(words), 3)


def _render_part(part, d, name, model=None):
    """Write a part to disk in its target modality.

    Charts and tables go through the fixed schema renderer, so every one of them
    is drawn identically and the drawing is not a variable across items. Only a
    diagram - whose meaning is in its arrangement, and which the schema cannot
    express at all - is drawn by model-written code.

    -> (filename, modality, how)
    """
    if part.get("target") == "visual":
        if part.get("form") == "diagram" and part.get("description") and model:
            p = d / f"{name}.png"
            res = codegen_render.draw_part(part["description"],
                                           "a diagram from a document", p, model)
            if res["ok"]:
                (d / f"{name}_draw.py").write_text(res["code"])
                return p.name, "image", f"codegen({res['attempts']})"
            # fall back to the schema renderer rather than drop the part
            if part.get("spec"):
                p.write_bytes(render.render_element(part["spec"]))
                return p.name, "image", "schema(codegen failed)"
            txt = part.get("description") or ""
            (d / f"{name}.txt").write_text(str(txt))
            return f"{name}.txt", "text", "text(codegen failed, no spec)"
        if part.get("spec"):
            p = d / f"{name}.png"
            p.write_bytes(render.render_element(part["spec"]))
            return p.name, "image", "schema"
    txt = part.get("text") or part.get("description") or json.dumps(part.get("spec", ""))
    p = d / f"{name}.txt"
    p.write_text(str(txt))
    return p.name, "text", "text"


def split_seed(seed, model, outdir, dpi=200):
    sid = seed["sid"]
    d = pathlib.Path(outdir) / sid
    d.mkdir(parents=True, exist_ok=True)
    rec = {"sid": sid, "model": model, "doc_id": seed["doc_id"],
           "question": seed["question"], "answer": seed["answer"],
           "modality": seed["modality"], "evidence_page": seed["ev_pages"][0],
           "pieces": {}}

    pg1 = seed["ev_pages"][0]
    page_png = pageio.page_png(seed["doc_id"], pg1)
    pageio.page_jpeg(seed["doc_id"], pg1, d / "page.jpg")

    inv = llm.ask_json(INVENTORY.format(pg=seed["ev_pages"][0], q=seed["question"],
                                        a=seed["answer"]), [page_png], model=model)
    rec["inventory"] = inv
    rec["n_items"] = len(inv.get("items", []))
    rec["inventory_complete"] = bool(inv.get("complete"))

    sp = llm.ask_json(SPLIT.format(q=seed["question"], a=seed["answer"],
                                   inv=json.dumps(inv)[:5000]), model=model)
    rec["split"] = sp
    if not sp.get("splittable", True):
        rec["skipped"] = sp.get("reason") or "not splittable"
        return rec
    a_, b_ = sp.get("part_a") or {}, sp.get("part_b") or {}
    if not a_ or not b_:
        rec["skipped"] = "model returned no parts"
        return rec
    blank = [n for n, x in (("part_a", a_), ("part_b", b_)) if part_is_empty(x)]
    if blank:
        rec["skipped"] = f"model left {' and '.join(blank)} empty"
        return rec
    if a_.get("target") == b_.get("target"):
        b_["target"] = "text" if a_.get("target") == "visual" else "visual"
        rec["forced_opposite_modality"] = True

    rec["split_kind"] = sp.get("split_kind")
    rec["seam"] = sp.get("seam")
    fa, ma, ha = _render_part(a_, d, "part_a", model)
    fb, mb, hb = _render_part(b_, d, "part_b", model)
    rec["pieces"] = {"part_a": fa, "part_b": fb}
    rec["modalities"] = {"part_a": ma, "part_b": mb}
    rec["render_method"] = {"part_a": ha, "part_b": hb}
    rec["forms"] = {"part_a": a_.get("form"), "part_b": b_.get("form")}
    rec["cross_modal"] = ma != mb
    page_text = pageio.page_text(seed["doc_id"], pg1)
    rec["provenance"] = {"part_a": provenance(a_, page_text),
                         "part_b": provenance(b_, page_text)}
    got = [v for v in rec["provenance"].values() if v is not None]
    rec["min_provenance"] = min(got) if got else None
    rec["page_has_text_layer"] = len(page_text.strip()) >= 50
    return rec


def run(seeds, model, outdir, workers=4):
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    out = []

    def one(s):
        try:
            return split_seed(s, model, outdir)
        except Exception as e:
            return {"sid": s["sid"], "model": model, "status": "error",
                    "error": repr(e), "trace": traceback.format_exc()[-800:],
                    "question": s["question"], "answer": s["answer"],
                    "modality": s["modality"], "doc_id": s["doc_id"]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed({ex.submit(one, s): s for s in seeds}):
            r = f.result(); out.append(r)
            how = "+".join(str(v) for v in (r.get("render_method") or {}).values())
            print(f"  [{len(out):>3}/{len(seeds)}] {r['sid']} "
                  f"items={r.get('n_items','-')} kind={str(r.get('split_kind'))[:11]:11s} "
                  f"{how:22s} {r.get('skipped') or r.get('error','')}"[:150], flush=True)
    out.sort(key=lambda r: r["sid"])
    (outdir / "records.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def summarise(recs):
    ok = [r for r in recs if r.get("status") != "error"]
    built = [r for r in ok if r.get("pieces")]
    kinds = collections.Counter(r.get("split_kind") for r in built)
    return {"n": len(recs), "errors": len(recs) - len(ok),
            "not_splittable": sum(1 for r in ok if r.get("skipped")),
            "built": len(built),
            "checkable": sum(1 for r in built if r.get("min_provenance") is not None),
            "low_provenance": sum(1 for r in built
                                  if r.get("min_provenance") is not None
                                  and r["min_provenance"] < 0.5),
            "partition": kinds.get("partition", 0),
            "composition": kinds.get("composition", 0),
            "cross_modal": sum(1 for r in built if r.get("cross_modal")),
            "diagram_codegen": sum(1 for r in built
                                   if any(str(v).startswith("codegen")
                                          for v in (r.get("render_method") or {}).values()))}


if __name__ == "__main__":
    import seeds as seedmod
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--models", default=f"{OPEN_MODEL},{CLOSED_MODEL}")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    df = seedmod.load()
    chosen = seedmod.stratified(df, a.n, seed=a.seed)
    print(f"{len(df)} eligible -> {len(chosen)} seeds\n")
    root = pathlib.Path(a.out); root.mkdir(parents=True, exist_ok=True)
    rep = {"n_seeds": len(chosen), "models": {}}
    for m in a.models.split(","):
        print(f"=== {m} ===", flush=True)
        recs = run(chosen, m, root / m.replace("/", "_"), a.workers)
        rep["models"][m] = summarise(recs)
        print("   ", rep["models"][m], flush=True)
    (root / "comparison.json").write_text(json.dumps(rep, indent=1))
    print("\n" + "=" * 92)
    cols = ["built", "partition", "composition", "diagram_codegen",
            "cross_modal", "checkable", "low_provenance", "not_splittable"]
    print(f"{'model':32s} " + " ".join(f"{c[:12]:>13s}" for c in cols))
    for m, sm in rep["models"].items():
        print(f"{m:32s} " + " ".join(f"{sm[c]:>13}" for c in cols))
