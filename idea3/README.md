# idea3 — splitting a document's evidence into complementary pieces

Takes a MMLongBench-Doc question whose evidence sits on one page, finds where that
evidence naturally comes apart, and re-expresses one half in the other modality.
The result is a pair of pieces, written to disk separately, where neither piece is
meant to answer the question on its own.

```
seed ─▶ inventory ─▶ split ─▶ render
        (VLM)        (VLM)    (fixed schema, or VLM-written code for diagrams)
```

## What it produces

```
out/<model>/S010/
  page.jpg          the evidence page, for reference
  part_a.png        one half, drawn
  part_a_draw.py    the drawing code, when a diagram was generated
  part_b.txt        the other half, as text
out/<model>/records.json   inventory, split, render method, provenance
out/comparison.json        per-model summary
```

## The two seams

A split is only interesting if the halves genuinely need each other, and there are
two ways that happens. The pipeline labels which one it used.

**Partition** — each half carries a different portion of the answer. A list of
three fonts divided two and one; a difference split into its two operands. A
retriever has to *find both*.

**Composition** — the half holding the answer also holds other candidates of the
same kind, so it cannot say which one is being asked for, and the other half does
nothing but select among them. A retriever has to *connect* them.

Composition needs that second clause. Without it, models reliably split the answer
from its *heading* — the three bad fonts on one side, the words "BAD FONTS" on the
other — and the answer still reads straight off its own side. A heading constrains
nothing.

## Rendering: fixed schema, except for diagrams

A visual half is **not** produced by an image model. Every value here is exact —
`$714.3 million`, a cross in a table cell — and a model that draws pixels cannot be
relied on to reproduce them; a chart whose bars disagree with its labels would
corrupt the gold answer silently.

So the split step declares a `form` and supplies structure, not pixels:

| `form` | how it is drawn | why |
|---|---|---|
| `chart` | fixed Matplotlib renderer (`render.py`) | identical styling across every item |
| `table` | fixed Matplotlib renderer | same |
| `diagram` | Matplotlib code written by a third model (`codegen_render.py`) | the schema has no way to express an arrangement |

The first two keep the drawing out of the experiment: styling is fixed in the
renderer, so a difference between items cannot be blamed on how they were drawn. A
diagram is the exception, because its meaning *is* its arrangement — a Venn diagram
flattened into a table of its words loses what it shows. Each part records which
path it took in `render_method`, so generated-code items can be separated later.

### Running model-written code

The drawing code is never executed in this process. It is checked for file, process
and network access and rejected before it runs; it must define exactly `draw(fig)`
and may not save or show; and it runs in a subprocess, in a scratch directory, under
a timeout. A failure is fed back with its traceback for up to three attempts, then
falls back to the schema renderer rather than dropping the part.

## Guards on the construction

- **Empty parts are rejected**, with the reason kept. A model that returns
  `{"kind":"table"}` with no rows has not produced a part, and rendering one
  silently yields a white box.
- **Provenance** scores how much of a half's wording appears in the page's own text,
  to catch halves the model wrote rather than relocated. It is `None`, not zero, for
  slides and scans, which carry no text layer to check against. It is recorded, not
  enforced: a faithful transcription of a figure legitimately has no matching page
  text.

## Layout

```
src/llm.py             gateway client, disk-cached, with JSON repair
src/seeds.py           seed selection from MMLongBench-Doc
src/pageio.py          page images and text
src/split.py           inventory + split + render        (entry point)
src/render.py          fixed-schema chart / table / prose renderer
src/codegen_render.py  model-written Matplotlib, sandboxed
```

## Use

```bash
PY=/data/user_data/eunsukim/envs/mmaug/bin/python
$PY src/split.py --n 24 --workers 3
```

`--models` takes a comma-separated list of gateway aliases; the defaults are
`wine-qwen3-vl-235b-a22b` (open) and `wine-gemini-3.1-pro-preview` (closed).

## Credentials and data

`~/.env` supplies `LITELLM_API_KEY` and `API_BASE` for the gateway. `data/`
symlinks the MMLongBench-Doc PDFs and QA parquet from
[`yubo2333/MMLongBench-Doc`](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc);
note its `evidence_pages` is 1-indexed, which this code uses throughout.

## Known limits

- **Nothing here checks that the split worked.** The pipeline builds pieces and
  records how it built them; whether a half can answer the question on its own is
  not measured. That check lives outside this repo.
- **Composition is rare.** It needs the page to already contain competing
  alternatives. Sampling across modalities yields roughly one per two dozen seeds;
  seeding from numeric answers in multi-row tables and charts would raise that.
- **A figure is only reproduced as a figure when the split step says `diagram`.**
  Otherwise its text survives and its visual structure does not.
- **The split step decides four things in one response** — where to cut, what each
  half contains, which modality each takes, and what form the visual takes. Yield
  dropped when `form` was added, which suggests the call is carrying too much;
  separating the cut from the conversion is the obvious next change.
