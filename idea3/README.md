# Analysis 3: synthetic complementary cases (Ideas 2 + 3)

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

