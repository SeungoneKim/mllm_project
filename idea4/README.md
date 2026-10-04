# Idea 4, Part A: human simulation of the proposer

All commands run from this folder with the conda env `cmu`:

```
PY=~/anaconda3/envs/cmu/bin/python
```

## Steps

1. **Sample.** `$PY sample.py`
   Writes `out/sample.csv` (gold included), `out/sample_docs.txt`, and renders every page of the 37 sampled
   documents to `cache/pages/<doc_stem>/<page:04d>.jpg` (144 dpi, long side at most 1600 px, JPEG 90).
   Seed 777. 40 items (12 chart, 12 table, 8 figure, 8 text, at most 2 per document) plus 3 practice items.
   Practice items come from documents outside the real sample, so their gold pages give no hint about a real item.

2. **Content inventory (GPU).** `screen -S inventory ./run_inventory.sh`
   One process per GPU (`inventory.py run --shard k --nshards 2`), then `inventory.py aggregate`.
   Writes `out/inventory/pages.jsonl` (per page: raw output, parsed elements with boxes in [0, 1], OCR text and
   method), `out/inventory/docs.json` (per document counts) and `out/inventory/check.txt` (gold-type recall).
   The run is resumable. Pages with fewer than 20 text-layer words get OCR: tesseract when it is readable
   (at least 5 words, mean confidence 70 or more), otherwise a transcription by the same VLM.

3. **Form.** `$PY build_form.py` (or `--inventory labels` for the label-based fallback).
   Writes `annotation/items.js` and `annotation/img/<item_id>.jpg`, runs the blindness checks, and zips the
   folder as `annotation.zip`. Annotators unzip it and double-click `annotation/index.html`. Work autosaves in
   the browser; **Export** downloads `annotations_<name>_<YYYYMMDD-HHMM>.json`.

4. **Analysis.** `$PY analyze_A.py annotations_A.json annotations_B.json`
   Writes `out/partA/` (`metrics.json`, `summary.md`, `items_long.csv`, `cues.csv`, `fig_partA*.pdf`).
   Practice items are ignored. Synthetic files are refused unless `--allow-synthetic` is given; their results go
   to `tests/out_SYNTHETIC/`, never to `out/partA/`.

## Part B: machine proposer, image generation, retrieval

1. `$PY proposer.py inputs` writes blind proposer inputs (question, document type, page count, inventory) for the
   144 pool questions in the sampled documents. The proposer (Claude Opus 5.5, fresh context; see `AI_USAGE.md`)
   writes `proposer/outputs_*.json`; `$PY proposer.py collect` checks them into `proposer/specs.jsonl`.
2. `$PY render.py specs` (code drawings), `$PY render.py questions` (question as image),
   `gen_images.py` (Qwen-Image-2.1). André's step-2 specs: `proposer/human_specs.jsonl`, `renders/*_human/`.
3. `retrieve.py pages` then `retrieve.py queries` (ColQwen2.5; models in `/media/ephemeral/dre/hf` because the
   `HF_HOME` quota is full). Writes `out/partB/scores.jsonl`.
4. `$PY analyze_B.py` writes `out/partB/` (summary, metrics, ranks, figures).

## Tests

- `$PY tests/make_synthetic.py` writes two random `tests/annotations_SYNTHETIC_*.json` files (random values,
  not annotations). `$PY analyze_A.py --allow-synthetic tests/annotations_SYNTHETIC_*.json` runs the full analysis.
- `build_form.py` checks that `items.js` parses, every image exists, gold fields decode, items hold only the
  allowed fields, and no document name, answer or evidence label appears in plain text in `annotation/`.
- The form was not tested in a browser (no browser on the cluster). Its scripts were syntax-checked with `esprima`.

## Models and downloads

| Model | Use | Size | Location |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-7B-Instruct` | Step 2 inventory and OCR fallback | 16.6 GB (bf16) | `$HF_HOME/hub` |
| `vidore/colqwen2.5-v0.2` + `vidore/colqwen2.5-base` | Part B retriever | 0.24 GB + 7.5 GB | `/media/ephemeral/dre/hf` |
| `Qwen/Qwen-Image-2.1` | Part B image generation (already on the cluster) | 31 GB | `$HF_HOME/hub` |

Run with Hugging Face transformers 5.18 (greedy decoding, images resized to at most 1 megapixel, sides multiples
of 28). vLLM 0.10.1.1 in `cmu` fails to load the model with transformers 5.18 (a rope config conflict), so it
was not used. Python packages added to `cmu`: `pymupdf`, `pytesseract`, `esprima` (form syntax check).

## Notes

- 9 questions list an evidence page outside the PDF; they are excluded from the pool.
- The inventory has no "layout" type. In `check.txt`, layout gold counts as found when text or other is listed.
- Answers of 4 items appear in their own question (multiple-choice questions). This is visible by design.
