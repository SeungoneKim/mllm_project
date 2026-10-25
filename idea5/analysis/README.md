# Idea 5: what does the page image add beyond the text layer? (feasibility analysis)

Idea 5 represents each page by its text layer, given exactly, plus a small set of visual tokens conditioned on that
text and trained to carry only what the text lacks (charts, structure, figures, layout). This analysis tests its
premises with no training, on MMLongBench-Doc, using the gold evidence page and no retrieval.

- **Part A (CPU, all 829 eligible questions):** is the evidence in the gold pages' text layer at all?
- **Part B (GPU, 72 sampled questions):** Qwen2.5-VL-7B answers each question from the gold page under 9 input
  conditions (question only, text layer, its own transcription, image at 1280 or 256 tokens, text + image, an image
  with every text-layer word painted white, alone and with the text).

## Hypotheses (fixed before running)

- **H1.** Text alone (the text layer, or the VLM's own markdown transcription) falls short of text + image on chart,
  figure and layout questions, but not on text questions.
- **H2.** The interaction differs by evidence type: text questions are mostly *equivalence* (either modality alone is
  right), chart and figure questions *image-dominant*, and tables show *emergence* (only both together are right);
  *interference* (a single modality is right, the combination is wrong) also occurs. Regression per evidence type,
  `y = w0 + w1 I + w2 T + w3 I*T` over the cells question only, image, text, text + image.
- **H3.** *Modulation*: lowering the image from 1280 to 256 tokens costs less when the text layer is present. *Residual*:
  text + words-masked image is as accurate as text + image at the same 256-token budget.

## Steps

| step | command | where | output |
|---|---|---|---|
| A | `python coverage.py` | CPU, about 20 s | `out/coverage/` summary, questions.csv, figure |
| B1 | `python sample.py` | CPU | `out/sample/items.jsonl`, `cache/sample/<qid>_{hi,lo,mlo}.png` |
| B2 | `python reader.py --shard k --nshards n` (or `job_abci.sh`) | GPU | `out/reader/{transcripts,answers}_<k>.jsonl` |
| B3 | `python analyze.py` | CPU | `out/analysis/` summary, metrics, scored answers, 2 figures |

`python score.py` runs the scorer's self-checks. On ABCI, steps B2 ran as two `rt_HG` jobs (one H200 each, see the
header of `job_abci.sh`); a smoke test (`-v MODE=smoke`) writes to `out/reader_smoke/` and never to the results.

## Setup

```
export MMLB_DIR=<dir with data/train-00000-of-00001.parquet and documents/>   # default <ROOT>/data/mmlongbench
export HF_HOME=<models>                                                        # Qwen/Qwen2.5-VL-7B-Instruct
```

`<ROOT>` defaults to two levels above this folder, as in idea4. Packages: torch, transformers (5.18 was used),
pymupdf, pandas, numpy, matplotlib, pillow. MMLongBench-Doc was fetched by direct URL because one PDF fails the
Hugging Face size check.

## Definitions

- **Eligible question:** answerable, at least one evidence page, every evidence page inside the PDF (qid = row index
  of the parquet file, as in idea4).
- **Sample (Part B):** eligible questions with one evidence page and one evidence type, from documents outside
  idea4's 37, at most 2 per document, 16 per type (layout has only 8 such questions). Seed 555.
- **Text layer:** `page.get_text("text", sort=True)` from PyMuPDF. Fewer than 20 words: no text layer.
- **Image budgets:** 1280 and 256 visual tokens (28 x 28 px each); 256 is about one page's share when five pages
  split a 1280-token budget.
- **Masked image:** the page with every text-layer word box painted white, then resized to 256 tokens.
- **Scoring:** the rule-based part of MMLongBench-Doc's `eval_score.py` (`score.py`). The reader is asked for a short
  answer in the expected format, and a light extraction takes the place of the benchmark's GPT-4o extractor (first
  number for Int and Float, a Python list for List, and a string answer that contains the reference as whole words
  counts as exact, e.g. "Ages 18-34" for "18-34"). A question is correct when its score is above 0 (ANLS for strings).
- **Answer in text (Part A):** every number of the reference answer, or at least 80% of its content words, occurs in
  the gold pages' text layer. Counting questions ("How many ...") are excluded from this rate.
