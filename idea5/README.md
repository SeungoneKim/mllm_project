# Idea 5: do look-alike pages cause retrieval failures? (feasibility analysis)

Revised Idea 5 trains the model that writes page descriptions so that each description picks out its own page from the
look-alike pages of the same document (self-retrieval reward). This analysis tests its premises before any training:

1. **Step 1-2, data analysis.** Run ColQwen2.5 on every eligible MMLongBench-Doc question and attribute each miss
   to a cause: the question cites a page, a look-alike page won, the question's words are not on the gold page, or other.
2. **Step 3, human simulation.** Annotators play the describer on about 20 gold pages, first with the page alone and
   then next to its 3 most similar pages, without seeing the question. Each description is scored as a ColQwen query
   (the proposed training reward), against the untrained VLM doing the same task.

## Hypotheses (fixed before running)

- **H1.** At least 25% of misses are look-alike misses (the page that beat the gold page is among the gold page's 3
  most similar pages), and the rate is above chance.
- **H2.** With document length and evidence type held fixed, a closer nearest look-alike raises the odds of a miss.
- **H3.** Descriptions written next to the look-alikes rank their own page first among the 4 pages more often than
  descriptions of the page alone, and the human does better than the untrained VLM (the room training must close).

Reading the outcomes: H1 and H2 hold, so look-alike confusion is a measured failure that no other idea addresses.
H3 holds, so the self-retrieval reward has signal and the gap to the VLM is learnable. If H1 fails, the cause table
says which failures dominate, and Idea 5 is revised for the midterm.

## Setup on the cluster

Place this folder next to André's `idea4` folder, or set the paths explicitly:

```
export MMLB_DIR=<dir with data/train-00000-of-00001.parquet and documents/>   # default <ROOT>/data/mmlongbench
export IDEA4_DIR=<André's idea4 folder with cache/pages and cache/colqwen>     # default ../idea4 (read only)
export HF_HOME=<models>    # vidore/colqwen2.5-v0.2 (+ colqwen2.5-base), Qwen/Qwen2.5-VL-7B-Instruct
PY=~/anaconda3/envs/cmu/bin/python
```

`<ROOT>` defaults to two levels above this folder, the same convention as idea4. New page images and embeddings go to
`idea5/cache/` (override with `IDEA5_CACHE`); André's caches are only read. Packages: the `cmu` env (torch,
colpali_engine, transformers, pymupdf, pandas, scipy, matplotlib).

## Steps

| step | command | where | output |
|---|---|---|---|
| 1 render | `$PY scan.py render --workers 8` | CPU | `cache/pages/` (pages not already in idea4's cache) |
| 1 embed | `CUDA_VISIBLE_DEVICES=0 $PY scan.py embed --shard 0 --nshards 2` (and shard 1 on GPU 1) | GPU | `cache/colqwen/` |
| 1 score | `CUDA_VISIBLE_DEVICES=0 $PY scan.py score` | GPU | `out/scan/scores.jsonl`, `out/scan/pagesim/` |
| 2 attribute | `$PY attribute.py` | CPU | `out/attr/` summary, metrics, questions.csv, 4 figures |
| 3a form | `$PY build_form.py --n 20` | CPU | `annotation/`, `annotation.zip`, `out/human/items.csv` |
| 3a annotate | unzip, open `annotation/index.html`, Export; put the JSON files in `exports/` | laptop | `exports/annotations_<name>_*.json` |
| 3b VLM | `CUDA_VISIBLE_DEVICES=0 $PY describe.py` | GPU | `out/human/vlm_descriptions.jsonl` |
| 3c score | `CUDA_VISIBLE_DEVICES=0 $PY score_desc.py` | GPU | `out/human/desc_scores.jsonl` |
| 3d analyse | `$PY analyze_human.py` | CPU | `out/human/` summary, metrics, 2 figures |

`run_scan.sh` runs steps 1-3a in order (two GPUs). Every GPU step is resumable. Use `--limit-docs 2` on any `scan.py`
step for a smoke test. Step 3b can run while people annotate; 3c needs both.

Rough cost: the benchmark has about 6,400 pages and idea4 already embedded 1,165. Rendering took about 0.04 s per page
on a laptop with 4 workers. Annotation takes about 3 minutes per item (two descriptions, cues, confidence), so 20 items
is about an hour per annotator. A second annotator is recommended by the assignment.

## Definitions

- **Eligible question**: answerable, at least one evidence page, every evidence page inside the PDF. The qid is the
  row index of the parquet file, the same as idea4.
- **Miss**: no gold page in the top 5. For multi-page questions, `g` is the best-ranked gold page.
- **Look-alike miss**: the top non-gold page is among the 3 non-gold pages most similar to `g`. Page similarity is
  ColQwen late interaction between two pages (mean over the tokens of one page of the max dot product with the other
  page, symmetrised), the same geometry as retrieval.
- **nn_z**: similarity of `g` to its most similar other page, z-scored within the document.
- **Page ref**: the question cites a page or slide (regex in `cues.py`); *label mismatch* when no cited number is a gold
  page index (printed page labels).
- **Vocabulary gap**: under half of the question's content words are in the text layer of `g` (pages with at least 20
  text-layer words; others count as *no text layer*).
- **Self-retrieval** (step 3): a description used as a ColQwen query; *1st of 4* is the target ranked above its 3
  look-alikes (chance 25%).

`attribute.py` also checks that its question-as-text ranks equal André's (idea4 Part B) on his 144 questions; the
first lines of `out/attr/summary.md` report it.

## Tests (no GPU)

```
export MMLB_DIR=<a folder with the parquet and a few PDFs>
IDEA5_DEVICE=cpu IDEA5_OUT=tests/out_SYNTHETIC_SCAN IDEA5_CACHE=/tmp/cache_SYNTHETIC python tests/test_scan_cpu.py
IDEA5_OUT=tests/out_SYNTHETIC python tests/make_synthetic.py && IDEA5_OUT=tests/out_SYNTHETIC python attribute.py
IDEA5_OUT=tests/out_SYNTHETIC IDEA5_FORM=/tmp/form_SYNTHETIC/annotation python build_form.py --n 8
IDEA5_OUT=tests/out_SYNTHETIC IDEA5_EXPORTS=tests/exports_SYNTHETIC python tests/make_synthetic_human.py
IDEA5_OUT=tests/out_SYNTHETIC IDEA5_EXPORTS=tests/exports_SYNTHETIC python analyze_human.py
```

`test_scan_cpu.py` checks the scoring math against idea4's per-page loop and runs `scan.score()` end to end with random
embeddings in place of ColQwen. The synthetic files hold random values, not results, and the scripts refuse to write
them outside paths containing `SYNTHETIC`. These tests passed on four MMLongBench-Doc documents. The form was checked
for JavaScript syntax but not clicked through in a browser; open it once before annotating.
