# Ideas 2 + 3: complementarity-aware evidence selection (midterm baseline)

Multi-page questions in MMLongBench-Doc usually need every one of their evidence pages (partition or composition,
Analysis 2), but a page retriever scores pages one at a time, and its top 5 holds all of them for only about half of
these questions. This folder holds the midterm baseline: one retrieve, select, read pipeline in which only the page
selector changes, so that set-level selectors can be compared with top-k under the same reader and page budget.

```
question ─▶ decompose (Qwen3-8B) ─▶ needs d_1..d_m
pages ─────▶ ColQwen2.5 ─▶ s(q, p), s(d_j, p), page vectors ─▶ selector (k pages) ─▶ Qwen2.5-VL-7B ─▶ answer ─▶ score
```

## Selectors

Scores s^ are ColQwen2.5 MaxSim scores min-max normalised over the pages of the document. Rerankers choose from the
same candidate pool C, the ColQwen top-20.

| method | rule | source |
|---|---|---|
| `topk` | top k by s(q, p) | M3DocRAG |
| `bm25` | top k by BM25 over the text layer | text channel |
| `hybrid` | top k by z(s(q, .)) + z(BM25) | text + image channels fused |
| `mmr` | greedy a s^(q, p) - (1 - a) max cos(e_p, e_p'), a = 0.5, mean-pooled page vectors | MMR / DF-RAG |
| `decomp_rrf` | reciprocal-rank fusion (c = 60) of the rankings of q and every need | Ammann et al. 2025 |
| `coverage` | greedy max F(S) = sum_j (1/m) max_{p in S} s^(d_j, p) + (1/k) sum_{p in S} s^(q, p) | ours, v0 |
| `coverage_beam` | the same, beam search with B = 3 | ours, ablation |
| `coverage_full` | coverage over the whole document instead of C | ablation, retrieval only |
| `gold` | the gold evidence pages | reader upper bound |

The constants were fixed before looking at results; `out/sensitivity.csv` shows how the retrieval numbers move with
them, for discussion only.

## Test sets

Nothing is trained, so all questions are test data. Valid labels: every evidence page inside the PDF, at least one
evidence-source label, and not one of the two mismatched PDFs (`BAD_DOCS`, see idea2/README.md).

- `multi`: answerable, 2 to 5 distinct evidence pages, 310 questions (201 `same_mod`, 109 `cross_multi`); the same
  population as the retrieval check of Analysis 2.
- `control`: answerable, one evidence page, 453 questions; a multi-page selector should not hurt them.

## Steps

| step | command | where | output |
|---|---|---|---|
| 1 | `python decompose.py` | GPU, Qwen3-8B | `out/decomp.jsonl` |
| 2a | `python embed.py pages` | GPU, ColQwen2.5 | `$IDEA23_CACHE/colqwen/*.pt` (page embeddings) |
| 2b | `python embed.py score` | GPU | `out/scores.pkl` |
| 3 | `python selection.py` | CPU, seconds | `out/selections.jsonl`, `out/retrieval.{md,csv}`, `out/sensitivity.csv` |
| 4 | `python reader.py --shard i --nshards n` | GPU, Qwen2.5-VL-7B | `out/reader/answers_<i>.jsonl` |
| 4b | `IDEA23_READER=out/reader_cot python reader.py --methods topk,coverage,gold --ks 5 --sets multi --reason` | GPU | ablation: the reader reasons before answering |
| 4c | `python tag_pages.py` | GPU, Qwen2.5-VL-7B | `out/page_tags.jsonl`: evidence form of each gold page |
| 5 | `python analyze.py` | CPU | `out/analysis/{summary.md,metrics.json,per_question.csv,examples.md}` |
| 6 | `python export.py` | CPU | `out/dashboard/*.json`, the tables behind the results dashboard |

On Babel, from this folder (paths in `jobs/env.sh`; the scratch `/tmp` of a login node is not visible to jobs):

```
sbatch -J pages jobs/gpu.sbatch bash jobs/retrieve.sh pages        # 2a  } in parallel
sbatch -J decompose jobs/gpu.sbatch bash jobs/retrieve.sh decompose # 1   }
sbatch -J score jobs/gpu.sbatch bash jobs/retrieve.sh score         # 2b, after both
$PY selection.py
sbatch -J reader --array=0-5 jobs/gpu.sbatch bash jobs/reader.sh    # 4, six shards (about 20 min)
sbatch -J readercot --array=0-5 --export=ALL,NSHARDS=6 jobs/gpu.sbatch bash -c \
  'export IDEA23_READER=$PWD/out/reader_cot; bash jobs/reader.sh --methods topk,coverage,gold --ks 5 --sets multi --reason'
sbatch -J tagpages jobs/gpu.sbatch bash -c '$PY -u tag_pages.py'   # 4c
$PY analyze.py && $PY export.py
```

Models are fetched once beforehand with `hf download` (vidore/colqwen2.5-v0.2, vidore/colqwen2.5-base,
Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-8B). Do not set `HF_HUB_OFFLINE`: the ColQwen processor looks up a
`config.json` that the adapter repository does not have, which fails offline.

All steps that call a model are resumable. Packages: torch 2.12, transformers 5.17, colpali-engine 0.3.18 (as idea1),
pymupdf, pandas, numpy, pillow, tabulate.

## First results (2026-10-21)

Full tables: `out/retrieval.md`, `out/analysis/summary.md`, examples in `out/analysis/examples.md`.

| multi-page, k = 5 | complete-evidence recall | reader accuracy |
|---|---|---|
| top-k (M3DocRAG) | 51.3 | 24.2 |
| hybrid / MMR / decompose + RRF | 47.7 / 43.9 / 49.4 | 23.2 / 23.9 / 23.2 |
| coverage (ours, v0) | 52.9 (k = 3: 42.6 vs 40.0) | 24.5 |
| gold pages | 100 | 32.9 (single-page control: 59.8) |

- Reading is the larger bottleneck: the gold pages fix only 36 of top-k's 235 wrong answers. Asking the reader to
  reason first lifts gold-page accuracy to 38.7, still far below single-page questions.
- Coverage helps only where the decomposer finds several needs (m = 2: complete@5 70.9 vs 66.7); 56% of
  multi-page questions get a single need (counting, enumeration, implicit formulas such as financial ratios), where
  it picks the same five pages as top-k for 91% of them. Needs rank a gold page first for only 39% of gold pages,
  which is why greedy, beam and the whole document give the same result.
- The coverage gain is a direction, not yet evidence: at k = 5 it completes 6 questions top-k does not and loses 1
  (McNemar p = 0.13), and answers 2 more right and 1 fewer (accuracy +0.3).
- MMR, fusion with BM25 and decompose + RRF lose evidence on multi-page questions and 1-2 points on the single-page
  control; coverage does not (control complete@5 86.5 vs 86.3).
- Hardest pages: tables in same-modality questions (median ColQwen rank 6, missed by top-k 53% of the time);
  financial reports (complete@5 21%, accuracy 0%).

## Reader and scoring

The reader sees the selected pages in document order, each labelled with its page number, at most 1,024 visual
tokens per page, and is asked for a short answer in the expected format (prompt as idea5). A (question, page set)
pair is read once and shared by every method that selected it. Answers are scored with `idea5/score.py`, the
rule-based part of the benchmark's scorer; a question is correct when its score is above 0.
