# Ideas 2 + 3: complementarity-aware evidence selection (midterm baseline)

Multi-page questions in MMLongBench-Doc usually need every one of their evidence pages (partition or composition,
Analysis 2), but a page retriever scores pages one at a time, and its top 5 holds all of them for only about half of
these questions. This folder holds the midterm baseline: one retrieve, select, read pipeline in which only the page
selector changes, so that set-level selectors can be compared with top-k under the same reader and page budget.

```
question ─▶ decompose (Qwen3-8B) ─▶ needs d_1..d_m
pages ─────▶ ColQwen2.5 ─▶ s(q, p), s(d_j, p), page vectors ─▶ selector (k pages) ─▶ Qwen2.5-VL-7B ─▶ answer ─▶ score
```

## Versions

One flag, `IDEA23_VERSION` (set by `pipeline.py --version`), chooses what the selector works with. The reader, its
prompt and the page budget k stay fixed, so every version hands the reader exactly k pages.

| version | unit | what the selector knows | own methods | idea |
|---|---|---|---|---|
| `v0` page | page | page scores of the question and of its needs | `coverage`, `coverage_beam` | 2 |
| `v1` page_mod | page | + the forms each need expects (Qwen3-8B, from the question only) and the forms on each page (layout detector) | `coverage_mod`, `coverage_mod_beam` | 2 + 3 at page level |
| `v2` evidence | element (text block, table, chart, figure, heading) | + element scores; per-form retrieval of elements; a VLM merges the retrieved elements into atomic facts (redundant copies, jointly necessary elements); coverage over facts | `elem_topk`, `elem_coverage`, `elem_coverage_mod`, `evidence` | 2 + 3 at evidence level |

```
v1:  need forms (Qwen3-8B) ──┐
     page forms (PP-DocLayoutV3) ─▶ coverage with a discount for pages of another form
v2:  pages ─▶ elements (PP-DocLayoutV3 + figure classifier) ─▶ ColQwen2.5 on each crop
           ─▶ candidates (<= 28): best elements of the first 8 coverage pages + per-form retrieval for each need
           ─▶ VLM (Qwen3-VL) ─▶ atomic facts {evidence, same_as, needs} ─▶ Qwen3-Embedding ─▶ coverage over facts
           ─▶ the facts' pages, then coverage over elements for the rest of the budget ─▶ reader (k pages)
```

The candidate set was revised once after a diagnostic on the test questions: per-form retrieval alone put all gold
pages of a multi-page question among its candidates for 53.9% of the questions, against 60.3% for ColQwen's top pages
at the same page count, so the candidates now also hold the best elements of the first 8 pages that element coverage
selects (70.0%, the same as ColQwen's top pages at the same count; `out/v2/candidates.md`). Nothing else was changed
after looking at results.

The ladder inside v2 separates the parts: `elem_topk` and `elem_coverage` change only the granularity of the scores,
`elem_coverage_mod` adds the forms, and `evidence` adds the VLM's facts. `versions.py` lists each version's methods;
`evidence.py` defines the v1 and v2 selectors. Three controls of v2 are versions too (`pipeline.py --plan` lists six):
`v2_page` and `v2_page_eq` (the VLM reads the candidates' whole pages, at 1,024 or 384 visual tokens a page) and
`v2_32b` (Qwen3-VL-32B, multi-page questions only); see "Controls" below.

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

Shared inputs (every version) are written to `out/`, the selections, metrics and analysis of a version to
`out/<version>/`, and the reader's answers to the shared cache `out/reader/` (a (question, pages) pair is read once,
whichever version selected it).

| step | command | where | output | versions |
|---|---|---|---|---|
| 1 | `python decompose.py` | GPU, Qwen3-8B | `out/decomp.jsonl` | all |
| 2a | `python embed.py pages` | GPU, ColQwen2.5 | `$IDEA23_CACHE/colqwen/*.pt` (page embeddings) | all |
| 2b | `python embed.py score` | GPU | `out/scores.pkl` | all |
| 2c | `python segment.py` | GPU, PP-DocLayoutV3 + figure classifier | `$IDEA23_CACHE/elements/*.json` | v1, v2 |
| 2d | `python need_forms.py` | GPU, Qwen3-8B | `out/need_forms.jsonl` | v1, v2 |
| 2e | `python embed.py elements` / `score_elements` | GPU, ColQwen2.5 | `$IDEA23_CACHE/colqwen_elements/`, `out/element_scores.pkl` | v2 |
| 2f | `python selection.py --candidates` | CPU | `out/<v>/candidates.{jsonl,md}`: up to 28 elements per question | v2 |
| 2g | `python aggregate.py` | GPU, Qwen3-VL-8B (v2) or -32B (v2_32b) | `out/<v>/aggregate/<model>/facts_*.jsonl` | v2 |
| 2h | `python units.py` | GPU, Qwen3-Embedding-4B | `out/<v>/unit_scores.pkl` | v2 |
| 3 | `python selection.py` | CPU, minutes | `out/<v>/selections.jsonl`, `retrieval.{md,csv}`, `sensitivity.csv` | all |
| 4 | `python reader.py --shard i --nshards n` | GPU, Qwen2.5-VL-7B | `out/reader/answers_<v>_<i>.jsonl` | all |
| 4b | `IDEA23_READER=out/reader_cot python reader.py --methods topk,coverage,gold --ks 5 --sets multi --reason` | GPU | ablation: the reader reasons before answering | v0 |
| 4c | `python tag_pages.py` | GPU, Qwen2.5-VL-7B | `out/page_tags.jsonl`: evidence form of each gold page | analysis |
| 5 | `python analyze.py` | CPU | `out/<v>/analysis/{summary.md,metrics.json,per_question.csv,examples.md}` | all |
| 6 | `python export.py` | CPU | `out/dashboard/*.json`, the tables behind the results dashboard | all |

Every script reads the version from `IDEA23_VERSION` (default `v0`). On Babel, `pipeline.py` submits a version's
steps as SLURM jobs with dependencies, skipping the model steps whose outputs are complete (the partition `general`
takes no job without a GPU, so the CPU steps hold one for a few minutes):

```
python pipeline.py --version v2 --plan              # steps and what is done
python pipeline.py --version v2 --submit            # everything not done; --dry prints the sbatch commands
python pipeline.py --version v2 --submit --steps aggregate,units --partition l3lab
python pipeline.py --version v2_32b --submit       # v2 with Qwen3-VL-32B as the VLM, 2 GPUs per shard
IDEA23_VERSION=v2 python analyze.py && python export.py
```

Each job copies the Python environment (`jobs/requirements.lock`, built with `uv`) and the models it needs to the
node's own disk first (`jobs/stage.sh`): on 2026-10-21 `/data/user_data` read at about 10 MB/s, too slow to import
torch from, while PyPI and the Hugging Face hub fill a local disk in a few minutes. `STAGE=0` runs from the shared env
in `jobs/env.sh` instead. The scratch `/tmp` of a login node is not visible to jobs. Do not set `HF_HUB_OFFLINE`: the
ColQwen processor looks up a `config.json` that the adapter repository does not have, which fails offline.

All steps that call a model are resumable. Packages: torch 2.12, transformers 5.17, colpali-engine 0.3.18 (as idea1),
pymupdf, opencv-python-headless, pandas, numpy, pillow, tabulate (`jobs/requirements.lock`).

## First results (2026-10-21)

### v0 (page)

Full tables: `out/v0/retrieval.md`, `out/v0/analysis/summary.md`, examples in `out/v0/analysis/examples.md`.

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

### v1 (page_mod)

Full tables: `out/v1/retrieval.md`, `out/v1/analysis/summary.md`.

| multi-page, k = 5 | complete-evidence recall | reader accuracy |
|---|---|---|
| v0 coverage | 52.9 | 24.5 |
| v1 coverage + forms | 52.9 (k = 3: 41.9 vs 42.6) | 24.2 |

- Page-level modality changes almost nothing: against v0 coverage at k = 5, coverage + forms completes 1 more question
  and 1 fewer, and answers 0 more and 1 fewer; the control is unchanged (complete@5 86.8, accuracy 53.9). The form
  discount gamma moves complete@5 only between 52.9 and 53.5 (`out/v1/sensitivity.csv`).
- Why: almost every page has text, and 79% of gold pages already hold a form their question's needs expect, so the
  discount rarely separates pages. The predicted forms are noisy too: they include one of the question's labelled
  evidence sources for 72% of multi-page questions and all of them for 48%.

### v2 (evidence, Qwen3-VL-8B)

Full tables: `out/v2/retrieval.md`, `out/v2/candidates.md`, `out/v2/analysis/summary.md` (examples with the VLM's
facts in `examples.md`).

| k = 5 | multi complete | multi accuracy | single-page complete | single-page accuracy |
|---|---|---|---|---|
| top-k | 51.3 | 24.2 | 86.3 | 53.4 |
| v0 coverage | 52.9 | 24.5 | 86.5 | 53.9 |
| element coverage + forms (no VLM) | 51.9 | 25.5 | 87.2 | 52.8 |
| evidence facts (v2) | 53.9 | 26.8 | 88.3 | 54.3 |
| gold pages | 100 | 32.9 | 100 | 59.8 |

- The clearest gain is on single-page questions at three pages: the VLM reading the candidate regions finds the one
  evidence page more often (complete@3 85.9 vs 82.3 for top-k, 27 wins / 11 losses, p = 0.014) and the reader
  answers more of them (56.1 vs 52.3, 23 / 6, p = 0.002).
- On multi-page questions v2 is about one point of complete-evidence recall above v0 coverage at k = 5 (16 / 13,
  p = 0.71) and 2.3 points of accuracy (14 / 7, p = 0.19): a direction, not evidence. Finer granularity alone costs
  recall at small budgets (element coverage + forms 38.4 vs coverage 42.6 at k = 3); the VLM's facts win it back
  (42.3). See "Multiple comparisons" below before reading any single p-value.
- The VLM gives no fact for 25% of multi-page questions (18 replies did not parse), and those are the hard ones:
  complete@5 is 32-33% there under every selector, against 61% (evidence) and 60% (coverage) where it gives facts. They
  concentrate in financial reports (49% without a fact), the same documents where v0 retrieves complete evidence for
  only 21% of questions.
- The modality forms show nothing on these questions, at page (v1) or element level (element coverage + forms vs
  element coverage). How much redundant and jointly necessary evidence the VLM finds depends on the VLM: Qwen3-VL-8B
  marks a redundant copy for 1% of multi-page questions and a jointly necessary fact for 9% (6.5% across pages),
  Qwen3-VL-32B for 30% and 25% (14% across pages; `v2_32b` below). So such structure is less rare among the retrieved
  regions than Analysis 2's 3 of 30 gold-evidence cases suggested; whether handling it helps selection is not yet
  separated from the 32B's other gains, and the synthetic cases remain the controlled test.
- Cost: one VLM pass per question over about 18 candidate crops (about 4,000 visual tokens; ~10 s per question on an
  L40S with Qwen3-VL-8B), on top of v0's retrieval.

### Controls: what in v2 does the work

`v2_page` is v2 with one change: the VLM reads the candidates' whole pages (1,024 visual tokens each, about 2.7 times
the crops' total budget) instead of the element crops. Same candidates, prompt, fact retrieval and selection.

| | multi complete@5 | multi acc@5 | multi complete@3 | single-page complete@3 | single-page acc@3 |
|---|---|---|---|---|---|
| top-k | 51.3 | 24.2 | 40.0 | 82.3 | 52.3 |
| v0 coverage | 52.9 | 24.5 | 42.6 | 83.4 | 52.3 |
| v2, VLM reads element crops | 53.9 | 26.8 | 42.3 | 85.9 | 56.1 |
| v2_page, VLM reads whole pages | 57.1 | 27.4 | 46.5 | 87.6 | 55.6 |
| v2_page_eq, whole pages at the crops' budget | 55.8 | 26.8 | 47.1 | 86.8 | 55.2 |
| v2_32b, Qwen3-VL-32B reads the crops (multi only) | 56.5 | 25.8 | 45.5 | - | - |

- The VLM reading the candidates is what helps; reading them as element crops does not add to it. With whole pages
  the selector beats top-k on multi-page complete-evidence recall (k = 5: 29 / 11, p = 0.006; k = 3: 34 / 14,
  p = 0.006) and v0 coverage (k = 5: 23 / 10, p = 0.035), and it beats the crop version directly (k = 5: 14 / 4;
  k = 3: 22 / 9; both p = 0.03). Accuracy follows only as a direction (multi acc@5 27.4 vs 24.2, 18 / 8, p = 0.08).
- So on MMLongBench-Doc the element level shows no measurable contribution of its own: as scoring granularity it
  matches page scores (element coverage 52.3 vs coverage 52.9 at k = 5), as the unit the 8B VLM reads it loses to
  whole pages (the 32B was run on crops only), and the candidate pages it yields hold all gold pages as often as
  ColQwen's top pages of the same count
  (70.0% for both). What works is the VLM reading about ten candidate pages and stating facts, with coverage over
  those facts choosing the pages (Idea 2's set selection, done over facts). Elements may still matter where evidence
  is split across modalities by construction, which is what the synthetic cases test.
- Page context, not budget, is what the crops lose: at the crops' own visual-token budget (384 tokens a page,
  `v2_page_eq`) whole pages still beat the crops on multi-page complete-evidence recall (k = 3: 47.1 vs 42.3,
  22 / 7, p = 0.008; k = 5: 55.8 vs 53.9, 9 / 3, p = 0.15); the larger budget adds a little at k = 5 only. The gain
  is mostly in same-modality questions (k = 3: 12 / 2), where several pages hold evidence of one kind; a plausible
  but unchecked reason is that a crop drops the headings and captions that tell such pages apart.
- A stronger VLM narrows the gap for crops: Qwen3-VL-32B leaves 15.8% of multi-page questions without a fact (8B:
  25.2%; financial reports 28% vs 49%) and reaches complete@5 56.5 (vs top-k 30 / 14, p = 0.023; vs the 8B on crops
  13 / 5, p = 0.10), about what the 8B reaches on whole pages. Accuracy does not follow (25.8 at k = 5): with better
  pages the reader stays the bottleneck. It runs at about one question a minute on two L40S GPUs (8B: ~10 s on one).
  Ignore the single-page rows of `out/v2_32b/analysis/summary.md`: 12 of those questions were read by the 32B before
  the run was limited to multi-page questions, the rest fall back to element coverage.

### Multiple comparisons

The six `summary.md` files report about 1,100 paired cells, many of them repeated across versions; nothing survives a
Bonferroni correction over all of them (p < 0.00005). Over the 60 headline tests (each new method against top-k at
three and five pages and against v0 coverage at five, on complete retrieval and on answers, for multi-page and
single-page questions), the threshold is about 0.0008, and only v2_page's single-page retrieval at three pages passes
it (30 / 6, p = 0.0001). The multi-page retrieval gains of v2_page and v2_32b (p = 0.006 to 0.03) point the same way at
every budget and in both strata but do not pass it one by one; no accuracy difference comes close.

## Reader and scoring

The reader sees the selected pages in document order, each labelled with its page number, at most 1,024 visual
tokens per page, and is asked for a short answer in the expected format (prompt as idea5). A (question, page set)
pair is read once and shared by every method that selected it. Answers are scored with `idea5/score.py`, the
rule-based part of the benchmark's scorer; a question is correct when its score is above 0.
