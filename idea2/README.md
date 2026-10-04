# Analysis 2: existing complementary cases (Ideas 2 + 3)

Code for the subsection "Existing complementary cases" of Analysis 3 in the report: does MMLongBench-Doc already
contain questions whose answer needs evidence from several sources, within one modality (Idea 2) or across
modalities (Idea 3), and does page-level retrieval miss part of that evidence?

The data is not included. Download MMLongBench-Doc as in `idea1/README.md`, e.g.

    hf download yubo2333/MMLongBench-Doc --repo-type dataset --local-dir data

so that you have `data/data/train-00000-of-00001.parquet` and `data/documents/*.pdf`. The download reports a size
mismatch for two PDFs; see the note below. Then, from this folder:

    pip install -r requirements.txt
    python sample.py DATA_DIR     # out/sample.csv + evidence pages rendered to cache/pages/ (CPU, ~1 min)
    python analysis.py DATA_DIR   # prints every number in the subsection, writes out/summary.json and
                                  # out/table_existing.tex (CPU, ~5 s)
    python review.py              # cache/review.html: the 30 labels next to their pages, for checking

`analysis.py` reads the ColQwen2.5 page scores of Analysis 1 from `../idea1/scores.pkl`, so it needs no GPU.

## What each step does

1. **Figure 2 counts.** `same_modality` (panel a): one modality label and at least two listed evidence pages (227
   questions). `cross_modality` (panel b): two or more modality labels and at least one listed page (234; the five
   questions with an empty page list are left out). "Same page" means one listed page. The script asserts both
   totals, so it fails if the definitions drift from the figure.
2. **Sample.** Ten answerable questions with valid page labels from each stratum, at most one per document,
   seed 777: `same_mod` (one modality, >= 2 pages), `cross_same` (>= 2 modalities, one page), `cross_multi`
   (>= 2 modalities, >= 2 pages).
3. **Labels** (`out/annotations.csv`). Every evidence page of the 30 questions was read and each question got one
   category, using the two complementary types of the synthetic-case subsection:
   - `partition`: the sources hold parallel pieces of the answer (list items, instances to count, operands that the
     question names), and all of them are needed;
   - `composition`: one source is needed to interpret or select what to read in another (a bridge entity, a
     mapping, a filter), or the answer combines different quantities through a formula the question does not
     spell out (financial ratios);
   - `redundant`: each source alone gives the answer;
   - `locator`: one source gives the answer and the other only points to it, usually by matching a reference in
     the question (a printed page number, a caption such as "Table 23", a section heading, a row icon).
   For multi-page complementary questions, `one_page_suffices` records whether a careful reader could answer from a
   single listed page anyway (e.g. a numbered list whose last item gives the count). `label_issue` flags gold
   labels that look wrong.
4. **Retrieval check.** For answerable questions with valid labels, the ColQwen2.5 rank of every evidence page;
   the share of questions whose top-k holds all, some or none of their evidence pages, per stratum. Multi-page
   strata keep questions with at most five evidence pages so that the top 5 can hold all of them.

## Notes

- Two PDFs in the Hugging Face release seem to not match their questions, and the hub serves the same wrong bytes at
  the original revision: `mi_phone.pdf` is a university housing guide and
  `dr-vorapptchapter1emissionsources-121120210508-phpapp02_95.pdf` is a marketing deck. Their 19 questions are
  excluded (`BAD_DOCS` in `common.py`). `idea1/scores.pkl` was computed on the same files.
- `sample.py` skips `BAD_DOCS` while drawing, without changing the pool or the random permutation.