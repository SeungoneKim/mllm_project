"""The three versions of the pipeline, selected with IDEA23_VERSION (pipeline.py --version sets it).

Every version answers the same test questions with the same frozen reader and gives it exactly k pages; the versions
differ in the unit that is retrieved and in what the selector knows about modality.

  v0  page       Pages are the units. Coverage over the decomposed needs of the question (Idea 2, the midterm
                 baseline).
  v1  page_mod   v0 plus modality at page level. Each need is labelled with the evidence forms it expects (text,
                 table, chart, figure, layout; predicted from the question only) and each page with the forms it
                 contains (from the layout detector); a page of another form covers a need only at a discount
                 (Idea 3 at page level).
  v2  evidence   Evidence units. Pages are split into elements by a layout detector and elements are retrieved per
                 form; a VLM merges the retrieved elements into atomic facts, keeping one copy of redundant evidence
                 and marking facts that need several elements together; coverage then runs over facts, and the
                 pages that hold the chosen facts go to the reader (Ideas 2 and 3 together, at evidence level).

`methods` are the selectors a version computes (selection.py) and `reader` those the reader answers. Selectors shared
with an earlier version give the same pages, so their answers come from the shared answer cache at no cost. `main` is
the version's own method, and analyze.py compares every method with each of the `paired` baselines. v2 reads the
elements with `agg_model`; v2_32b is the same version with a larger VLM, and v2_page a control in which the same VLM
reads the candidates' whole pages (1,024 visual tokens each) instead of the element crops, so that what the evidence
level adds can be told apart from what a VLM reading the candidates adds; v2_page_eq gives the pages only the crops'
total visual-token budget (384 tokens a page).
"""

VERSIONS = {
    "v0": {
        "name": "page",
        "methods": ["topk", "bm25", "hybrid", "mmr", "decomp_rrf", "coverage", "coverage_beam", "coverage_full", "gold"],
        "reader": ["topk", "hybrid", "mmr", "decomp_rrf", "coverage", "coverage_beam", "gold"],
        "main": "coverage",
        "paired": ["topk"],
    },
    "v1": {
        "name": "page_mod",
        "methods": ["topk", "coverage", "coverage_mod", "coverage_mod_beam", "gold"],
        "reader": ["topk", "coverage", "coverage_mod", "coverage_mod_beam", "gold"],
        "main": "coverage_mod",
        "paired": ["topk", "coverage"],
    },
    "v2": {
        "name": "evidence",
        "methods": ["topk", "coverage", "coverage_mod", "elem_topk", "elem_coverage", "elem_coverage_mod", "evidence",
                    "gold"],
        "reader": ["topk", "coverage", "coverage_mod", "elem_topk", "elem_coverage", "elem_coverage_mod", "evidence",
                   "gold"],
        "main": "evidence",
        "paired": ["topk", "coverage", "elem_coverage_mod"],
        "agg_model": "Qwen/Qwen3-VL-8B-Instruct",
    },
}
VERSIONS["v2_32b"] = {**VERSIONS["v2"], "name": "evidence_32b", "agg_model": "Qwen/Qwen3-VL-32B-Instruct",
                      "agg_sets": "multi",  # about 1 question per minute on two GPUs: multi-page questions only
                      "rename": {"evidence": "evidence_32b"}}  # how export.py tells it apart from v2
VERSIONS["v2_page"] = {**VERSIONS["v2"], "name": "evidence_page_vlm", "agg_unit": "page", "page_tokens": 1024,
                       "rename": {"evidence": "evidence_pagevlm"}}
# the same control at the crops' visual-token budget (about 4,000 per question over about 10.5 pages)
VERSIONS["v2_page_eq"] = {**VERSIONS["v2_page"], "name": "evidence_page_vlm_eq", "page_tokens": 384,
                          "rename": {"evidence": "evidence_pagevlm_eq"}}
