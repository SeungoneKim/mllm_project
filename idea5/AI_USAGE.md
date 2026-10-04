DRAFT, to be edited by Seungone before it goes into the report.

I used Claude Code (Claude Opus 5.5) for this analysis. It read our proposal and André's Idea 4 outputs, checked prior
work (SimpleDoc, MLDocRAG, REAL-MM-RAG, RegionRAG) and helped me reframe Idea 5 so that it no longer overlaps Ideas 2
and 3. A first look at André's 22 ColQwen misses, which Claude ran, showed that most were not look-alike pages, so we
designed this analysis to test that premise on the full benchmark.

Claude wrote the code in this folder (scan.py, attribute.py, cues.py, build_form.py, form_template.html, describe.py,
score_desc.py, analyze_human.py), the synthetic tests, and this folder's README, following André's idea4 code for
rendering, retrieval and the annotation form. I [describe what you reviewed, changed, ran and decided].

The human annotations were written by [names] without AI assistance. Qwen2.5-VL-7B-Instruct wrote the machine baseline
descriptions as part of the experiment.
