# mllm_project

Course project for 11-777 Multimodal Machine Learning (CMU, Fall 2026):
**Beyond Pages: Evidence-Centric Retrieval for Long Multimodal Documents**.
Team: André Duarte, Nuno Gonçalves, Eunsu Kim, Seungone Kim, Peiyang Song.

All experiments run on [MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc)
(test-only; nothing is trained on it). The data is not included; `idea1/README.md` shows how to download it so that
`data/data/train-00000-of-00001.parquet` and `data/documents/*.pdf` exist. Two PDFs in the release do not match their
questions and are excluded everywhere (`BAD_DOCS`, see `idea23/idea2_data_analysis/README.md`).

| folder | research idea | what is in it |
|---|---|---|
| `idea1/` | Idea 1: sparse page selection with a null candidate | Analysis 1 (can retrieval scores decide whether and how much to read?): ColQwen2.5 and BM25 page scores, figures |
| `idea23/` | Ideas 2 + 3 (merged): complementarity- and modality-aware evidence selection | Analysis 2 (`idea2_data_analysis/`, `idea3_data_analysis/`) and the **midterm baseline pipeline**: `v0` page-level coverage, `v1` page level + modality forms, `v2` evidence level with a VLM merging retrieved elements into facts, plus three controls; all selections, answers and analysis tables under `idea23/out/` |
| `idea4/` | Idea 4: drawing the evidence a question expects and retrieving with the drawing | Analysis 3: human and machine proposer, rendered queries, ColQwen2.5 retrieval |
| `idea5/` | Idea 5: text layer plus a few text-conditioned visual tokens in the reader | Analysis 4: what the page image adds beyond the text layer |

Each folder has its own `README.md` with setup, steps and results. Refer there for individual information and guidance.
