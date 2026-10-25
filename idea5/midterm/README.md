# Idea 5 midterm baseline: the text layer plus K visual tokens chosen without training

Idea 5 gives the reader each page's text layer exactly and only a few visual tokens, chosen to carry what the text
lacks. The final method learns that choice. The midterm baseline is the same reader **before any training**: the
visual tokens are chosen by training-free rules, with SparseVLM (Zhang et al., ICML 2025), a text-guided visual
token selector, as the main baseline from prior work.

- **Reader:** Qwen2.5-VL-7B-Instruct, greedy, on the gold evidence pages only (no retrieval).
- **Test set:** all 829 MMLongBench-Doc questions that are answerable and whose evidence pages lie inside the PDF
  (475 single-page, 354 multi-page with up to 24 pages; 1,220 pages; 219 questions have a gold page without a text
  layer). MMLongBench-Doc is test-only; nothing is trained or tuned on it. The 244 unanswerable questions have no gold
  pages and the 18 with page labels outside the PDF cannot be read, so they are left out.
- **Budgets:** K = 64, 128 or 256 visual tokens per page (out of about 1,250 at full resolution), so K x N for a
  question with N gold pages. Every selector chooses across all of a question's pages jointly.

## Conditions (24)

| condition | what the reader gets besides the question and the text layer |
|---|---|
| `text` | nothing (lower reference) |
| `full` | all visual tokens (upper reference) |
| `full_imgfirst` | all visual tokens, image before the text (the analysis's order) |
| `down_K` | the whole page downscaled to K tokens |
| `rand_K` | K random tokens of the full-resolution page |
| `ink_K` | the K tokens with the most ink |
| `comp_K` | the K tokens with the most ink outside the text layer (Idea 5's premise, untrained) |
| `fastv_K` | FastV (ECCV 2024): after layer 1, the K tokens the last prompt token attends to most |
| `svlm_K` | SparseVLM: question raters, rater attention at layer 2, top tokens + recycled merged tokens = K |
| `svlmnr_K` | SparseVLM without recycling (ablation) |

The text layer comes before the image and the question after it (except `full_imgfirst`), so the visual tokens are
computed in the context of the text and SparseVLM's raters are the question tokens, as in the original method. Kept
tokens keep their original 3D rotary positions. FastV and SparseVLM see all visual tokens in their first layers
(1 and 3 layers); `vis_per_layer` in the outputs reports the average over the 28 layers.

FastV and SparseVLM are ported from their official code (pkunlp-icler/FastV; Gumpest/SparseVLMs, v1 algorithm:
rater selection, rater-averaged attention, top-k, density-peak recycling). Their code supports LLaVA only, so the
port runs Qwen2.5-VL's decoder layer by layer (`reader.py`).

## Steps

| step | command | where | output |
|---|---|---|---|
| 1 | `python build_test.py` | CPU, ~80 s | `out/test/{items,pages}.jsonl`, `out/test/tokinfo.npz`, `cache/pages/` |
| 2 | `python reader.py --shard k --nshards 48` | GPU | `out/reader/answers_<k>.jsonl` |
| 3 | `python evaluate.py` | CPU | `out/eval/` summary, metrics, scored rows, 3 figures |

On ABCI, `jobs/reader.pbs` runs 8 reader shards on one reservation node; the midterm run used 6 nodes (48 shards),
and the job that finds every row written runs `evaluate.py` (see the header, and `class_documents/abci_doc.md` for
the cluster rules). With several gold pages, the pages are numbered in reading order, each page's text layer comes
before its image, and the text layers share 36,000 characters; single-page prompts are worded as in the analysis. Environment: torch
2.11, transformers 5.18, pymupdf, numpy, pandas, matplotlib, pillow. Set `MMLB_DIR` to the MMLongBench-Doc folder
and `HF_HOME` to the model cache.

## Definitions

- **Visual token grid:** Qwen2.5-VL turns each 28 x 28 pixel block of the resized page into one visual token
  (row-major). `tokinfo.npz` holds, per block, `ink` (share of pixels darker than 200 gray) and `txt` (share covered
  by text-layer word boxes).
- **Scoring:** the analysis's rule-based MMLongBench-Doc scorer (`../analysis/score.py`); correct = score > 0.
- **Image-needed questions:** wrong with the text layer only and right with the full image; **text-sufficient:**
  right with the text layer only.
- **Where kept tokens fall:** on text-layer words (`txt` > 0.5), other ink (`ink` >= 0.02), or blank.

## Tests (no GPU)

```
python tests/test_reader_cpu.py        # tiny random Qwen2.5-VL with the real config and processor
IDEA5_MT_READER=tests/out_SYNTHETIC/reader python tests/make_synthetic_answers.py
IDEA5_MT_READER=tests/out_SYNTHETIC/reader IDEA5_MT_EVAL=tests/out_SYNTHETIC/eval python evaluate.py
```

`test_reader_cpu.py` checks that the layer-by-layer reader gives the same logits and tokens as the model's own
`forward` and `generate`, that its attention scores equal the model's eager attention, that pruning which keeps every
token changes nothing, that every condition respects its budget, and that the run loop resumes without duplicates.
The synthetic answers are random values, not results; the scripts refuse to write them outside `SYNTHETIC` paths.
