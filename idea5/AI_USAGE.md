DRAFT, to be edited by Seungone before it goes into the report.

I used Claude Code (Claude Opus 5.5) for Idea 5 and this analysis.

- **Idea development.** After the proposal feedback (overlap with Ideas 2 and 3, no training, multimodality not
  explained), I discussed several revisions with Claude Code. I rejected a look-alike-page describer (overlap with
  the retrieval ideas), a learned zoom policy, and a page-level image-or-text router (too naive). I asked for the idea
  to be grounded in the fusion lecture (redundancy, dominance, emergence, modulation, gated and residual fusion) and
  checked, with example questions from the benchmark, whether the text layer alone already holds the answer. The
  final idea (text layer plus text-conditioned residual visual tokens) and the hypotheses came out of that discussion.
- **Code.** Claude Code wrote the code in this folder (common.py, coverage.py, sample.py, reader.py, score.py,
  analyze.py, job_abci.sh) and this README, and I set up the Python environment and downloads on my gpu cluster, and submitted the
  GPU jobs.
- **Report.** I wrote the report in Korean and asked Claude Code to translate into English.

Qwen2.5-VL-7B-Instruct is part of the experiment: it answers the questions and writes the page transcriptions.
