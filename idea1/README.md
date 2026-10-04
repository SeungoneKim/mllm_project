# Analysis 1 code

Code for Analysis 1 of the report (can retrieval scores decide whether and how much to read?).

The data is not included, download MMLongBench-Doc yourself from
https://huggingface.co/datasets/yubo2333/MMLongBench-Doc, for example with

    hf download yubo2333/MMLongBench-Doc --repo-type dataset --local-dir data

so that you have `data/data/train-00000-of-00001.parquet` and `data/documents/*.pdf`. Then

    pip install -r requirements.txt
    python scores.py data      # page text, BM25 and ColQwen2.5 scores -> scores.pkl (GPU, ~15 min)
    python analysis.py data    # prints all the numbers in the report and saves the 3 figures (CPU, ~10 s)

The questions printed at the end of H2 are the ones we read by hand.
