#!/bin/bash
# Steps 1-3a on a machine with two GPUs: render, embed (one shard per GPU), score, attribute, build the form.
# Set MMLB_DIR, IDEA4_DIR and HF_HOME first (README.md). Logs go to out/log_*.txt. Resumable: rerun after a failure.
set -e
cd "$(dirname "$0")"
PY=${PY:-~/anaconda3/envs/cmu/bin/python}
mkdir -p out
$PY scan.py render --workers 8 > out/log_render.txt 2>&1
CUDA_VISIBLE_DEVICES=0 $PY scan.py embed --shard 0 --nshards 2 > out/log_embed0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scan.py embed --shard 1 --nshards 2 > out/log_embed1.txt 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY scan.py score > out/log_score.txt 2>&1
$PY attribute.py > out/log_attribute.txt 2>&1
$PY build_form.py --n 20 > out/log_form.txt 2>&1
echo "finished $(date)" >> out/log_form.txt
