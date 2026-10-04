#!/bin/bash
# Waits for page embeddings, generates the second half of the images on GPU 0, waits for GPU 1, then scores all queries.
cd "$(dirname "$0")"
while screen -ls | grep -q "\.colqwen"; do sleep 20; done
./run_partB_gen.sh 0 1 2
while screen -ls | grep -q "\.gen1"; do sleep 20; done
HF_HOME=/media/ephemeral/dre/hf CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/cmu/bin/python retrieve.py queries > out/log_colqwen_queries.txt 2>&1
