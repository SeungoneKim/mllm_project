#!/bin/bash
# ColQwen2.5 page embeddings. Models live on the local disk (the HF_HOME quota is full).
cd "$(dirname "$0")"
export HF_HOME=/media/ephemeral/dre/hf
CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/cmu/bin/python retrieve.py pages > out/log_colqwen_pages.txt 2>&1
