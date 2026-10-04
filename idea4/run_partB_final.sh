#!/bin/bash
# After every generation is done: rescore all query variants (now with André's generated images), then analyse.
cd "$(dirname "$0")"
while screen -ls | grep -q -E "\.(gen1|genh|partB)"; do sleep 30; done
HF_HOME=/media/ephemeral/dre/hf CUDA_VISIBLE_DEVICES=0 ~/anaconda3/envs/cmu/bin/python retrieve.py queries > out/log_colqwen_queries2.txt 2>&1
~/anaconda3/envs/cmu/bin/python analyze_B.py > out/log_analyze_B.txt 2>&1
echo "finished $(date)" >> out/log_analyze_B.txt
