#!/bin/bash
# Qwen-Image generation of the proposer specs. Usage: ./run_partB_gen.sh <gpu> <shard> <nshards>
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=$1 ~/anaconda3/envs/cmu/bin/python gen_images.py --shard $2 --nshards $3 > out/log_gen_shard$2.txt 2>&1
