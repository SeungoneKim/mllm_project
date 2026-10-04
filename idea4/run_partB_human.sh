#!/bin/bash
# Generates André's 20 specs on GPU 1 once the first generation shard is done.
cd "$(dirname "$0")"
while screen -ls | grep -q "\.gen1"; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 ~/anaconda3/envs/cmu/bin/python gen_images.py --specs proposer/human_specs.jsonl --out renders/gen_human > out/log_gen_human.txt 2>&1
