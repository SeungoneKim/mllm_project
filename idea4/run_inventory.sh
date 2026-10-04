#!/bin/bash
# Step 2 inventory pass on both GPUs. Run inside screen:  screen -S inventory ./run_inventory.sh
cd "$(dirname "$0")"
PY=~/anaconda3/envs/cmu/bin/python
mkdir -p out/inventory
CUDA_VISIBLE_DEVICES=0 $PY inventory.py run --shard 0 --nshards 2 > out/inventory/log_shard0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY inventory.py run --shard 1 --nshards 2 > out/inventory/log_shard1.txt 2>&1 &
wait
$PY inventory.py aggregate | tee out/inventory/log_aggregate.txt
