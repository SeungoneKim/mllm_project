#!/bin/bash
# One GPU job running the command given after the script, from idea23/. Submit from idea23/, e.g.
#   sbatch -J retrieve jobs/gpu.sbatch bash jobs/retrieve.sh
#   sbatch -J reader --array=0-3 --export=ALL,MODELS=Qwen/Qwen2.5-VL-7B-Instruct jobs/gpu.sbatch bash jobs/reader.sh
# pipeline.py submits every step this way. The environment and the models in MODELS are copied to the node's disk
# first (jobs/stage.sh); STAGE=0 runs from /data/user_data instead.
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=out/logs/%x_%A_%a.txt
set -e
source jobs/env.sh
[ "${STAGE:-1}" = 0 ] || source jobs/stage.sh
echo "host $(hostname), $(date)"; nvidia-smi -L 2>/dev/null || echo "no GPU"
"$@"
echo "finished $(date)"
