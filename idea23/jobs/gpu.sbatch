#!/bin/bash
# One GPU job running the command given after the script, from idea23/. Submit from idea23/, e.g.
#   sbatch -J retrieve jobs/gpu.sbatch bash jobs/retrieve.sh
#   sbatch -J reader --array=0-3 jobs/gpu.sbatch bash jobs/reader.sh
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=out/logs/%x_%A_%a.txt
set -e
source jobs/env.sh
echo "host $(hostname), $(date)"; nvidia-smi -L
"$@"
echo "finished $(date)"
