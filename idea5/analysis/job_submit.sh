#!/bin/bash
# Part B reader on one ABCI H200 (rt_HG). Submit from this folder:
#   qsub -P <group> -q rt_HG -l select=1 -l walltime=00:20:00 -v MODE=smoke -N idea5_smoke job_abci.sh
#   qsub -P <group> -q rt_HG -l select=1 -l walltime=01:00:00 -v SHARD=0,NSHARDS=2 -N idea5_s0 job_abci.sh
# Set W to the folder that holds venv/, hf/ and this repository's copy.
#PBS -j oe
set -e
W=${W:-/groups/gai51740/aci18915qy/mllm_project}
cd $W/idea5
source $W/venv/bin/activate
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 MMLB_DIR=$W/data/mmlongbench IDEA4_DIR=$W/idea4
nvidia-smi -L
mkdir -p out/logs
if [ "$MODE" = "smoke" ]; then
  IDEA5_READER=out/reader_smoke python -u reader.py --limit 2 > out/logs/smoke.txt 2>&1
else
  python -u reader.py --shard ${SHARD:-0} --nshards ${NSHARDS:-1} > out/logs/shard_${SHARD:-0}_of_${NSHARDS:-1}.txt 2>&1
fi
echo "finished $(date)"
