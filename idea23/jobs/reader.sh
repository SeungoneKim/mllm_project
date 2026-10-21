# Step 4, one shard per array task; extra arguments go to reader.py.
set -e
$PY -u reader.py --shard ${SLURM_ARRAY_TASK_ID:-0} --nshards ${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}} "$@"
