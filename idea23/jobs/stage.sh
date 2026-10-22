# Node-local copies of the Python environment and of the models a job needs, on the node's own disk.
# /data/user_data (babel-nas-2) can be too slow to import torch or load a model from (about 10 MB/s and 0.2 s per
# small file on 2026-10-21), while PyPI and the Hugging Face hub fill a local disk at about 120 MB/s. Sourced by
# gpu.sbatch unless STAGE=0; MODELS lists the repositories the job needs. Jobs on the same node share the copies.
LOCAL=${LOCAL_DIR:-/scratch/$USER/idea23}
[ -d /scratch ] || LOCAL=/tmp/$USER/idea23
mkdir -p $LOCAL
export UV_CACHE_DIR=$LOCAL/uvcache UV_PYTHON_INSTALL_DIR=$LOCAL/python HF_HOME=$LOCAL/hf
(
  flock 9
  if [ ! -x $LOCAL/env/bin/python ] || ! cmp -s jobs/requirements.lock $LOCAL/requirements.lock; then
    echo "building $LOCAL/env $(date +%T)"
    ~/.local/bin/uv venv -q --allow-existing -p 3.12 $LOCAL/env
    ~/.local/bin/uv pip install -q --python $LOCAL/env/bin/python -r jobs/requirements.lock
    cp jobs/requirements.lock $LOCAL/requirements.lock
  fi
  for m in $MODELS; do
    $LOCAL/env/bin/hf download $m > /dev/null 2>&1 || echo "download of $m failed"
  done
  echo "staged in $LOCAL $(date +%T): ${MODELS:-no models}"
) 9>$LOCAL/.lock
export PY=$LOCAL/env/bin/python
