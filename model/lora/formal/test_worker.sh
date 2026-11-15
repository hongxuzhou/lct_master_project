#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cluster=${1:?expected cluster}
root=${2:?expected test work directory}
run_id=${3:?expected frozen run ID}
manifest=${4:-frozen-$cluster.json}
case "$cluster" in habrok|lst) ;; *) exit 2 ;; esac
source "$here/../tuning/$cluster/environment_v2.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR
TRITON_CACHE_DIR=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/qwen35-test.XXXXXX")
trap 'rm -rf -- "$TRITON_CACHE_DIR"' EXIT
"$PYTHON" "$here/predict.py" --manifest "$root/$manifest" --run-id "$run_id" \
    --test-root "$root/test" --test-hashes "$root/test-hashes.json" \
    --output "$root/predictions/$run_id.tsv"
