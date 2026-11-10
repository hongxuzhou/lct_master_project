#!/usr/bin/env bash
# Lightweight checks only; no training or package imports on the login node.
set -euo pipefail
HABROK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LORA_DIR=$(cd "$HABROK_DIR/.." && pwd)
source "$HABROK_DIR/environment.sh"
JOB_SCRIPT=${1:-$HABROK_DIR/canary.sbatch}
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing interpreter: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "Missing prepared model cache: $MODEL_CACHE" >&2
    exit 1
fi
for SPLIT in train validation; do
    if [[ ! -r "$DATASET_DIR/data/$SPLIT.parquet" ]]; then
        echo "Missing prepared split: $DATASET_DIR/data/$SPLIT.parquet" >&2
        exit 1
    fi
done
bash -n "$JOB_SCRIPT"
command -v sbatch >/dev/null
mkdir -p "$HABROK_DIR/logs" "$RUN_ROOT"
printf 'PYTHON=%s\nMODEL_CACHE=%s\nDATASET_DIR=%s\nRUN_ROOT=%s\nWANDB_MODE=%s\nJOB_SCRIPT=%s\n' \
    "$PYTHON" "$MODEL_CACHE" "$DATASET_DIR" "$RUN_ROOT" "$WANDB_MODE" "$JOB_SCRIPT"
cd "$HABROK_DIR"
sbatch --export=ALL "$JOB_SCRIPT"
