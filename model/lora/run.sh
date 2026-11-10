#!/usr/bin/env bash
# One independent development run in an existing one-GPU allocation.
set -euo pipefail
CLUSTER=${1:?usage: run.sh CLUSTER MODEL_SIZE CONDITION RUN_ID [extra train.py arguments]}
MODEL_SIZE=${2:?}
CONDITION=${3:?}
RUN_ID=${4:?}
shift 4
case "$CLUSTER" in habrok|lst) ;; *) echo 'Expected habrok or lst' >&2; exit 2;; esac
case "$RUN_ID" in *[!a-zA-Z0-9_-]*|'') echo 'Use letters, digits, underscores or hyphens in RUN_ID' >&2; exit 2;; esac
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$CLUSTER" == habrok ]]; then
    source "$LORA_DIR/habrok/environment.sh"
fi
PYTHON=${PYTHON:-python}
MODEL_CACHE=${MODEL_CACHE:?Set MODEL_CACHE}
DATASET_DIR=${DATASET_DIR:?Set DATASET_DIR to the prepared release}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to persistent scratch storage}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
export WANDB_LOG_MODEL=false WANDB_WATCH=false
export WANDB_MODE=${WANDB_MODE:-online}
export PYTHONHASHSEED=${SEED:-17}
echo "host=$(hostname) cluster=$CLUSTER model=$MODEL_SIZE condition=$CONDITION run=$RUN_ID"
nvidia-smi
exec "$PYTHON" "$LORA_DIR/train.py" --cluster "$CLUSTER" --model-size "$MODEL_SIZE" \
    --condition "$CONDITION" --seed "${SEED:-17}" --dataset-dir "$DATASET_DIR" \
    --model-cache "$MODEL_CACHE" --output-dir "$RUN_ROOT/$RUN_ID" \
    --wandb-mode "$WANDB_MODE" "$@"
