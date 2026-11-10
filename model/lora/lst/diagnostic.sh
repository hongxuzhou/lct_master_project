#!/bin/bash
set -euo pipefail
trap 'rc=$?; echo "Finished: $(date -Is), exit=$rc"' EXIT
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOB_ID=${1:?Condor job ID required}
export PYTHON=/nethome/honzhou/miniconda3/envs/thesis-qwen35-opt/bin/python
export MODEL_CACHE=/scratch/common_models/HuggingFace
export DATASET_DIR=/scratch/hongxuzhou/thesis/repair-sbn-9b7d8812
export RUN_ROOT=/scratch/hongxuzhou/thesis/lst_kernel_wandb_canary
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONHASHSEED=17
export WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
export WANDB_CONSOLE=off WANDB_LOG_MODEL=false WANDB_WATCH=false
export WANDB_BASE_URL=https://api.forge.coreweave.com
unset WANDB_RUN_ID WANDB_RESUME
test -x "$PYTHON"
mkdir -p "$RUN_ROOT"
nvidia-smi
"$PYTHON" -m pip check
"$PYTHON" "$LORA_DIR/lst/diagnostic.py" \
    --protocol "$LORA_DIR/prompts/sbn_v2/protocol.json" \
    --cluster lst --model-size 2B --condition C3 --purpose canary \
    --seed 17 --max-steps 30 --eval-every 10 --generation-examples 20 \
    --micro-batch-size 1 --dataset-dir "$DATASET_DIR" --model-cache "$MODEL_CACHE" \
    --output-dir "$RUN_ROOT/canary-lst-2B-C3-kernels-wandb-s17-$JOB_ID" \
    --wandb-mode online
