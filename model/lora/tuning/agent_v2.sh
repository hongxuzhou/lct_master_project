#!/usr/bin/env bash
# Run only inside an allocated GPU job. One condition per agent.
set -euo pipefail
CLUSTER=${1:?usage: agent_v2.sh CLUSTER SIZE REGISTRY CONDITION COUNT [fit]}
MODEL_SIZE=${2:?}
REGISTRY=${3:?}
CONDITION=${4:?}
COUNT=${5:?}
MODE=${6:-tuning}
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$CLUSTER/$MODEL_SIZE" in habrok/2B|lst/9B|lst/27B) ;; *) echo 'Unsupported cluster/model allocation' >&2; exit 2;; esac
[[ "$COUNT" =~ ^[1-9]$ ]] || { echo 'COUNT must be 1–9' >&2; exit 2; }
[[ "$MODE" == tuning || "$MODE" == fit ]] || exit 2
case "$CONDITION" in C1|C2|C3) ;; *) exit 2;; esac
source "$LORA_DIR/tuning/$CLUSTER/environment_v2.sh"
export TUNING_CLUSTER=$CLUSTER TUNING_MODEL_SIZE=$MODEL_SIZE TUNING_CONDITION=$CONDITION
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONHASHSEED=17
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_MODE=online
export WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
export WANDB_LOG_MODEL=false WANDB_WATCH=false
unset WANDB_RUN_ID WANDB_RESUME
export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p "$TUNING_RUN_ROOT"
export WANDB_DIR=$TUNING_RUN_ROOT/wandb WANDB_CACHE_DIR=$TUNING_RUN_ROOT/wandb-cache
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
if [[ "$MODE" == tuning ]]; then
    SWEEP_PATH=$("$PYTHON" "$LORA_DIR/tuning/registry.py" "$REGISTRY" "$MODEL_SIZE" "$CONDITION")
fi
cd "$LORA_DIR"
"$PYTHON" - <<'PY'
import os
from importlib.metadata import version
import torch
if version('wandb') != '0.31.0':
    raise SystemExit('Requires the existing wandb==0.31.0 environment')
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expected exactly one allocated visible GPU')
gpu = torch.cuda.get_device_properties(0)
size = os.environ['TUNING_MODEL_SIZE']
expected = 'H100' if size == '27B' else 'A100'
if expected not in gpu.name or gpu.total_memory < (79 if size == '27B' else 39) * 1_000_000_000:
    raise SystemExit(f'Expected {expected} with sufficient memory; got {gpu.name}')
print(f'GPU: {gpu.name}; {gpu.total_memory / 2**30:.1f} GiB; torch {torch.__version__}')
from data import load_rows, read_protocol, select_train_subset
from tuning.plans import load_plan
plan = load_plan('lr-rank-v2')
protocol = read_protocol(plan['protocol'])
select_train_subset(load_rows(os.environ['DATASET_DIR'], protocol), plan['subset'], protocol)
PY
if [[ "$MODE" == fit ]]; then
    # A separate three-step calibration: rank 64, full dev and best reload.
    # It never consumes a sweep assignment or contributes to tuning selection.
    FIT_ID="${SLURM_JOB_ID:-${_CONDOR_JOB_AD:-manual}}"
    FIT_ID=$(basename "$FIT_ID")
    FIT_OUTPUT="$TUNING_RUN_ROOT/fit-checks/$MODEL_SIZE/$CONDITION/$FIT_ID-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    exec "$PYTHON" train.py --protocol protocols/fluent_groups_v1.published.json \
        --purpose calibration --model-size "$MODEL_SIZE" --condition "$CONDITION" \
        --cluster "$CLUSTER" --seed 17 --max-steps 3 --eval-every 3 \
        --lora-rank 64 --lora-alpha 128 --learning-rate 0.0004 \
        --train-subset tuning/subsets/train10_fluent_groups_v1_s20261103/manifest.json \
        --micro-batch-size 1 --generation-examples 0 --wandb-mode disabled \
        --dataset-dir "$DATASET_DIR" --model-cache "$MODEL_CACHE" --output-dir "$FIT_OUTPUT"
fi
AGENT_STATUS=0
"$(dirname "$PYTHON")/wandb" agent --count "$COUNT" "$SWEEP_PATH" || AGENT_STATUS=$?
# Each condition writes its own summary; no human gate between trials.
"$PYTHON" tuning/summarise.py --plan lr-rank-v2 --run-root "$TUNING_RUN_ROOT" \
    --model-size "$MODEL_SIZE" --condition "$CONDITION" \
    --output "$TUNING_RUN_ROOT/summaries/$MODEL_SIZE-$CONDITION.json"
exit "$AGENT_STATUS"
