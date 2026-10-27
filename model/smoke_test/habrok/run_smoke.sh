#!/usr/bin/env bash
# Run one Qwen3.5 LoRA smoke test on an allocated Habrok A100.
set -euo pipefail

MODEL_SIZE=${1:?usage: run_smoke.sh MODEL_SIZE CONDITION [RUN_ID]}
CONDITION=${2:?usage: run_smoke.sh MODEL_SIZE CONDITION [RUN_ID]}
RUN_ID=${3:-${SLURM_JOB_ID:-manual}-$(date -u +%Y%m%dT%H%M%SZ)}

case "$MODEL_SIZE" in
    2B)
        MODEL_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc
        ;;
    9B)
        MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
        ;;
    *)
        echo "MODEL_SIZE must be 2B or 9B on Habrok (got: $MODEL_SIZE)" >&2
        exit 2
        ;;
esac

case "$CONDITION" in
    clean|repair) ;;
    *)
        echo "CONDITION must be clean or repair (got: $CONDITION)" >&2
        exit 2
        ;;
esac

HABROK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SMOKE_DIR=$(cd "$HABROK_DIR/.." && pwd)
SCRIPT=$SMOKE_DIR/qwen35_lora_smoke_test.py
PYTHON=${PYTHON:-$HOME/miniconda3/envs/thesis/bin/python}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/huggingface/hub}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH_ROOT/huggingface/datasets}
REVISION_DIR=${REVISION_DIR:-$SCRATCH_ROOT/thesis/qwen35_lora_smoke/revisions}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_ROOT/thesis/qwen35_lora_smoke/$MODEL_SIZE-$CONDITION-$RUN_ID}
DATASET_REVISION_FILE=$REVISION_DIR/repair-sbn.revision

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing thesis Python interpreter: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "Missing smoke-test script: $SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$DATASET_REVISION_FILE" ]]; then
    echo "Missing dataset revision file: $DATASET_REVISION_FILE" >&2
    echo "Run habrok/prepare.sh first" >&2
    exit 1
fi

DATASET_REVISION=$(<"$DATASET_REVISION_FILE")
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$(dirname "$OUTPUT_DIR")"

export HF_HUB_CACHE HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

echo "host=$(hostname)"
echo "model_size=$MODEL_SIZE condition=$CONDITION run_id=$RUN_ID"
echo "model_revision=$MODEL_REVISION"
echo "dataset_revision=$DATASET_REVISION"
echo "script=$SCRIPT"
echo "output_dir=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

exec "$PYTHON" "$SCRIPT" \
    --model-size "$MODEL_SIZE" \
    --model-revision "$MODEL_REVISION" \
    --condition "$CONDITION" \
    --dataset-revision "$DATASET_REVISION" \
    --optimizer-steps 30 \
    --gradient-accumulation-steps 4 \
    --micro-batch-size 1 \
    --examples-per-variant 32 \
    --max-length 512 \
    --model-cache "$HF_HUB_CACHE" \
    --dataset-cache "$HF_DATASETS_CACHE" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0
