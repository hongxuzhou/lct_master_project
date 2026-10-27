#!/usr/bin/env bash
# One HTCondor/manual smoke-test run for one Qwen3.5 size and one target condition.
set -euo pipefail

MODEL_SIZE=${1:?usage: qwen35_lora_smoke_test.sh MODEL_SIZE CONDITION [RUN_ID]}
CONDITION=${2:?usage: qwen35_lora_smoke_test.sh MODEL_SIZE CONDITION [RUN_ID]}
RUN_ID=${3:-manual-$(date -u +%Y%m%dT%H%M%SZ)-$$}

case "$MODEL_SIZE" in
    2B|9B|27B) ;;
    *)
        echo "MODEL_SIZE must be one of: 2B, 9B, 27B (got: $MODEL_SIZE)" >&2
        exit 2
        ;;
esac

case "$CONDITION" in
    clean|repair) ;;
    *)
        echo "CONDITION must be one of: clean, repair (got: $CONDITION)" >&2
        exit 2
        ;;
esac

PYTHON=/nethome/honzhou/miniconda3/envs/thesis/bin/python
SCRIPT=/nethome/honzhou/thesis_zone/model/smoke_test/qwen35_lora_smoke_test.py
MODEL_CACHE=/scratch/common_models/HuggingFace
HF_CACHE=/scratch/hongxuzhou/huggingface_cache
OUTPUT_DIR=/scratch/hongxuzhou/thesis/qwen35_lora_smoke/${MODEL_SIZE}-${CONDITION}-${RUN_ID}

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python interpreter: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "Missing smoke-test script: $SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "Missing offline model cache: $MODEL_CACHE" >&2
    exit 1
fi

mkdir -p "$HF_CACHE/datasets" "$(dirname "$OUTPUT_DIR")"

export HF_HOME="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "host=$(hostname)"
echo "model_size=$MODEL_SIZE condition=$CONDITION run_id=$RUN_ID"
echo "script=$SCRIPT"
echo "output_dir=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

exec "$PYTHON" "$SCRIPT" \
    --model-size "$MODEL_SIZE" \
    --condition "$CONDITION" \
    --optimizer-steps 30 \
    --gradient-accumulation-steps 4 \
    --micro-batch-size 1 \
    --examples-per-variant 32 \
    --max-length 512 \
    --model-cache "$MODEL_CACHE" \
    --dataset-cache "$HF_DATASETS_CACHE" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0
