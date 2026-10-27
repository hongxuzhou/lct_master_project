#!/usr/bin/env bash
# Run one Qwen3.5 LoRA smoke test in an LST HTCondor GPU allocation.
set -euo pipefail

MODEL_SIZE=${1:?usage: run_smoke.sh MODEL_SIZE CONDITION [RUN_ID]}
CONDITION=${2:?usage: run_smoke.sh MODEL_SIZE CONDITION [RUN_ID]}
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

LST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SMOKE_DIR=$(cd "$LST_DIR/.." && pwd)
PYTHON=${PYTHON:-/nethome/honzhou/miniconda3/envs/thesis/bin/python}
SCRIPT=$SMOKE_DIR/qwen35_lora_smoke_test.py
MANIFEST=$SMOKE_DIR/revisions.json
MODEL_CACHE=${MODEL_CACHE:-/scratch/common_models/HuggingFace}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/scratch/hongxuzhou/huggingface_cache/datasets}
FINAL_OUTPUT_DIR=${OUTPUT_DIR:-/scratch/hongxuzhou/thesis/qwen35_lora_smoke/${MODEL_SIZE}-${CONDITION}-${RUN_ID}}

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python interpreter: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
    echo "Missing smoke-test script: $SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing revision manifest: $MANIFEST" >&2
    exit 1
fi
if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "Missing offline model cache: $MODEL_CACHE" >&2
    exit 1
fi

readarray -t REVISIONS < <(
    "$PYTHON" -c \
        'import json, sys; m=json.load(open(sys.argv[1])); print(m["models"][sys.argv[2]]["revision"]); print(m["dataset"]["revision"])' \
        "$MANIFEST" "$MODEL_SIZE"
)
MODEL_REVISION=${REVISIONS[0]}
DATASET_REVISION=${REVISIONS[1]}

mkdir -p "$HF_DATASETS_CACHE" "$(dirname "$FINAL_OUTPUT_DIR")"
if [[ -n "${_CONDOR_SCRATCH_DIR:-}" ]]; then
    RUN_OUTPUT_DIR=$_CONDOR_SCRATCH_DIR/result
else
    RUN_OUTPUT_DIR=$FINAL_OUTPUT_DIR
fi

copy_results() {
    if [[ "$RUN_OUTPUT_DIR" != "$FINAL_OUTPUT_DIR" && -d "$RUN_OUTPUT_DIR" ]]; then
        mkdir -p "$FINAL_OUTPUT_DIR"
        cp -a "$RUN_OUTPUT_DIR"/. "$FINAL_OUTPUT_DIR"/
    fi
}

export HF_DATASETS_CACHE
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

echo "host=$(hostname)"
echo "model_size=$MODEL_SIZE condition=$CONDITION run_id=$RUN_ID"
echo "script=$SCRIPT"
echo "model_revision=$MODEL_REVISION"
echo "dataset_revision=$DATASET_REVISION"
echo "code_sha256=$(sha256sum "$SCRIPT" | cut -d' ' -f1)"
echo "output_dir=$FINAL_OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

RUN_STATUS=0
"$PYTHON" "$SCRIPT" \
    --model-size "$MODEL_SIZE" \
    --model-revision "$MODEL_REVISION" \
    --condition "$CONDITION" \
    --dataset-revision "$DATASET_REVISION" \
    --optimizer-steps 30 \
    --gradient-accumulation-steps 4 \
    --micro-batch-size 1 \
    --examples-per-variant 32 \
    --max-length 512 \
    --model-cache "$MODEL_CACHE" \
    --dataset-cache "$HF_DATASETS_CACHE" \
    --output-dir "$RUN_OUTPUT_DIR" \
    --device cuda:0 || RUN_STATUS=$?

copy_results
exit "$RUN_STATUS"
