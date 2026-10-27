#!/usr/bin/env bash
# Cache one Qwen3.5 checkpoint and the smoke-test dataset on Habrok.
set -euo pipefail

MODEL_SIZE=${1:?usage: prepare.sh MODEL_SIZE (2B or 9B)}

case "$MODEL_SIZE" in
    2B)
        MODEL_ID=Qwen/Qwen3.5-2B
        MODEL_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc
        ;;
    9B)
        MODEL_ID=Qwen/Qwen3.5-9B
        MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
        ;;
    *)
        echo "MODEL_SIZE must be 2B or 9B (got: $MODEL_SIZE)" >&2
        exit 2
        ;;
esac

PYTHON=${PYTHON:-$HOME/miniconda3/envs/thesis/bin/python}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/huggingface/hub}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH_ROOT/huggingface/datasets}
REVISION_DIR=${REVISION_DIR:-$SCRATCH_ROOT/thesis/qwen35_lora_smoke/revisions}

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing thesis Python interpreter: $PYTHON" >&2
    exit 1
fi
if ! command -v hf >/dev/null 2>&1; then
    echo "Missing hf CLI in PATH; activate the thesis environment first" >&2
    exit 1
fi

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$REVISION_DIR"
export HF_HUB_CACHE HF_DATASETS_CACHE
export HF_HUB_DISABLE_TELEMETRY=1

echo "model=$MODEL_ID revision=$MODEL_REVISION"
echo "model_cache=$HF_HUB_CACHE"
echo "dataset_cache=$HF_DATASETS_CACHE"
hf auth whoami
hf download "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --cache-dir "$HF_HUB_CACHE"
hf cache verify "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --cache-dir "$HF_HUB_CACHE" \
    --fail-on-missing-files

DATASET_REVISION=$(
    "$PYTHON" -c \
        'from huggingface_hub import HfApi; print(HfApi().dataset_info("Shrikes/repair-sbn").sha)'
)
printf '%s\n' "$DATASET_REVISION" > "$REVISION_DIR/repair-sbn.revision"

"$PYTHON" -c \
    'import sys; from collections import Counter; from datasets import load_dataset; cache, revision = sys.argv[1:]; ds = load_dataset("Shrikes/repair-sbn", revision=revision, cache_dir=cache); assert list(ds) == ["train"]; print({"revision": revision, "rows": len(ds["train"]), "variants": dict(Counter(ds["train"]["variant"]))})' \
    "$HF_DATASETS_CACHE" "$DATASET_REVISION"

echo "Preparation complete for $MODEL_SIZE"
