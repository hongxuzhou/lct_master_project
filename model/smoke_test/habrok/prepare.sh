#!/usr/bin/env bash
# Cache one pinned Qwen3.5 checkpoint and the pinned dataset on Habrok.
set -euo pipefail

MODEL_SIZE=${1:?usage: prepare.sh MODEL_SIZE (2B or 9B)}

case "$MODEL_SIZE" in
    2B|9B) ;;
    *)
        echo "MODEL_SIZE must be 2B or 9B (got: $MODEL_SIZE)" >&2
        exit 2
        ;;
esac

PYTHON=${PYTHON:-$HOME/miniconda3/envs/thesis/bin/python}
HABROK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SMOKE_DIR=$(cd "$HABROK_DIR/.." && pwd)
MANIFEST=$SMOKE_DIR/revisions.json
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}
HF_HUB_CACHE=${HF_HUB_CACHE:-$SCRATCH_ROOT/huggingface/hub}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH_ROOT/huggingface/datasets}

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing thesis Python interpreter: $PYTHON" >&2
    exit 1
fi
if ! command -v hf >/dev/null 2>&1; then
    echo "Missing hf CLI in PATH; activate the thesis environment first" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing revision manifest: $MANIFEST" >&2
    exit 1
fi

readarray -t PINNED < <(
    "$PYTHON" -c \
        'import json, sys; m=json.load(open(sys.argv[1])); print(m["models"][sys.argv[2]]["id"]); print(m["models"][sys.argv[2]]["revision"]); print(m["dataset"]["revision"])' \
        "$MANIFEST" "$MODEL_SIZE"
)
MODEL_ID=${PINNED[0]}
MODEL_REVISION=${PINNED[1]}
DATASET_REVISION=${PINNED[2]}

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
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

"$PYTHON" -c \
    'import sys; from collections import Counter; from datasets import load_dataset; cache, revision = sys.argv[1:]; ds = load_dataset("Shrikes/repair-sbn", revision=revision, cache_dir=cache); assert list(ds) == ["train"]; print({"revision": revision, "rows": len(ds["train"]), "variants": dict(Counter(ds["train"]["variant"]))})' \
    "$HF_DATASETS_CACHE" "$DATASET_REVISION"

echo "Preparation complete for $MODEL_SIZE"
