#!/usr/bin/env bash
# Verify pinned LST caches and run CPU tokenization preflight.
set -euo pipefail

LST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SMOKE_DIR=$(cd "$LST_DIR/.." && pwd)
PYTHON=${PYTHON:-/nethome/honzhou/miniconda3/envs/thesis/bin/python}
HF_CLI=${HF_CLI:-$(dirname "$PYTHON")/hf}
MANIFEST=$SMOKE_DIR/revisions.json
PREFLIGHT=$SMOKE_DIR/preflight_tokenization.py
MODEL_CACHE=${MODEL_CACHE:-/scratch/common_models/HuggingFace}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/scratch/hongxuzhou/huggingface_cache/datasets}

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python interpreter: $PYTHON" >&2
    exit 1
fi
if [[ ! -x "$HF_CLI" ]]; then
    echo "Missing Hugging Face CLI: $HF_CLI" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" || ! -f "$PREFLIGHT" ]]; then
    echo "Incomplete smoke-test checkout under $SMOKE_DIR" >&2
    exit 1
fi
if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "Missing shared model cache: $MODEL_CACHE" >&2
    exit 1
fi

mkdir -p "$HF_DATASETS_CACHE"
export HF_DATASETS_CACHE
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/nethome/honzhou/.cache/huggingface/token}
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)"
echo "code_sha256=$(sha256sum "$SMOKE_DIR/qwen35_lora_smoke_test.py" | cut -d' ' -f1)"
"$HF_CLI" auth whoami

DATASET_REVISION=$(
    "$PYTHON" -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["dataset"]["revision"])' \
        "$MANIFEST"
)
"$PYTHON" -c \
    'import sys; from collections import Counter; from datasets import load_dataset; cache, revision = sys.argv[1:]; ds = load_dataset("Shrikes/repair-sbn", revision=revision, cache_dir=cache); assert list(ds) == ["train"]; print({"revision": revision, "rows": len(ds["train"]), "variants": dict(Counter(ds["train"]["variant"]))})' \
    "$HF_DATASETS_CACHE" "$DATASET_REVISION"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
for MODEL_SIZE in 2B 9B 27B; do
    "$PYTHON" "$PREFLIGHT" \
        --model-size "$MODEL_SIZE" \
        --max-length 512 \
        --model-cache "$MODEL_CACHE" \
        --dataset-cache "$HF_DATASETS_CACHE" \
        --manifest "$MANIFEST"
done

echo "LST preparation and tokenization preflight passed"
