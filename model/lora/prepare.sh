#!/usr/bin/env bash
# Run on a permitted preparation/CPU node with the intended cluster environment.
set -euo pipefail
MODEL_SIZE=${1:?usage: prepare.sh MODEL_SIZE}
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
HF_CLI=${HF_CLI:-hf}
MODEL_CACHE=${MODEL_CACHE:?Set MODEL_CACHE to the existing model cache}
DATASET_DIR=${DATASET_DIR:?Set DATASET_DIR to a new pinned release directory}
readarray -t PINNED < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["models"][sys.argv[2]]; print(m["id"]); print(m["revision"]); print(p["dataset"]["id"]); print(p["dataset"]["revision"])' "$LORA_DIR/protocol.json" "$MODEL_SIZE")
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
"$HF_CLI" download "${PINNED[0]}" --revision "${PINNED[1]}" --cache-dir "$MODEL_CACHE"
"$HF_CLI" download "${PINNED[2]}" --type dataset --revision "${PINNED[3]}" \
    --include 'data/train.parquet' --include 'data/validation.parquet' \
    --include README.md --include manifest.json --include validation.manifest.json \
    --include validation.acceptance.json --local-dir "$DATASET_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PREFLIGHT_REPORT=${PREFLIGHT_REPORT:-$DATASET_DIR/preflight-$MODEL_SIZE-$(date -u +%Y%m%dT%H%M%SZ)-$$.json}
"$PYTHON" "$LORA_DIR/preflight.py" --model-sizes "$MODEL_SIZE" \
    --dataset-dir "$DATASET_DIR" --model-cache "$MODEL_CACHE" --output "$PREFLIGHT_REPORT"
