#!/usr/bin/env bash
# Run only in a permitted CPU/preparation allocation, never on LST login/submit.
set -euo pipefail
CLUSTER=${1:?usage: prepare_v2.sh CLUSTER SIZE}
MODEL_SIZE=${2:?}
case "$CLUSTER/$MODEL_SIZE" in habrok/2B|lst/9B|lst/27B) ;; *) exit 2;; esac
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$LORA_DIR/tuning/$CLUSTER/environment_v2.sh"
HF_CLI=${HF_CLI:-$(dirname "$PYTHON")/hf}
PROTOCOL=$LORA_DIR/protocols/fluent_groups_v1.published.json
readarray -t PINNED < <("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["models"][sys.argv[2]]; print(m["id"]); print(m["revision"]); print(p["dataset"]["id"]); print(p["dataset"]["revision"])' "$PROTOCOL" "$MODEL_SIZE")
[[ ${#PINNED[@]} == 4 ]] || exit 1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
"$HF_CLI" download "${PINNED[0]}" --revision "${PINNED[1]}" --cache-dir "$MODEL_CACHE"
"$HF_CLI" download "${PINNED[2]}" --repo-type dataset --revision "${PINNED[3]}" \
    --include 'data/train.parquet' --include 'data/validation.parquet' --local-dir "$DATASET_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
REPORT="$DATASET_DIR/preflight-v2-$MODEL_SIZE-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
"$PYTHON" "$LORA_DIR/preflight.py" --protocol "$PROTOCOL" --model-sizes "$MODEL_SIZE" \
    --dataset-dir "$DATASET_DIR" --model-cache "$MODEL_CACHE" --output "$REPORT"
for CONDITION in C1 C2 C3; do
    "$PYTHON" "$LORA_DIR/tuning/run_sweep.py" --plan lr-rank-v2 --dry-run \
        --model-size "$MODEL_SIZE" --condition "$CONDITION" --cluster "$CLUSTER"
done
