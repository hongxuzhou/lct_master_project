#!/usr/bin/env bash
# DAG parent: CPU preparation and metadata registration, not GPU training.
set -euo pipefail
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$LORA_DIR/tuning/lst/environment_uploaded_v2.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
cd "$LORA_DIR"
for MODEL_SIZE in 9B 27B; do
    "$PYTHON" preflight.py --protocol protocols/fluent_groups_v1.published.json \
        --model-sizes "$MODEL_SIZE" --dataset-dir "$DATASET_DIR" --model-cache "$MODEL_CACHE" \
        --output "$DATASET_DIR/preflight-v2-$MODEL_SIZE-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
    REGISTRY="tuning/registrations/lr-rank-v2-$MODEL_SIZE.json"
    if [[ ! -e "$REGISTRY" ]]; then
        "$PYTHON" tuning/register_sweeps.py --plan lr-rank-v2 --model-size "$MODEL_SIZE" \
            --create --registry "$REGISTRY"
    fi
    for CONDITION in C1 C2 C3; do
        "$PYTHON" tuning/registry.py "$REGISTRY" "$MODEL_SIZE" "$CONDITION"
    done
done
echo 'Preparation complete. DAGMan will submit the six GPU agents automatically.'
