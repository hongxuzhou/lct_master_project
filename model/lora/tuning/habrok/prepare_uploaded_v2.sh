#!/usr/bin/env bash
# Run on a Habrok interactive/preparation node, after staging the supplied data.
set -euo pipefail
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHON=/home1/s5788668/miniconda3/envs/thesis-opt/bin/python
export MODEL_CACHE=/scratch/s5788668/huggingface/hub
export DATASET_DIR=/scratch/s5788668/thesis/repair-sbn-d1ed13ec
export TUNING_RUN_ROOT=/scratch/s5788668/thesis/qwen35_lora_tuning/lr-rank-v2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off
cd "$LORA_DIR"
"$PYTHON" preflight.py --protocol protocols/fluent_groups_v1.published.json \
    --model-sizes 2B --dataset-dir "$DATASET_DIR" --model-cache "$MODEL_CACHE" \
    --output "$DATASET_DIR/preflight-v2-2B-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
REGISTRY=tuning/registrations/lr-rank-v2-2B.json
if [[ ! -e "$REGISTRY" ]]; then
    "$PYTHON" tuning/register_sweeps.py --plan lr-rank-v2 --model-size 2B \
        --create --registry "$REGISTRY"
fi
for CONDITION in C1 C2 C3; do
    "$PYTHON" tuning/registry.py "$REGISTRY" 2B "$CONDITION"
done
echo 'Preparation complete. Return to the login node and run tuning/habrok/submit_v2.sh --submit.'
