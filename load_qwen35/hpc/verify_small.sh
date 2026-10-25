#!/usr/bin/env bash
# GPU stage for 2B and 9B. Separate Python processes guarantee that all memory
# from the 2B smoke test is released before 9B is loaded.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/nethome/honzhou/thesis_zone/load_qwen35}
QWEN35_PYTHON=${QWEN35_PYTHON:-/nethome/honzhou/miniconda3/envs/thesis/bin/python}
REPORT_DIR=${QWEN35_REPORT_DIR:-$PROJECT_DIR/architecture_reports}
CACHE_DIR=${QWEN35_CACHE_DIR:-/scratch/common_models/HuggingFace}

if [[ ! -x "$QWEN35_PYTHON" ]]; then
    echo "Missing Python environment: $QWEN35_PYTHON" >&2
    exit 1
fi

export HF_HOME=/scratch/hongxuzhou/huggingface_cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$QWEN35_PYTHON" "$PROJECT_DIR/load_3models" verify --model 2b \
    --cache-dir "$CACHE_DIR" --report-dir "$REPORT_DIR"
"$QWEN35_PYTHON" "$PROJECT_DIR/load_3models" verify --model 9b \
    --cache-dir "$CACHE_DIR" --report-dir "$REPORT_DIR"
