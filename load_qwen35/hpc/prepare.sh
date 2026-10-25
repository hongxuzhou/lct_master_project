#!/usr/bin/env bash
# CPU stage: download all checkpoints, then inspect their text architectures on
# the meta device. This job deliberately requests no GPU.
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/nethome/honzhou/thesis_zone/load_qwen35}
QWEN35_PYTHON=${QWEN35_PYTHON:-/nethome/honzhou/miniconda3/envs/thesis/bin/python}
REPORT_DIR=${QWEN35_REPORT_DIR:-$PROJECT_DIR/architecture_reports}
CACHE_DIR=${QWEN35_CACHE_DIR:-/scratch/common_models/HuggingFace}

if [[ ! -x "$QWEN35_PYTHON" ]]; then
    echo "Missing Python environment: $QWEN35_PYTHON" >&2
    echo "Create the Qwen3.5 venv described in hpc/README.md first." >&2
    exit 1
fi

export HF_HOME=/scratch/hongxuzhou/huggingface_cache
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
umask 022

mkdir -p "$REPORT_DIR"

"$QWEN35_PYTHON" "$PROJECT_DIR/load_3models" download --model all \
    --cache-dir "$CACHE_DIR" --report-dir "$REPORT_DIR"
"$QWEN35_PYTHON" "$PROJECT_DIR/load_3models" inspect --model all \
    --cache-dir "$CACHE_DIR" --report-dir "$REPORT_DIR"
