#!/usr/bin/env bash
# Source on the submit host and worker; explicit overrides remain supported.
export PYTHON=${PYTHON:-/nethome/honzhou/miniconda3/envs/thesis-qwen35-opt/bin/python}
export MODEL_CACHE=${MODEL_CACHE:-/scratch/common_models/HuggingFace}
export DATASET_DIR=${DATASET_DIR:-/scratch/hongxuzhou/thesis/repair-sbn-d1ed13ec}
export TUNING_RUN_ROOT=${TUNING_RUN_ROOT:-/scratch/hongxuzhou/thesis/qwen35_lora_tuning/lr-rank-v2}
