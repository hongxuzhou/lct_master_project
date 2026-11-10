#!/usr/bin/env bash
# Source in the existing environment; do not install packages here.
export PYTHON=${PYTHON:-$HOME/miniconda3/envs/thesis-opt/bin/python}
export MODEL_CACHE=${MODEL_CACHE:-/scratch/$USER/huggingface/hub}
export DATASET_DIR=${DATASET_DIR:-/scratch/$USER/thesis/repair-sbn-d1ed13ec}
export TUNING_RUN_ROOT=${TUNING_RUN_ROOT:-/scratch/$USER/thesis/qwen35_lora_tuning/lr-rank-v2}
