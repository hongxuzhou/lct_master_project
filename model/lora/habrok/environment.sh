#!/usr/bin/env bash
# Shared defaults for login-shell submission and compute-node execution.
export PYTHON=${PYTHON:-$HOME/miniconda3/envs/thesis-opt/bin/python}
export MODEL_CACHE=${MODEL_CACHE:-/scratch/$USER/huggingface/hub}
export DATASET_DIR=${DATASET_DIR:-/scratch/$USER/thesis/repair-sbn-9b7d8812}
export RUN_ROOT=${RUN_ROOT:-/scratch/$USER/thesis/qwen35_lora_dev}
export WANDB_MODE=${WANDB_MODE:-online}
