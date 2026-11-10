#!/usr/bin/env bash
# Concrete paths for honzhou's uploaded v2 round; avoid inherited v1 defaults.
export PYTHON=/nethome/honzhou/miniconda3/envs/thesis-qwen35-opt/bin/python
export MODEL_CACHE=/scratch/common_models/HuggingFace
export DATASET_DIR=/scratch/hongxuzhou/thesis/repair-sbn-d1ed13ec
export TUNING_RUN_ROOT=/scratch/hongxuzhou/thesis/qwen35_lora_tuning/lr-rank-v2
