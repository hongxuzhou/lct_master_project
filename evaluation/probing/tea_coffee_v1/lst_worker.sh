#!/usr/bin/env bash
# Called only by the A100 Condor allocation; no environment activation.
set -euo pipefail
root=${1:?expected deployment root}
job_id=${2:?expected ClusterId-ProcId}
case "$job_id" in *[!0-9-]*|'') exit 2 ;; esac
probe_python=/nethome/honzhou/miniconda3/envs/thesis-qwen35-opt/bin/python
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export HF_TOKEN_PATH=/nethome/honzhou/.cache/huggingface/token
export MODEL_CACHE=/scratch/common_models/HuggingFace
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=120
attempt=/scratch/hongxuzhou/thesis/restart-probe/tea_coffee_v1/job-$job_id
mkdir -p "$(dirname "$attempt")"
mkdir "$attempt"
export TRITON_CACHE_DIR="$attempt/triton-cache"
mkdir "$TRITON_CACHE_DIR"
trap 'code=$?; printf "Worker exit=%s; results=%s\n" "$code" "$attempt"; exit "$code"' EXIT
probe="$root/evaluation/probing/tea_coffee_v1"
cd "$root"
echo "Compute job $job_id on $(hostname); results: $attempt"
"$probe_python" "$probe/lst_preflight.py" --root "$root" \
    --model-cache "$MODEL_CACHE" --output "$attempt/preflight.json"
for seed in 24 42 4224; do
    echo "Starting seed $seed"
    "$probe_python" "$probe/run.py" --manifest "$root/frozen-9B-C3.json" \
        --seed "$seed" --adapter-store /scratch/hongxuzhou/thesis/repair-sbn-formal-adapters \
        --model-cache "$MODEL_CACHE" --output "$attempt/seed$seed"
    echo "Completed seed $seed"
done
echo "COMPLETE: 3 seeds, 48 predictions; results: $attempt"
