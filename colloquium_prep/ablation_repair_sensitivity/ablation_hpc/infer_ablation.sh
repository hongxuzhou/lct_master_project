#!/bin/bash
# Stage 1 inference wrapper for HTCondor — repair-sensitivity ablation.
# Controlled by env vars passed from the .sub `environment` line:
#   OUTFILE  (required) output parquet name under $OUT
#   LIMIT    (optional) smoke-test: only first N long rows
#   BATCH    (optional) batch size, default 16
#   MAXNEW   (optional) max new tokens, default 200
# No LoRA adapter (ablation design decision 1) and no --use-4bit: the same
# >40GB GPU requirement as the main pilot run gives ample VRAM for 9B bf16.
set -e

source /nethome/honzhou/lora_gemma2/bin/activate

export HF_HOME=/scratch/hongxuzhou/huggingface_cache
export TOKENIZERS_PARALLELISM=false

SCRIPT=/nethome/honzhou/thesis_zone/pilot/script/ablation_repair_sensitivity
OUT=/scratch/hongxuzhou/thesis/ablation
mkdir -p "$OUT"

LIM=""
[ -n "$LIMIT" ] && LIM="--limit $LIMIT"

python "$SCRIPT/run_inference_ablation.py" \
    --dataset Shrikes/self_repair_parsing_pilot_data \
    --output "$OUT/${OUTFILE:-preds_ablation.parquet}" \
    --batch-size "${BATCH:-16}" \
    --max-new-tokens "${MAXNEW:-200}" \
    $LIM
