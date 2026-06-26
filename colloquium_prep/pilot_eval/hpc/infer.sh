#!/bin/bash
# Stage 1 inference wrapper for HTCondor.
# Controlled by env vars passed from the .sub `environment` line:
#   OUTFILE  (required) output parquet name under $OUT
#   LIMIT    (optional) smoke-test: only first N long rows
#   BATCH    (optional) batch size, default 8
#   MAXNEW   (optional) max new tokens, default 512
# 4-bit is intentionally NOT used (the >40GB GPU requirement gives us enough VRAM
# for 9B bf16 inference), so --use-4bit is never passed.
set -e

source /nethome/honzhou/lora_gemma2/bin/activate

# Keep all HF downloads (adapter, dataset) on /scratch (no backup, unlimited).
export HF_HOME=/scratch/hongxuzhou/huggingface_cache
export TOKENIZERS_PARALLELISM=false

SCRIPT=/nethome/honzhou/thesis_zone/pilot/script/pilot_eval
OUT=/scratch/hongxuzhou/thesis/pilot
mkdir -p "$OUT"

LIM=""
[ -n "$LIMIT" ] && LIM="--limit $LIMIT"

python "$SCRIPT/run_inference.py" \
    --dataset Shrikes/self_repair_parsing_pilot_data \
    --output "$OUT/${OUTFILE:-preds_long.parquet}" \
    --batch-size "${BATCH:-8}" \
    --max-new-tokens "${MAXNEW:-512}" \
    $LIM
