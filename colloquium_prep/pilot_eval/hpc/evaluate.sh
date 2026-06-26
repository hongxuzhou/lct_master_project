#!/bin/bash
# Stage 2 smatch++ scoring wrapper for HTCondor. CPU only.
#   INFILE   (optional) predictions parquet under $OUT, default preds_long.parquet
#   OUTFILE  (optional) scored parquet under $OUT, default preds_long_scored.parquet
set -e

source /nethome/honzhou/lora_gemma2/bin/activate

SCRIPT=/nethome/honzhou/thesis_zone/pilot/script/pilot_eval
OUT=/scratch/hongxuzhou/thesis/pilot
GOLD=/nethome/honzhou/thesis_zone/pilot/data/pilot_dataset.parquet

python "$SCRIPT/evaluate_smatchpp.py" \
    -i "$OUT/${INFILE:-preds_long.parquet}" \
    -o "$OUT/${OUTFILE:-preds_long_scored.parquet}" \
    --gold-file "$GOLD" --gold-file-gold-col mr \
    --solver ilp
