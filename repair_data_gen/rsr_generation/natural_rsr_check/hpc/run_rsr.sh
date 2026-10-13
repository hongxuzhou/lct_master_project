#!/bin/bash
# One CPU job: normalise both sides, score them with both models, then the
# lexical features (todo §3 B1, §4). Writes into natural_rsr_check/:
#   swbd/lm_items_real.tsv, swbd/lm_requests_real.tsv
#   lm/lm_items_synth.tsv,  lm/lm_requests_synth.tsv
#   swbd/scores_<model>_real.tsv, lm/scores_<model>_synth.tsv
#   swbd/lexical_real.tsv, lm/lexical_synth.tsv
# Controlled by env vars from the .sub `environment` line:
#   GPT2_DIR   (optional) the cluster's GPT-2 small, see setup log
#   ONLY       (optional) "pythia" or "gpt2": score with one model only
set -e

NETHOME=/nethome/honzhou
SCR=/scratch/hongxuzhou
VENV=$NETHOME/venvs/rsr
CODE=$NETHOME/thesis_zone/rsr_generation/natural_rsr_check
RES=$SCR/resources
PYTHIA_DIR=/scratch/common_models/pythia-160m-deduped
GPT2_DIR=${GPT2_DIR:-/scratch/common_models/gpt2}

export HF_HOME=$SCR/huggingface_cache
export HF_HUB_OFFLINE=1
export NLTK_DATA=$NETHOME/nltk_data
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PATH=$VENV/bin:$PATH

echo "script: $(md5sum "$0")"   # compare with `md5 hpc/run_rsr.sh` locally
cd "$CODE"
md5sum build_sets.py lm_items.py score_lm.py lexical.py synth_typed.tsv \
    swbd/sentences.tsv swbd/disfluencies.tsv swbd/real_set.tsv

echo "== normalise"
python lm_items.py --strict

echo "== score"
MODELS=()
[ "$ONLY" != gpt2 ] && MODELS+=("pythia-160m-deduped=$PYTHIA_DIR")
[ "$ONLY" != pythia ] && MODELS+=("gpt2=$GPT2_DIR")
for entry in "${MODELS[@]}"; do
    name=${entry%%=*}
    dir=${entry#*=}
    [ -f "$dir/config.json" ] || { echo "no model at $dir (see setup log)"; exit 1; }
    python score_lm.py --model "$dir" --name "$name" \
        --requests swbd/lm_requests_real.tsv --out "swbd/scores_${name}_real.tsv"
    python score_lm.py --model "$dir" --name "$name" \
        --requests lm/lm_requests_synth.tsv --out "lm/scores_${name}_synth.tsv"
done

echo "== lexical"
python lexical.py --glove "$RES/glove.6B.300d.txt" \
    --subtlex "$(ls "$RES"/subtlexus/*.txt | head -1)"

echo "done"
