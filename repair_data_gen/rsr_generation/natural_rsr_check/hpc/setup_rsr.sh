#!/bin/bash
# One-time setup for the naturalness measures (todo §4), as a CPU HTCondor
# job, in the same way as ../../hpc/setup.sh:
#   - uv venv with a CPU torch, transformers, nltk, num2words;
#   - NLTK WordNet and the CMU dictionary;
#   - Pythia-160m-deduped into /scratch/common_models, unless it is there;
#   - GloVe 2014 (glove.6B.300d) and SUBTLEXus into $SCR/resources;
#   - a report: versions, the GPT-2 found in /scratch/common_models, and
#     whether both tokenisers mark word starts the way score_lm.py expects.
set -e
echo "script: $(md5sum "$0")"   # compare with `md5 hpc/setup_rsr.sh` locally

NETHOME=/nethome/honzhou
SCR=/scratch/hongxuzhou
VENV=$NETHOME/venvs/rsr
RES=$SCR/resources
PYTHIA_ID=EleutherAI/pythia-160m-deduped
PYTHIA_DIR=/scratch/common_models/pythia-160m-deduped
# The cluster's GPT-2 small ("the smallest version of GPT-2, with 124M
# parameter"). Check the directory name with `ls /scratch/common_models/`
# and set it here and in run_rsr.sh.
GPT2_DIR=${GPT2_DIR:-/scratch/common_models/gpt2}

export UV_INSTALL_DIR=$NETHOME/.local/bin
export UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR=$SCR/uv_cache
export UV_PYTHON_INSTALL_DIR=$NETHOME/.uv_python
export UV_LINK_MODE=copy
export HF_HOME=$SCR/huggingface_cache
export PATH=$UV_INSTALL_DIR:$PATH

echo "== uv"
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

echo "== venv"
# A CPU job is enough for a 160M model, so the CPU torch build: no CUDA
# version questions, and a much smaller venv than the vLLM one.
if ! "$VENV/bin/python" -c 'import torch, transformers, nltk, num2words' 2>/dev/null; then
    rm -rf "$VENV"
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" --torch-backend=cpu \
        torch transformers nltk num2words
fi

echo "== NLTK data"
"$VENV/bin/python" -m nltk.downloader -d "$NETHOME/nltk_data" wordnet cmudict

echo "== Pythia"
echo "-- already in /scratch/common_models:"
ls -d /scratch/common_models/*[Pp]ythia* 2>/dev/null || echo "   none"
if [ -f "$PYTHIA_DIR/config.json" ]; then
    echo "$PYTHIA_DIR exists, not downloading."
elif [ -w /scratch/common_models ]; then
    umask 022                 # shared directory: readable by everyone
    mkdir -p "$PYTHIA_DIR"
    "$VENV/bin/hf" download "$PYTHIA_ID" --local-dir "$PYTHIA_DIR"
    rm -rf "$PYTHIA_DIR/.cache"
    chmod -R a+rX "$PYTHIA_DIR"
else
    echo "No write access to /scratch/common_models: ask the admins to add $PYTHIA_ID."
fi

echo "== GloVe 2014, 6B, 300d"
mkdir -p "$RES"
if [ ! -s "$RES/glove.6B.300d.txt" ]; then
    curl -fL -o "$RES/glove.6B.zip" https://nlp.stanford.edu/data/glove.6B.zip
    unzip -o -j "$RES/glove.6B.zip" glove.6B.300d.txt -d "$RES"
    rm "$RES/glove.6B.zip"
fi

echo "== SUBTLEXus (text version, all 74,286 words)"
if ! ls "$RES"/subtlexus/*.txt >/dev/null 2>&1; then
    curl -fL -o "$RES/subtlexus2.zip" \
        https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip
    unzip -o "$RES/subtlexus2.zip" -d "$RES/subtlexus"
    rm "$RES/subtlexus2.zip"
fi

echo "== report"
"$VENV/bin/python" --version
"$VENV/bin/python" - <<EOF
import torch, transformers, nltk, num2words
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| nltk", nltk.__version__)
EOF
echo "-- GPT-2 candidates in /scratch/common_models:"
ls -d /scratch/common_models/*[Gg][Pp][Tt]* 2>/dev/null || echo "   none"
for M in "$PYTHIA_DIR" "$GPT2_DIR"; do
    if [ -f "$M/config.json" ]; then
        "$VENV/bin/python" - "$M" <<'EOF'
import json, sys
from transformers import AutoTokenizer
m = sys.argv[1]
tok = AutoTokenizer.from_pretrained(m, local_files_only=True)
starts = sum(t.startswith(('Ġ', 'Ċ')) for t in tok.get_vocab())
cfg = json.load(open(f'{m}/config.json'))
print(f'{m}: {cfg.get("model_type")}, layers {cfg.get("n_layer", cfg.get("num_hidden_layers"))}, '
      f'bos {tok.bos_token_id}, {starts} word-start tokens (expect > 10000)')
EOF
    else
        echo "$M: no config.json"
    fi
done
echo "-- resources:"
ls -la "$RES" "$RES/subtlexus"
head -2 "$RES"/subtlexus/*.txt
head -c 120 "$RES/glove.6B.300d.txt"; echo
wc -l "$RES/glove.6B.300d.txt"     # expect 400000
