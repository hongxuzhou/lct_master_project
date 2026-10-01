#!/bin/bash
# One-time setup for the open-model runs, as a CPU HTCondor job (the submit
# node is not for environment builds). No conda, no Docker:
#   - uv installs a standalone Python 3.12 (system Python is 3.9.21; vLLM
#     needs >= 3.10) and a venv with vLLM + nltk;
#   - WordNet for wn_lookup.py;
#   - the model weights, unless /scratch/common_models already has them;
#   - a report at the end: versions, and whether this vLLM knows the model's
#     architecture and has a tool-call / reasoning parser for it.
set -e

NETHOME=/nethome/honzhou
SCR=/scratch/hongxuzhou
VENV=$NETHOME/venvs/vllm
MODEL_ID=meta-models/Muse-Glimmer-30B
# Cluster rule: every model lives under /scratch/common_models, shared by all
# users. Match the directory name to how the others there are named.
MODEL_DIR=/scratch/common_models/Muse-Glimmer-30B
ARCH=MuseGlimmerForConditionalGeneration

export UV_INSTALL_DIR=$NETHOME/.local/bin
export UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR=$SCR/uv_cache
export UV_PYTHON_INSTALL_DIR=$NETHOME/.uv_python
export HF_HOME=$SCR/huggingface_cache
export PATH=$UV_INSTALL_DIR:$PATH

echo "== uv"
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

echo "== venv"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# The nodes' driver supports CUDA 12.8: pin the torch build to it, or a newer
# default wheel (cu13x) installs fine here and then fails on the GPU.
uv pip install --python "$VENV/bin/python" --torch-backend=cu128 vllm nltk

echo "== WordNet"
"$VENV/bin/python" -m nltk.downloader -d "$NETHOME/nltk_data" wordnet

echo "== weights"
echo "-- already in /scratch/common_models (any copy here can be used instead):"
ls -d /scratch/common_models/*[Gg]limmer* 2>/dev/null || echo "   none"
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "$MODEL_DIR exists, not downloading."
elif [ -w /scratch/common_models ]; then
    # Shared directory: readable by everyone, not only by me.
    umask 022
    mkdir -p "$MODEL_DIR"
    "$VENV/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_DIR"
    rm -rf "$MODEL_DIR/.cache"
    chmod -R a+rX "$MODEL_DIR"
else
    echo "No write access to /scratch/common_models: ask the admins to add $MODEL_ID."
fi

echo "== report"
"$VENV/bin/python" - <<EOF
import torch, transformers, vllm
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
print("transformers", transformers.__version__, "| vllm", vllm.__version__)
from vllm import ModelRegistry
print("$ARCH supported:", "$ARCH" in ModelRegistry.get_supported_archs())
EOF
echo "-- parser names mentioning muse/glimmer/atem (empty = none; use TOOL_MODE=text):"
"$VENV/bin/vllm" serve --help=all 2>/dev/null \
    | grep -io '[a-z0-9_]*\(muse\|glimmer\|atem\)[a-z0-9_]*' | sort -u || true
