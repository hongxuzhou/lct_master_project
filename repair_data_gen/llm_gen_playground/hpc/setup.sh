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
# Muse Glimmer (architecture + muse_glimmer tool/reasoning parsers) needs
# vLLM >= 0.28. An unpinned install under --torch-backend=cu128 silently
# resolved to 0.24.0, which has neither: releases since 0.28 are built for
# CUDA 12.9 or 13.0 only. The nodes' driver reports CUDA 12.8; CUDA 12.x
# minor-version compatibility lets a cu129 build run on it, cu130 would not.
# So: the release's cu129 wheel, pinned, with the cu129 torch.
VLLM_VERSION=0.30.0
CU=cu129
WHEEL=https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION%2B$CU-cp38-abi3-manylinux_2_28_x86_64.whl
export UV_LINK_MODE=copy   # cache on /scratch, venv on /nethome: no hardlinks
if [ "$("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)" != "$VLLM_VERSION+$CU" ]; then
    rm -rf "$VENV"         # start clean rather than upgrade a cu128 venv in place
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" --torch-backend=$CU "vllm @ $WHEEL" nltk
fi

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
echo "-- vLLM source files mentioning muse_glimmer (parsers expected; empty = use TOOL_MODE=text):"
grep -rl muse_glimmer "$VENV"/lib/python3.12/site-packages/vllm --include=*.py | sed "s|$VENV/||" || true
echo "-- weights (expect ~60G):"
du -sh "$MODEL_DIR"; ls -la "$MODEL_DIR"
