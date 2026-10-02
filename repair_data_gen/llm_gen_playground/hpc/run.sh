#!/bin/bash
# One HTCondor job = one vLLM server on this node's GPU + run_llm.py against it.
# The server listens on 127.0.0.1 only and is stopped when the job ends, so no
# port forwarding is needed.
# Controlled by env vars from the .sub `environment` line:
#   TAG        (optional) results file suffix, e.g. smoke
#   SAMPLE     (optional) sample TSV, relative to the playground; default
#              pinned_sample.tsv (round 1)
#   LIMIT      (optional) only the first N sample rows
#   WORKERS    (optional) parallel items, default 8
#   REASONING  (optional) low/medium/high/xhigh, default high
#   TOOL_MODE  (optional) text (default) or auto; auto needs TOOL_PARSER
#   TOOL_PARSER, REASONING_PARSER (optional) vLLM parser names, see setup log
set -e

NETHOME=/nethome/honzhou
SCR=/scratch/hongxuzhou
VENV=$NETHOME/venvs/vllm
CODE=$NETHOME/thesis_zone/llm_gen_playground
MODEL_ID=meta-models/Muse-Glimmer-30B
MODEL_PATH=${MODEL_PATH:-/scratch/common_models/Muse-Glimmer-30B}
# Room for the ~22k-token prompt, earlier turns' reasoning, and a 32768-token
# turn (run_llm.py --max-tokens). 65536 in the first Muse Glimmer round.
MAX_LEN=${MAX_LEN:-98304}
[ -f "$MODEL_PATH/config.json" ] || { echo "no model at $MODEL_PATH (see setup log)"; exit 1; }

export HF_HOME=$SCR/huggingface_cache
export HF_HUB_OFFLINE=1
export NLTK_DATA=$NETHOME/nltk_data
export TOKENIZERS_PARALLELISM=false
# FlashInfer's top-k/top-p sampler is JIT-compiled on first use with the
# node's /usr/local/cuda nvcc (12.8), which rejects `--compress-mode=size`
# (FlashInfer assumes nvcc >= 12.9, after torch cu129). The first smoke run
# failed there. vLLM's PyTorch sampler needs no compilation; for one request
# at a time the speed difference is negligible.
export VLLM_USE_FLASHINFER_SAMPLER=0
# wn_lookup.py is run with the venv's Python (sys.executable), which has nltk.
export PATH=$VENV/bin:$PATH

echo "script: $(md5sum "$0")"   # compare with `md5 hpc/run.sh` locally

# HTCondor hands out GPUs by UUID in CUDA_VISIBLE_DEVICES; rename them to
# indices, as the LST wiki's test_gpu.sh does (vLLM expects indices).
echo "condor: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
new_devices=""
IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
for id in "${gpu_ids[@]}"; do
    new_devices=${new_devices}$(nvidia-smi -L | grep "$id" | sed -E "s/^GPU ([0-9]+):.*$/\1/"),
done
export CUDA_VISIBLE_DEVICES=${new_devices%?}
echo "now:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")

PARSER_FLAGS=()
[ -n "$TOOL_PARSER" ] && PARSER_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && PARSER_FLAGS+=(--reasoning-parser "$REASONING_PARSER")

# Text only: image limit 0 keeps the vision encoder's profiling memory free
# for KV cache. Prefix caching: all items share the ~22k-token system prompt.
vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    "${PARSER_FLAGS[@]}" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

echo "waiting for vLLM on port $PORT ..."
for i in $(seq 1 240); do      # 240 x 15 s = 1 h: loading 60 GB from /scratch is slow
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "vLLM exited during start-up"; exit 1; }
    sleep 15
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM not up after 1 h"; exit 1; }
echo "vLLM up after ~$((i * 15)) s"

ARGS=(run --backend openai --base-url "http://127.0.0.1:$PORT/v1"
      --model "$MODEL_ID"
      --tool-mode "${TOOL_MODE:-text}"
      --reasoning "${REASONING:-high}"
      --workers "${WORKERS:-8}")
[ -n "$TAG" ] && ARGS+=(--tag "$TAG")
[ -n "$SAMPLE" ] && ARGS+=(--sample "$SAMPLE")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")

cd "$CODE"
python run_llm.py "${ARGS[@]}"
