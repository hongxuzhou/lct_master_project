#!/usr/bin/env bash
# Both schedulers call this worker inside a GPU allocation. Reuse the existing
# tuning environment, cached model revisions and W&B login on each cluster.
set -euo pipefail
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CLUSTER=${1:?expected cluster}
case "$CLUSTER" in habrok|lst) ;; *) exit 2 ;; esac
source "$LORA_DIR/tuning/$CLUSTER/environment_v2.sh"
export FORMAL_RUN_ROOT=${FORMAL_RUN_ROOT:-"$(dirname "$DATASET_DIR")/qwen35_lora_formal"}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONHASHSEED=${4:?expected seed}
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
if [[ "$CLUSTER" == habrok ]]; then
    JOB_ID=${SLURM_JOB_ID:?expected a Slurm allocation}
    set -- "$@" "$JOB_ID"
    # Triton compiles GPU kernels on demand, including during the first
    # backward pass. Give each Slurm job its own cache so concurrent jobs
    # don't read an incomplete or stale compiled kernel from ~/.triton.
    CACHE_BASE=${SLURM_TMPDIR:-"$FORMAL_RUN_ROOT/triton-cache"}
    mkdir -p "$CACHE_BASE"
    export TRITON_CACHE_DIR
    TRITON_CACHE_DIR=$(mktemp -d "$CACHE_BASE/qwen35-${JOB_ID}.XXXXXX")
    echo "Triton cache: $TRITON_CACHE_DIR"
fi
# Training writes the selected PEFT adapter and tokenizer to scratch.
# Test inference and Hub upload are separate jobs/steps after selection.
exec "$PYTHON" "$LORA_DIR/formal/run.py" "$@"
