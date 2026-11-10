#!/usr/bin/env bash
#SBATCH --job-name=qwen35-tuning
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
LORA_DIR=${LORA_DIR:-$(cd "${SLURM_SUBMIT_DIR:?}/../.." && pwd)}
source "$LORA_DIR/habrok/environment.sh"
export TUNING_RUN_ROOT=${TUNING_RUN_ROOT:-/scratch/$USER/thesis/qwen35_lora_tuning/lr-rank-v1}
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_MODE=online
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
export PATH="$(dirname "$PYTHON"):$PATH"
SWEEP_PATH=${1:?Pass entity/project/sweep-id}
COUNT=${2:-1}
if [[ ! "$COUNT" =~ ^[1-6]$ ]]; then
    echo 'Count must be 1–6' >&2
    exit 1
fi
"$PYTHON" - <<'PY'
from importlib.metadata import version
for package, expected in {'wandb': '0.31.0', 'causal-conv1d': '1.7.0',
                          'fla-core': '0.5.2', 'flash-linear-attention': '0.5.2'}.items():
    if version(package) != expected:
        raise SystemExit(f'Unexpected {package} version')
PY
cd "$LORA_DIR"
wandb agent --count "$COUNT" "$SWEEP_PATH"
