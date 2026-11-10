#!/usr/bin/env bash
#SBATCH --job-name=qwen35-dev-2B
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
LORA_DIR=${LORA_DIR:-$(cd "$SLURM_SUBMIT_DIR/.." && pwd)}
source "$LORA_DIR/habrok/environment.sh"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Habrok optional-kernel interpreter: $PYTHON" >&2
    exit 1
fi

"$PYTHON" - <<'PY'
from importlib.metadata import version

expected = {
    "causal-conv1d": "1.7.0",
    "fla-core": "0.5.2",
    "flash-linear-attention": "0.5.2",
}
actual = {name: version(name) for name in expected}
print("Habrok optional kernels:", actual)
if actual != expected:
    raise SystemExit(f"unexpected Habrok optional-kernel versions: {actual}")
PY

export PYTHON
for CONDITION in C1 C2 C3; do
    bash "$LORA_DIR/run.sh" habrok 2B "$CONDITION" "canary-habrok-2B-$CONDITION-s17-$SLURM_JOB_ID" \
        --purpose canary --max-steps 30 --eval-every 10
done
