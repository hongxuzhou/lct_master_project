#!/usr/bin/env bash
#SBATCH --job-name=qwen35-full-2B
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Three independent full-budget development runs; not formal experiment runs.
set -euo pipefail
LORA_DIR=${LORA_DIR:-$(cd "${SLURM_SUBMIT_DIR:?}/.." && pwd)}
source "$LORA_DIR/habrok/environment.sh"
PROTOCOL="$LORA_DIR/prompts/sbn_v2/protocol.json"
export SEED=17

if [[ ! -x "$PYTHON" || ! -r "$PROTOCOL" ]]; then
    echo "Missing interpreter or full-development protocol: $PYTHON / $PROTOCOL" >&2
    exit 1
fi

# Reuse the validated optional-kernel environment, without installing anything.
"$PYTHON" - <<'PY'
from importlib.metadata import version

expected = {"causal-conv1d": "1.7.0", "fla-core": "0.5.2", "flash-linear-attention": "0.5.2"}
actual = {name: version(name) for name in expected}
print("Habrok optional kernels:", actual)
if actual != expected:
    raise SystemExit(f"unexpected Habrok optional-kernel versions: {actual}")
PY

for CONDITION in C1 C2 C3; do
    RUN_ID="full-dev-habrok-2B-${CONDITION}-s17-sbnv2-${SLURM_JOB_ID:?}"
    echo "Starting $RUN_ID: three epochs, eval/save every 100 steps"
    bash "$LORA_DIR/run.sh" habrok 2B "$CONDITION" "$RUN_ID" \
        --protocol "$PROTOCOL" --purpose calibration --max-steps 0 \
        --eval-every 100 --micro-batch-size 1 --generation-examples 20
done
echo "All three full-development conditions completed. Outputs: $RUN_ROOT"
