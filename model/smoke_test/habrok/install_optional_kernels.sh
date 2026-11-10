#!/usr/bin/env bash
# Clone the validated Habrok environment and add the optional Qwen3.5 kernels.
set -euo pipefail

SOURCE_ENV=${SOURCE_ENV:-thesis}
TARGET_ENV=${TARGET_ENV:-thesis-kernels}
CONDA_BASE=${CONDA_BASE:-$HOME/miniconda3}
REUSE_EXISTING=0

if [[ ${1:-} == "--reuse-existing" ]]; then
    REUSE_EXISTING=1
elif [[ $# -ne 0 ]]; then
    echo "usage: install_optional_kernels.sh [--reuse-existing]" >&2
    exit 2
fi

CONDA=$CONDA_BASE/bin/conda
SOURCE_PYTHON=$CONDA_BASE/envs/$SOURCE_ENV/bin/python
TARGET_PYTHON=$CONDA_BASE/envs/$TARGET_ENV/bin/python
HABROK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -x "$CONDA" ]]; then
    echo "Missing conda executable: $CONDA" >&2
    exit 1
fi
if [[ ! -x "$SOURCE_PYTHON" ]]; then
    echo "Missing validated source environment: $SOURCE_PYTHON" >&2
    exit 1
fi

if [[ -e "$CONDA_BASE/envs/$TARGET_ENV" ]]; then
    if [[ $REUSE_EXISTING -ne 1 ]]; then
        echo "Target environment already exists: $CONDA_BASE/envs/$TARGET_ENV" >&2
        echo "Inspect it first, then rerun with --reuse-existing if it is the intended clone." >&2
        exit 1
    fi
else
    "$CONDA" create --yes --name "$TARGET_ENV" --clone "$SOURCE_ENV"
fi

if [[ ! -x "$TARGET_PYTHON" ]]; then
    echo "Missing target interpreter after clone: $TARGET_PYTHON" >&2
    exit 1
fi

"$TARGET_PYTHON" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.14.1+cu130",
    "transformers": "5.20.2",
    "peft": "0.21.2",
    "datasets": "5.1.0",
}
actual = {name: version(name) for name in expected}
print("cloned core versions:", actual)
if actual != expected:
    raise SystemExit(f"refusing to modify clone with unexpected core versions: {actual}")
PY

"$TARGET_PYTHON" -m pip install --upgrade ninja packaging setuptools wheel
"$TARGET_PYTHON" -m pip install --no-build-isolation "causal-conv1d==1.7.0"
"$TARGET_PYTHON" -m pip install \
    "fla-core[cuda]==0.5.2" \
    "flash-linear-attention[cuda]==0.5.2"
"$TARGET_PYTHON" -m pip check
"$TARGET_PYTHON" "$HABROK_DIR/verify_optional_kernels.py"

echo "Optional-kernel environment is ready for a GPU validation: $TARGET_ENV"
echo "Submit: sbatch qwen35_lora_kernel_check.sbatch"
