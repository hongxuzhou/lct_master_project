#!/usr/bin/env bash
# Pass --model-size 9B (A100 40GB) or 27B (H100 80GB).
set -euo pipefail
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$LORA_DIR/tuning/lst/environment_v2.sh"
exec "$PYTHON" "$LORA_DIR/tuning/submit.py" --cluster lst "$@"
