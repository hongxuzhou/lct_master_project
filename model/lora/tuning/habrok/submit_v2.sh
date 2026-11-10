#!/usr/bin/env bash
# Preview by default. Add --check or --submit explicitly.
set -euo pipefail
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$LORA_DIR/tuning/habrok/environment_v2.sh"
exec "$PYTHON" "$LORA_DIR/tuning/submit.py" --cluster habrok --model-size 2B "$@"
