#!/usr/bin/env bash
set -euo pipefail
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$LORA_DIR/tuning/lst/environment_uploaded_v2.sh"
exec bash "$LORA_DIR/tuning/agent_v2.sh" lst "$@"
