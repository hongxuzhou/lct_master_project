#!/usr/bin/env bash
# Safe named entry point: never falls back to the 30-step canary.
set -euo pipefail
HABROK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -ne 0 ]]; then
    echo "usage: bash submit_full_2b.sh" >&2
    exit 2
fi
if [[ ! -r "$HABROK_DIR/../prompts/sbn_v2/protocol.json" ]]; then
    echo "Missing SBN v2 protocol; upload the complete lora directory" >&2
    exit 1
fi
exec bash "$HABROK_DIR/submit.sh" "$HABROK_DIR/full_2b.sbatch"
