#!/usr/bin/env bash
set -euo pipefail
LORA_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JOB_ID=${1:?Condor job ID required}
for CONDITION in C1 C2 C3; do
    bash "$LORA_DIR/run.sh" lst 2B "$CONDITION" "canary-lst-2B-$CONDITION-s17-$JOB_ID" \
        --purpose canary --max-steps 30 --eval-every 10
done
