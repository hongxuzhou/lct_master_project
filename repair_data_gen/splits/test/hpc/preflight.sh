#!/bin/bash
# Run on the LST submit host. No GPU work or job submission.
set -euo pipefail
ROOT=/nethome/honzhou/thesis_zone/test_synthesis_v1
cd "$ROOT"
sha256sum -c files.sha256
[ -x /nethome/honzhou/venvs/vllm/bin/python ]
[ -x /nethome/honzhou/venvs/vllm/bin/vllm ]
[ -f /scratch/common_models/Muse-Glimmer-30B/config.json ]
[ -d /nethome/honzhou/nltk_data ]
mkdir -p /scratch/hongxuzhou/logs/llm_gen
for stage in edge_place temporal_p1 edge_direction base; do
    condor_submit -dry-run "/tmp/test_${stage}_preflight.$$.ads" \
        "splits/test/hpc/run_test_${stage}.sub"
done
condor_q -submitter honzhou
printf '%s\n' 'Preflight passed. No jobs submitted.'
