# HTCondor job files — pilot pipeline

Paths assume the toolkit lives at
`/nethome/honzhou/thesis_zone/pilot/script/pilot_eval/` and gold at
`/nethome/honzhou/thesis_zone/pilot/data/pilot_dataset.parquet`. Outputs and
logs go to `/scratch/hongxuzhou/` (no backup, unlimited). Edit the paths in the
`.sh`/`.sub` files if yours differ.

## One-time setup
```bash
# after scp-ing this hpc/ folder into pilot_eval/
chmod +x /nethome/honzhou/thesis_zone/pilot/script/pilot_eval/hpc/*.sh
mkdir -p /scratch/hongxuzhou/logs/pilot_infer /scratch/hongxuzhou/logs/pilot_eval
mkdir -p /scratch/hongxuzhou/thesis/pilot

# env self-check (login node ok — imports only)
source /nethome/honzhou/lora_gemma2/bin/activate
python -c "import torch,transformers,peft,datasets,pandas,pyarrow,smatchpp,penman,networkx; print('all ok')"
```

## Run order
```bash
# 1. SMOKE: 16 rows on GPU, verify output looks like SBN
condor_submit infer_smoke.sub
#    -> /scratch/hongxuzhou/thesis/pilot/preds_smoke.parquet
#    inspect: python -c "import pandas as pd; print(pd.read_parquet('/scratch/hongxuzhou/thesis/pilot/preds_smoke.parquet').to_string())"

# 2. FULL inference (all 5852 rows)
condor_submit infer_full.sub
#    -> preds_long.parquet

# 3. SCORE (CPU)
condor_submit evaluate.sub
#    -> preds_long_scored.parquet  (also prints per-condition summary in the .out log)

# 4. pull scored parquet back to local, run analysis.ipynb
```

Monitor: `condor_q`, then tail the `.out`/`.err` under `/scratch/hongxuzhou/logs/`.
