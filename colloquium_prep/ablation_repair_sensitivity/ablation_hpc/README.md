# HTCondor job files — repair-sensitivity ablation

Sibling of `pilot_eval/hpc/`, same conventions. Paths assume the toolkit is
scp'd to `/nethome/honzhou/thesis_zone/pilot/script/ablation_repair_sensitivity/`.
Outputs and logs go to `/scratch/hongxuzhou/` (no backup, unlimited). Edit the
paths in the `.sh`/`.sub` files if yours differ.

Unlike the main pilot pipeline, there is **no HPC evaluation stage** here —
`evaluate_ablation.py` is meant to run locally (it needs no GPU and pulls
ground truth straight from the HF dataset via `ground_truth.py`), so develop
and test it against the local MLX smoke test output while the HPC job queues,
then just download the real `preds_ablation*.parquet` and run it there.

## One-time setup
```bash
# after scp-ing this ablation_repair_sensitivity/ folder into script/
chmod +x /nethome/honzhou/thesis_zone/pilot/script/ablation_repair_sensitivity/hpc/*.sh
mkdir -p /scratch/hongxuzhou/logs/ablation_infer
mkdir -p /scratch/hongxuzhou/thesis/ablation

# env self-check (login node ok — imports only; same venv as the main pilot)
source /nethome/honzhou/lora_gemma2/bin/activate
python -c "import torch,transformers,datasets,pandas,pyarrow; print('all ok')"
```

## Run order
```bash
# 1. SMOKE: 35 rows on GPU, verify output looks like the local MLX dry run
condor_submit infer_ablation_smoke.sub
#    -> /scratch/hongxuzhou/thesis/ablation/preds_ablation_smoke.parquet
#    inspect: python -c "import pandas as pd; print(pd.read_parquet('/scratch/hongxuzhou/thesis/ablation/preds_ablation_smoke.parquet').to_string())"

# 2. FULL inference (all 5852 rows: 836 sentences x 7 conditions, single prompt)
condor_submit infer_ablation_full.sub
#    -> preds_ablation.parquet

# 3. Pull back to local, run:
#    python3 evaluate_ablation.py -i preds_ablation.parquet -o preds_ablation_scored.parquet
```

Monitor: `condor_q`, then tail the `.out`/`.err` under `/scratch/hongxuzhou/logs/ablation_infer/`.
