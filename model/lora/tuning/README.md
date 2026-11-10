# Hyperparameter tuning

Keep tuning-specific scripts, configurations and subset manifests here.
Shared training and data code remain in the parent directory.

## lr-rank-v2: fluent-reference data

The current plan retains the original six points and adds (LR, rank)
`(4e-4, 32)`, `(3e-4, 64)` and `(4e-4, 64)`: nine configurations per
size/condition, not a full 4-by-3 grid. Alpha remains twice rank; seed 17 and
all other training settings are unchanged. Discuss another boundary optimum
with the supervisor before extending further.

`configs/lr-rank-v2.json` owns the candidate list, protocol and subset paths.
The sweep searches a single categorical `candidate` parameter; the runner
resolves and records its actual LR/rank. Registration pins the candidate-file,
protocol and subset hashes. Use a new registry and output root for this round.

The protocol pins HF revision `d1ed13ec46823a7ddabdfe449e6ffe265d7d4a5b`.
`subsets/train10_fluent_groups_v1_s20261103/` contains 1,423 IDs: all 1,403 old
IDs plus 20 fluent-reference examples. C1 uses 969 train / 1,234 dev rows;
C2/C3 use 1,423 train / 1,746 dev rows. Validation remains complete.
C1 runs 183 steps, warmup 6, evaluation every 19; C2/C3 run 267 steps,
warmup 9, evaluation every 27. Schedules are computed from row counts.

Local checks, with no W&B registration or GPU training:

```bash
python model/lora/tuning/register_sweeps.py --plan lr-rank-v2 --model-size 2B
DATASET_DIR=repair_data_gen/training_dataset MODEL_CACHE=/tmp/unused \
  python model/lora/tuning/run_sweep.py --plan lr-rank-v2 --dry-run \
  --model-size 2B --condition C3 --cluster local
```

Pass `--plan lr-rank-v2` to registration, dry runs and summarisation.
Registered v2 sweeps pass this option to the runner automatically.
The [v2 cluster workflow](CLUSTER_RUNS.md) provides preview-first wrappers:
Hábrók 2B, LST A100 9B (pending fit checks) and LST H100 27B. They set
`TUNING_CLUSTER`, validate the registry and request the intended GPU resources.
The historical cluster submission scripts below still target v1.

```bash
python tuning/summarise.py --plan lr-rank-v2 \
  --run-root /path/to/lr-rank-v2 --output /path/to/lr-rank-v2/results.json
```

Selection requires all nine distinct successful candidates with matching code,
plan, protocol and subset provenance. Missing, duplicate, unexpected or mixed
versions receive no selected run. Losses remain comparable within each
size/condition only. Start fresh adapters for all new trials.

CLI defaults remain v1 so existing sweep commands cannot silently change rounds.
The following sections document the historical v1 workflow.

`build_subset.py` creates the fixed training-ID manifest in
`subsets/train10_s20261103/`; see the parent README for the rebuild command.
The accepted train and validation files are not rewritten.

## lr-rank-v1

Each size/condition gets a separate six-point grid: LR 1e-4/2e-4/3e-4,
rank 16/32, alpha = 2 * rank. Seed 17, effective batch 16, three epochs,
dropout 0.05, cosine schedule and warmup 0.03 are fixed. Use the existing
SBN v2 protocol without editing it. This is subset tuning, not a formal run.

The CLI agent starts a fresh Python process for every trial. `run_sweep.py`
initialises exactly one W&B run; the shared trainer records into that run.
Requires `wandb==0.31.0`. Initialisation or metric-upload errors fail the trial;
local records remain available. No automatic retries or checkpoint resume.
Failed configurations require explicit recovery before comparing a full grid.

The cluster computes the schedule after selecting train rows. C1 runs 180 steps
with six warmup steps and eval every 18 steps; C2/C3 run 264 steps with eight
warmup steps and eval every 27 steps. Validation is complete and unchanged.
Every trial reloads its best adapter and verifies dev loss, but generates no
diagnostic text. Metrics/configuration go to Forge; data and checkpoints stay local.

## Habrok

Install nothing: reuse `thesis-opt`, cached models and accepted Parquets.
Code is at `/home1/s5788668/thesis_zone/lora/tuning/`; outputs are under
`/scratch/s5788668/thesis/qwen35_lora_tuning/lr-rank-v1/SIZE/CONDITION/RUN_ID/`.
Slurm logs are under `tuning/habrok/logs/`. Keep registrations in
`tuning/registrations/` and reuse their sweep IDs across job submissions.
W&B SDK cache/logs stay under the tuning result root's `wandb/` directory.
The subset builder needs the local repository; do not rebuild subsets on cluster.

From the uploaded `lora/` directory, inspect assignments without contacting W&B:

```bash
source habrok/environment.sh
"$PYTHON" tuning/register_sweeps.py --model-size 2B
"$PYTHON" tuning/run_sweep.py --dry-run --model-size 2B --condition C1
"$PYTHON" tuning/run_sweep.py --dry-run --model-size 2B --condition C2
"$PYTHON" tuning/run_sweep.py --dry-run --model-size 2B --condition C3
```

On a permitted submission/preparation node, register the three sweeps once.
Registration creates remote metadata, not GPU jobs. Preserve a partial registry
if registration fails; do not blindly create replacements.

```bash
"$PYTHON" tuning/register_sweeps.py --model-size 2B --create \
  --registry tuning/registrations/lr-rank-v1-2B.json
cd tuning/habrok
mkdir -p logs
sbatch agent.sbatch hxzhou-research/qwen35-rsr-lora/SWEEP_ID 1
```

Use the registry's actual C1/C2/C3 sweep path in place of `SWEEP_ID`.
The first job runs one full-budget configuration for inspection. After it
passes, submit count 5 for that sweep and count 6 for each remaining condition.
One agent runs trials sequentially on its allocated A100. The twelve-hour job
request is headroom for a 2B six-trial group, not a measured runtime guarantee.
Do not use this 40GB A100 resource request for larger sizes without checking fit.
The agent host and training host are both `https://api.forge.coreweave.com`.

Summarise downloaded or cluster-local results:

```bash
"$PYTHON" tuning/summarise.py \
  --run-root /scratch/$USER/thesis/qwen35_lora_tuning/lr-rank-v1 \
  --output /scratch/$USER/thesis/qwen35_lora_tuning/lr-rank-v1/summaries/results.json
```

Only a complete six-point grid with matching code/data provenance and no
duplicate configurations receives a selected run. Compare losses within each
size/condition only. W&B Runs provides the table; no `wandb.Table` is needed.
