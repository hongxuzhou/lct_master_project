# Qwen3.5 development training

Hábrók uses `thesis-opt`. `habrok/environment.sh` provides default interpreter,
cache and output paths for both submission and execution. Explicit environment
overrides remain supported. `bash habrok/submit.sh` checks the interpreter,
prepared files and shell syntax before submitting the canary. It does not rerun
tokenisation or request a GPU for configuration checks. An explicit job-script
argument can be supplied for subsequent development jobs. This helper still
defaults to the existing 30-step canary; it is not the full-training submission.

One single-GPU runner serves both clusters. It reads the accepted train and
validation Parquets without rewriting them. `protocol.json` pins the published
dataset and model revisions. This is a development entry point: it deliberately
does not offer a formal-run mode before configuration calibration and freezing.

The published fluent-reference version uses the explicit protocol
`protocols/fluent_groups_v1.published.json` and dataset directory
`repair_data_gen/training_dataset`. It includes both fluent variants in C1 and
preserves the same new examples in C2/C3. The nine-candidate tuning round uses
`--plan lr-rank-v2`; see the [tuning workflow](tuning/README.md).
Previous protocols and the v1 tuning subset remain pinned to the old release;
use dataset directory
`repair_data_gen/training_dataset/releases/published-9b7d881` to reproduce them.

C1 selects fluent rows and `sbn_clean`. C2 and C3 use the same inputs and order,
with `sbn_clean` and `sbn_repair` respectively. Every run starts with a fresh
base model and adapter. The neutral instruction and chat rendering match the
development acceptance audit. Overlength targets cause an error, not truncation.

The candidate retains smoke-tested rank 16 / alpha 32 and projection coverage.
It uses effective batch 16, cosine scheduling, warmup 0.03 and a three-epoch
budget. Learning rate 2e-4 is a starting candidate, not a calibrated choice.
Canaries stop after 30 steps using the full-budget schedule. Their seed 17 and
10-step evaluation interval are development settings, not formal decisions.

## Full 2B development on Hábrók

Upload the complete `lora/` source directory to
`/home1/s5788668/thesis_zone/lora/`, including `prompts/`. Reuse the prepared
model/data caches and validated `thesis-opt` environment. No installation,
download or additional GPU canary is needed for this submission.

```bash
cd /home1/s5788668/thesis_zone/lora
bash habrok/submit_full_2b.sh
```

This named helper selects `full_2b.sbatch`, never the default canary. It exports
the shared environment defaults and requests one A100, four CPUs, 64 GB RAM and
24 hours in `gpu`. The [Hábrók partition guide](https://wiki.hpc.rug.nl/habrok/job_management/partitions)
lists a three-day GPU limit. Canary step timings imply about 10.5 hours of
training across the three conditions before evaluation, reload and generation;
24 hours leaves headroom but is not a runtime guarantee.

The job runs C1, C2 and C3 sequentially, each in a separate Python process with
a fresh original base and adapter. It stops if a condition fails; retained
outputs identify the completed conditions. There is no checkpoint resume or
automatic retry. Run IDs include the Slurm job ID, condition, seed and prompt
version, so resubmission starts new runs without overwriting earlier outputs.

Settings are rank 16 / alpha 32 / dropout 0.05, peak LR 2e-4, microbatch 1,
accumulation 16, seed 17, three epochs and no early stopping. Full dev evaluation
and saving occur every 100 optimizer steps and at the final step. C1 runs
1,779 steps; C2 and C3 each run 2,631. Each condition selects its own minimum
dev-loss checkpoint, verifies reload loss and produces 20 greedy dev generations
with a 512-token cap. `calibration` is the existing CLI name for a longer
development run; no hyperparameter search is performed. These runs are outside
the formal 27-run matrix.

The script explicitly selects `prompts/sbn_v2/protocol.json`, which names SBN in
the shared instruction. [Prompt hashes](prompts/README.md) identify both versions.
The original `protocol.json` and canary script retain the old instruction. Model
and dataset revisions, target mapping and candidate hyperparameters are unchanged.
The new prompt has a full-data CPU audit; training also audits the selected
condition before loading model weights. To repeat the CPU audit on a permitted
preparation node using the existing caches:

```bash
source habrok/environment.sh
"$PYTHON" preflight.py --protocol prompts/sbn_v2/protocol.json --model-sizes 2B \
  --dataset-dir "$DATASET_DIR" --model-cache "$MODEL_CACHE" \
  --output /tmp/qwen35-sbnv2-preflight.json
```

Outputs appear under `$RUN_ROOT/full-dev-habrok-2B-C1-s17-sbnv2-JOBID/` and the
corresponding C2/C3 directories. Check each `summary.json`, `selection.json`,
`metrics.jsonl` and `dev_generations.json`; job completion alone does not establish
parsing quality. Slurm logs are under `habrok/logs/qwen35-full-2B-JOBID.{out,err}`.

## Fixed tuning subset

`tuning/subsets/train10_s20261103/manifest.json` selects 1,403 train rows from the
accepted release: 949 fluent, 349 base RSR and 105 marked RSR. It contains IDs
and provenance only; the original train and validation files remain unchanged.
Use the same manifest for every model size and candidate configuration.
C1 uses its fluent rows; C2/C3 use all selected rows with their respective targets.
The trainer accepts it through `--train-subset`. The
[tuning workflow](tuning/README.md) owns Sweep configuration and submission.

Rebuild in the `repairsbn` environment from the repository root:

```bash
python model/lora/tuning/build_subset.py \
  --dataset-dir repair_data_gen/training_dataset \
  --output-dir model/lora/tuning/subsets/train10_s20261103
```

Sampling seed 20261103 is independent of the training seed. Repair items are
stratified by repair type and marked availability, with base/marked variants
selected together. Fluent examples are sampled separately. Integer quotas use
largest remainders; SHA-256 ranking chooses groups; selected rows retain source
order. Different items from the same document are not grouped together: this is
a training subset, not a new train/validation split. Word lengths are reported,
not stratified. `composition.json` records counts and condition input digests.
The builder verifies source checksums and rejects hard-negative documents.

## W&B

Both clusters use the existing `hxzhou-research/qwen35-rsr-lora` project on Forge.
Use the environment in which W&B is already configured; no login or install
step is needed. Online is the default. Set `WANDB_MODE=offline` if desired.
Only scalar metrics, configuration and runtime metadata are logged. No code,
dataset, generated text, adapter or checkpoint is uploaded. Local JSON records
and checkpoints remain authoritative if dashboard initialisation or logging fails.

## Prepare on each cluster

Upload the complete `model/lora/` directory. Activate the existing validated
Qwen/W&B environment; `python`, `hf`, `pyarrow`, `transformers` and `peft` must
be available. Set `PYTHON` and `HF_CLI` explicitly if they are not on PATH.

On Habrok, from the uploaded `lora/` directory:

```bash
export PYTHON=$HOME/miniconda3/envs/thesis-opt/bin/python
export HF_CLI=$HOME/miniconda3/envs/thesis-opt/bin/hf
export MODEL_CACHE=/scratch/$USER/huggingface/hub
export DATASET_DIR=/scratch/$USER/thesis/repair-sbn-9b7d8812
export RUN_ROOT=/scratch/$USER/thesis/qwen35_lora_dev
bash prepare.sh 2B
```

The validated Hábrók `thesis-opt` environment has `causal-conv1d==1.7.0`,
`fla-core==0.5.2` and `flash-linear-attention==0.5.2`. The job scripts check
these versions. The earlier isolated installation workflow is documented in
`../smoke_test/habrok/`; do not repeat it for an already prepared environment.

This caches the pinned release and runs CPU preflight on all train and dev rows
for each condition. Use a preparation/CPU allocation if cluster policy requires
one. A model-cache download is incremental when the pinned model is present.
The preparation uses the `hf-cli` workflow and preserves home-level credentials.

Submit only after all three preflights pass:

```bash
cd habrok
mkdir -p logs
sbatch canary.sbatch
```

The Slurm job runs C1, C2 and C3 independently and sequentially on one A100.
Each uses all applicable train rows for shuffling and the entire condition's dev
split at each evaluation; only the number of training steps is shortened.
Override `PYTHON` only if the validated clone uses a non-default path.

LST uses the same entry point. Upload to `/nethome/honzhou/thesis_zone/lora/`.
Set these paths before submitting preparation through the existing CPU-job
workflow (do not prepare data or environments on the LST login host):

```bash
export PYTHON=/nethome/honzhou/miniconda3/envs/thesis-qwen35-opt/bin/python
export HF_CLI=/nethome/honzhou/miniconda3/envs/thesis-qwen35-opt/bin/hf
export MODEL_CACHE=/scratch/common_models/HuggingFace
export DATASET_DIR=/scratch/hongxuzhou/thesis/repair-sbn-9b7d8812
export RUN_ROOT=/scratch/hongxuzhou/thesis/qwen35_lora_dev
```

The optional `lst/canary.sub` repeats the 2B development group on an A100; it
is not a 27B formal submission. `getenv` transfers the explicit paths and existing
W&B settings. Ensure `/scratch/hongxuzhou/logs` exists before submission.

## Selection and verification

Prompt and padding labels are masked; target suffix and EOS are supervised.
Training accumulation and development loss weight all supervised tokens equally,
including a partial final batch. Evaluation never shuffles or generates text.
At each evaluation, the runner saves an adapter, tokenizer, optimizer, scheduler,
RNG states and evaluation record. It retains the lowest-dev-loss checkpoint and
the latest checkpoint, with earliest step winning an exact tie. There is no
early stopping. Resume is not implemented yet; use a new run ID for a restart.

After training, the original model and optimizer are released. The best adapter
is loaded on a fresh base and its full dev loss must match within tolerance.
Then 20 deterministic dev generations cover the available variants, with EOS
termination recorded. These are pipeline diagnostics; a short canary is not
evidence of parsing quality. Inspect EOS counts before longer development runs.

Each successful run has `summary.json` with `development_passed`, `config.json`,
`data_audit.json`, `metrics.jsonl`, `selection.json`, `lora_modules.json`, retained
checkpoints and local-only `dev_generations.json`. Compare C2/C3 input digests,
token counts and row counts. Do not compare their absolute dev losses as task
scores. Canary completion, W&B upload and cluster execution require actual jobs;
local CPU preflight alone does not establish them.

Calibration can use `--purpose calibration --max-steps 0 --eval-every N` and an
explicit `--learning-rate`. Set N deliberately before the run. Formal seeds,
per-size settings, final evaluation cadence, resume support and the 27-run
manifest remain separate work after the canary evidence has been reviewed.

## Local checks

```bash
python -m unittest discover -s model/lora/tests -v
python model/lora/train.py --preflight-only --model-size 2B --condition C3 \
  --cluster local --wandb-mode disabled --dataset-dir repair_data_gen/training_dataset \
  --model-cache /path/to/cache --tokenizer-dir /path/to/pinned/2B \
  --output-dir /tmp/qwen35-C3-preflight
```
