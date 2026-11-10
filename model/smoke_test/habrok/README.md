# Habrok Qwen3.5 LoRA smoke test

This directory is the Slurm adapter for the shared smoke-test Python script.
It supports the 2B and 9B checkpoints on one A100. The 27B smoke test remains
on the LST H100 workflow.

## Preconditions

The `thesis` conda environment must contain these exact versions:

```text
torch 2.14.1+cu130
transformers 5.20.2
peft 0.21.2
datasets 5.1.0
accelerate 1.15.0
huggingface_hub 2.3.0
safetensors 0.8.0
```

The CUDA build must include `sm_80`. Do not run this workflow on a V100.

The validated `thesis` environment intentionally remains the fallback baseline.
Optional compiled kernels are installed in a separate `thesis-kernels` clone.

## Prepare the offline caches

Activate the environment, then cache the first model and dataset:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /path/to/lct_master_project/model/smoke_test/habrok
chmod +x prepare.sh run_smoke.sh
./prepare.sh 2B
```

`prepare.sh` reads the fixed model and dataset commits from the shared
`../revisions.json`, downloads/verifies them, and leaves the Hugging Face token
in the user's home directory. Model and dataset caches live under
`/scratch/$USER/huggingface/`.

Prepare the 9B checkpoint only after the 2B smoke test passes:

```bash
./prepare.sh 9B
```

Before requesting another GPU, validate the deterministic 96-row subset and
both supervision conditions on a login or CPU node:

```bash
python ../preflight_tokenization.py \
  --model-size 2B \
  --max-length 512 \
  --model-cache /scratch/$USER/huggingface/hub \
  --dataset-cache /scratch/$USER/huggingface/datasets
```

This reads only the offline caches. A passing report must show a positive
`min_supervised_tokens` for both `clean` and `repair`.

## Run on an existing A100 allocation

From the same directory:

```bash
./run_smoke.sh 2B repair
```

The job uses the caches offline and writes a unique result directory under
`/scratch/$USER/thesis/qwen35_lora_smoke/`.

## Submit with Slurm

Create the log directory before submission because Slurm opens log files
before the job script starts:

```bash
cd /path/to/lct_master_project/model/smoke_test/habrok
mkdir -p logs
sbatch qwen35_lora_smoke_test.sbatch 2B repair
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/qwen35-lora-smoke-JOB_ID.out
```

Static validation of these files does not prove Habrok execution. A smoke test
passes only when training, adapter save, clean base-plus-adapter reload, and
post-reload generation all complete and `summary.json` reports `passed`.

## Add the optional Qwen3.5 kernels

Run the installer on a Habrok compute node. It clones `thesis` instead of
modifying it, then installs the pinned packages and checks their imports:

```bash
cd /path/to/lct_master_project/model/smoke_test/habrok
chmod +x install_optional_kernels.sh
./install_optional_kernels.sh
```

The pinned optional packages are:

```text
causal-conv1d 1.7.0
fla-core 0.5.2
flash-linear-attention 0.5.2
```

If an interrupted installation left the intended clone in place, inspect it
before explicitly continuing:

```bash
./install_optional_kernels.sh --reuse-existing
```

Do not use this flag for an unrelated existing environment.

The import check is not GPU validation. Submit the dedicated A100 job next:

```bash
mkdir -p logs
sbatch qwen35_lora_kernel_check.sbatch
```

The job first runs a CUDA forward/backward check for `causal-conv1d`, then runs
the full 2B repair smoke test with the cloned environment. Accept the accelerated
environment only if the job exits normally, stderr contains no optional-kernel
fallback warning, and the new `summary.json` reports `passed`. Keep using
`thesis` if installation or validation fails.
