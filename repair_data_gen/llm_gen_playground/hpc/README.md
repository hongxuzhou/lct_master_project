# HTCondor jobs: open-model runs (LST cluster, vanilla universe)

Paths assume the playground lives at
`/nethome/honzhou/thesis_zone/llm_gen_playground/`, the venv at
`/nethome/honzhou/venvs/vllm/`, caches and logs under `/scratch/hongxuzhou/`.
The weights follow the cluster rule and live in the shared
`/scratch/common_models/Muse-Glimmer-30B/` (`MODEL_DIR` in `setup.sh`,
`MODEL_PATH` in `run.sh`; the two must agree). Edit the variables at the top
of the `.sh` files if yours differ.

Before the first setup, check how the other models there are named and
whether you can write to it:
```bash
ls /scratch/common_models/ ; test -w /scratch/common_models && echo writable
```
If it is not writable, setup.sh skips the download and says so; ask the
admins to add the model, then go on from step 2.

## Copy and prepare (from the laptop, then the submit node)
```bash
rsync -av --exclude results/raw repair_data_gen/llm_gen_playground/ \
    honzhou@submit.coli.uni-saarland.de:/nethome/honzhou/thesis_zone/llm_gen_playground/
# on the submit node
chmod +x /nethome/honzhou/thesis_zone/llm_gen_playground/hpc/*.sh
mkdir -p /scratch/hongxuzhou/logs/llm_gen
```

## Run order
```bash
# 1. SETUP (CPU, once): venv, WordNet, weights, and a report
condor_submit setup.sub
#    read setup.*.out: "MuseGlimmerForConditionalGeneration supported: True"?
#    any parser names listed? -> TOOL_PARSER / REASONING_PARSER in the .sub files

# 2. SMOKE: 3 items -> results/meta-models_Muse-Glimmer-30B.smoke.tsv
condor_submit run_smoke.sub
#    check: parse=ok, turns >= 2 on content_word (the lookup ran), and the raw
#    transcript in results/raw/meta-models_Muse-Glimmer-30B.smoke/

# 3. FULL: 70 items -> results/meta-models_Muse-Glimmer-30B.tsv (resumable)
condor_submit run_full.sub

# 4. pull results/ back and score locally with analyse_run.py etc.
```

Monitor with `condor_q`, then read the `.out`/`.err` files under
`/scratch/hongxuzhou/logs/llm_gen/`.

## If something fails

- **Architecture not supported**: the installed vLLM is too old for
  `muse_glimmer`. Reinstall a newer vLLM (or the nightly) in the venv.
- **`torch sees no GPU` or CUDA errors**: the wheel was built for a newer
  CUDA than the 12.8 driver. Reinstall torch/vLLM from the cu128 builds.
- **No tool calls in the transcript** (`turns=1` on every content-word item,
  ATEM text in `result`): try `TOOL_MODE=auto` with a parser from the setup
  log, or check what the raw `content` looks like before changing the regex.
