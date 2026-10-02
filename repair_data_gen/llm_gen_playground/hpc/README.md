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

## Copy and prepare
Files go through the login node with scp (LST wiki, "File Transfer"); the
home folder is shared, so the submit node sees them. Only what the jobs need
is copied, so the cluster's `results/` is never overwritten:
```bash
# on the laptop, in repair_data_gen/llm_gen_playground/
scp -r run_llm.py wn_lookup.py check_repair_part.py pinned_sample.tsv \
    prompt_docs hpc honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground/
md5 hpc/setup.sh hpc/run.sh        # the jobs print the same sums in .out

# then: ssh honzhou@login.lst.uni-saarland.de, then: ssh submit
chmod +x /nethome/honzhou/thesis_zone/llm_gen_playground/hpc/*.sh
mkdir -p /scratch/hongxuzhou/logs/llm_gen
```
Results come back the same way:
```bash
scp -r honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground/results .
```

## Run order
```bash
# 1. SETUP (CPU, once): venv, WordNet, weights, and a report
condor_submit setup.sub
#    read setup.*.out: "vllm 0.30.0+cu129" and
#    "MuseGlimmerForConditionalGeneration supported: True"?
#    muse_glimmer parser files listed? (else TOOL_MODE=text in the .sub files)

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

- **Still `vllm 0.24.0`**: the job ran an old `setup.sh`. Compare the
  `script:` md5 at the top of `.out` with `md5 hpc/setup.sh` locally.
- **Architecture not supported**: the installed vLLM is too old for
  `muse_glimmer` (needs >= 0.28; `VLLM_VERSION` in `setup.sh`).
- **`torch sees no GPU` or CUDA errors**: first check the two
  `CUDA_VISIBLE_DEVICES` lines at the top of `.out` (the UUID → index
  renaming). If those look right, the cu129 build does not run on the 12.8
  driver after all; ask the admins whether a newer driver is planned.
- **No tool calls in the transcript** (`turns=1` on every content-word item,
  ATEM text in `result`): try `TOOL_MODE=auto` with a parser from the setup
  log, or check what the raw `content` looks like before changing the regex.
