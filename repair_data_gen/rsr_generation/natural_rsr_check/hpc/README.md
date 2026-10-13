# HTCondor jobs: naturalness measures (LST cluster, CPU only)

Same conventions as `../../hpc/README.md`: files go through the login node
with scp, the venv is built with uv by a CPU job, models live in
`/scratch/common_models/`, logs under `/scratch/hongxuzhou/logs/`, and every
script prints its md5 at the top of `.out`.

Both jobs are CPU jobs. A 124M and a 160M model score some tens of thousands
of short requests in minutes, so there is no GPU queue and no UUID renaming.

`swbd/` holds Switchboard-derived data. It may be on this laptop and on the
cluster, and nowhere else (todo §7).

## Copy

```bash
# on the laptop, in repair_data_gen/rsr_generation/natural_rsr_check/
L=honzhou@login.lst.uni-saarland.de
D=thesis_zone/rsr_generation/natural_rsr_check
ssh $L "mkdir -p $D/swbd $D/lm"
scp build_sets.py lm_items.py score_lm.py lexical.py synth_typed.tsv $L:$D/
scp -r hpc $L:$D/
scp swbd/sentences.tsv swbd/disfluencies.tsv swbd/real_set.tsv $L:$D/swbd/
md5 hpc/setup_rsr.sh hpc/run_rsr.sh lm_items.py score_lm.py lexical.py

# then: ssh $L, then: ssh submit
chmod +x ~/thesis_zone/rsr_generation/natural_rsr_check/hpc/*.sh
mkdir -p /scratch/hongxuzhou/logs/rsr
```

## Run order

```bash
# 1. SETUP (CPU, once): venv, NLTK data, Pythia, GloVe, SUBTLEXus, report
condor_submit hpc/setup_rsr.sub
#    in setup.*.out check:
#    - the GPT-2 directory under "GPT-2 candidates"; if it is not
#      /scratch/common_models/gpt2, set GPT2_DIR in run_rsr.sub (and rerun
#      the setup with that GPT2_DIR to see its report line);
#    - both models: "word-start tokens" above 10000;
#    - glove.6B.300d.txt has 400000 lines; the SUBTLEXus header shows Word
#      and SUBTLWF.

# 2. RUN (CPU): normalise, score with both models, lexical features
condor_submit hpc/run_rsr.sub
#    ONLY=pythia or ONLY=gpt2 in `environment` scores with one model only.
```

Monitor with `condor_q`, then read the `.out`/`.err` files under
`/scratch/hongxuzhou/logs/rsr/`.

## Copy back

Only the files the run made:

```bash
scp "$L:$D/swbd/{lm_items_real,lm_requests_real,lexical_real}.tsv" swbd/
scp "$L:$D/swbd/scores_*_real.tsv" swbd/
scp "$L:$D/lm/*.tsv" lm/
```

## If something fails

- **`num2words is not installed (--strict)`**: the run used another venv;
  check `VENV` in `run_rsr.sh`.
- **`only N word-start tokens`**: the model's tokeniser does not mark word
  starts with `Ġ`, so the whitespace correction in `score_lm.py` would be
  wrong. Do not use that model without changing `word_start_ids`.
- **`no model at …`**: Pythia could not be put in `/scratch/common_models`
  (see the setup log), or `GPT2_DIR` is wrong.
