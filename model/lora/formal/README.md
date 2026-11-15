# Formal full-data LoRA

Run the wrapper from each cluster's shared checkout. It prints the scheduler
commands by default. Pass the same three agreed seeds on both clusters:

```bash
python3 formal/submit.py --cluster habrok --seeds SEED1 SEED2 SEED3
python3 formal/submit.py --cluster lst --seeds SEED1 SEED2 SEED3
```

Replace the seed placeholders with integers. Review the 18 Hábrók jobs
(2B/9B) and nine LST jobs (27B). Add `--check` to verify the interpreter,
train/dev hashes and cached model files without running a GPU job.
Then use `--submit` on the relevant submission host; it repeats the same
lightweight checks. There is one job per size, condition and seed. Use
`--sizes`, `--conditions` and `--seeds` to submit only the pending runs.
LST can pick up pending 2B/9B work on A100; cancel or remove the old pending
job first to avoid duplicate work. The 27B profile remains on LST H100.

`submit.py` sets the resource requests: one A100, four CPUs and 64 GB host
RAM for 2B/9B (24/48 hours); one H100, eight CPUs and 128 GB host RAM for
27B (72 hours). The existing cluster environment files provide `PYTHON`,
`DATASET_DIR` and `MODEL_CACHE`; set these explicitly if the prepared
paths differ. `FORMAL_RUN_ROOT` overrides the output directory. W&B uses
the same project and server as tuning. Job logs and local run files are
retained independently of W&B.

The tuning allocation placed 9B on LST. Ensure the pinned 9B model is
available in the Hábrók cache before moving those jobs there.

Read `run.py` first for the size/condition settings, then `../train.py`
for the training loop. In order:

1. Read the pinned train and validation data. C1 selects fluent inputs;
   C2 and C3 see the same inputs and differ only in the target SBN.
   Tokenisation masks the prompt from the loss and includes the answer's EOS.
2. Load a fresh frozen base model and attach LoRA to the verified projection
   layers. Only the adapter parameters are trained. Rank comes from the
   selected tuning configuration and alpha is twice rank.
3. Train for three epochs with microbatch one and effective batch 16.
   Accumulate gradients before each optimizer update. Use the tuning
   schedule and evaluate full dev every 0.3 epochs (rounded up in optimizer
   steps) and at the end.
4. Retain the best and latest checkpoints. Select the minimum dev loss,
   reload the best adapter on a fresh base, and verify that its dev loss
   matches. Save a small sample of dev generations for inspection.

`selection.json` names the selected checkpoint. It contains the PEFT
adapter and tokenizer alongside local training state. When Hub publishing
is configured, stage only the selected adapter, tokenizer and approved
metadata; exclude optimizer state and raw outputs. Formal test inference and
scoring are separate stages and are not called by the training job.

For formal test inference, use the audited W&B checkpoint TSV and run
`freeze.py CLUSTER checkpoints.tsv` on each site. It checks the saved
selection and run metadata and emits hashes for the selected adapter and
tokenizer. Keep that JSON with the test results. Use `test_submit.py CLUSTER
WORK_DIR` to preview one job per frozen run. The workers verify the frozen
files and test table hashes before loading a model. They read text and IDs,
then generate one TSV per run. For standalone scoring, prepare a reference
table for that run and pass it with the TSV to `evaluation/score_basic.py`.
Keep sizes and seeds separate so prediction keys stay unique. C1 covers
fluent PMB and hard negatives; C2 and C3 also cover core repair and later
reference.
