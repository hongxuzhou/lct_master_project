# Self-repair parsing pilot — toolkit (inference + smatch++ eval + analysis)

Self-contained pilot pipeline. Portable: `scp -r` this whole folder to HPC.

## Pipeline (3 stages)
```
run_inference.py        # Stage 1: LoRA Gemma-2 parses 7 conditions  -> preds_long.parquet
evaluate_smatchpp.py    # Stage 2: smatch++ scores preds vs gold mr   -> preds_long_scored.parquet
analysis.ipynb          # Stage 3: summary tables + figures (run locally if preferred)
```

Stages 1+2 run on HPC (GPU for stage 1). Stage 3 is a lightweight notebook you
can pull back and run locally on the scored table.

## Layout
- `run_inference.py` — Stage 1. Loads base `google/gemma-2-9b-it` + adapter
  `Shrikes/sbn-gemma2-9b-lora-pmb`, melts the WIDE pilot dataset to LONG, greedy
  decode with left-padding, prompt byte-identical to training
  (`INSTRUCTION + "\n" + sentence`). Output LONG: `id, condition, input_nl, pred_sbn`.
- `evaluate_smatchpp.py` — Stage 2. CLI + importable scoring functions.
- `analysis.ipynb` — Stage 3. Reads the scored table; success rate, F1 (success
  + penalized), failure breakdown, position×interregnum, per-item Δ-vs-gold.
- `sbn_lib/` — bundled PMB modules (`sbn_smatch.py`, `sbn_spec.py`,
  `graph_base.py`, `penman_model.py`). Copies of `data/pmb-5.1.0/src/sbn/` with
  the legacy `from smatch import ...` guarded so no `smatch.py`/`amr.py` is
  needed. `sbn_spec.py` carries Hongxu's CORRECTION + invertible-role edits.

## Repair evaluation (two metrics)

A repair-aware parser is characterised by two numbers, answering different
questions. Neither alone is the score.

| | question | module | test |
|---|---|---|---|
| **Metric A** | did it find the repair, quarantine the right material, merge into the right context? | `sbn_lib/repair_metrics.py` | `test_repair_metrics.py` |
| **Metric B** | with the repair scaffolding removed, does the speaker's final semantics match? | `sbn_lib/repair_strip.py` | `test_repair_strip.py` |

Metric B strips CORRECTION/CONJUNCTION structure and the two kinds of
scaffolding dummy at the **graph** level, then scores the result with smatch++
against natural fluent-sentence SBN. It is a partial function: items outside
its machine-checkable domain return `status="na"` and belong in the challenge
set, where only Metric A applies. Metric A is deliberately un-gated so those
items still get a score.

The two are **not independent** — mislabelling a cross-turn denial as an
intra-turn repair is penalised in A and again in B, because stripping then
deletes content the speaker did commit to. State this when reporting.

Both modules' docstrings carry the design rationale; the test suites are driven
by the worked examples in `documentation/knowledge_base/repair_sbn_notation.qmd`
and double as a regression check on the notation itself.

```bash
python3 test_repair_strip.py -v   # 16 cases; 11 single-sentence repairs at F1 1.00
python3 test_repair_metrics.py    # 8 cases, all well-formed wrong predictions
```

## Why this exists
PMB SBN parsing research moved to **smatch++** (flipz357/smatchpp). This replaces
the legacy `evaluate_repair_smatch.py` which used the bundled `smatch.py`.
Pipeline: `SBN string → sbn_smatch.to_penman_string() → smatchpp F1`.
Use `sbn_smatch.py`, NOT `sbn2penman.py` (the latter rejects `>n`/`<n` box
indices).

## Deps
`pip install smatchpp penman networkx pandas pyarrow`

## Input / output
Input is a LONG table (one row per (id, condition)) with a prediction column.
Gold comes inline (`--gold-col`) or from a separate id→gold table
(`--gold-file`). Output = input rows + `status` and `f1` (0–1, NaN unless
`status == success`).

`status ∈ {success, ill_formed, parse_error, gold_error, smatch_error}`

## Example
```bash
python3 evaluate_smatchpp.py \
    -i preds_long.parquet -o preds_long_scored.parquet \
    --gold-file ../pilot_dataset.parquet --gold-file-gold-col mr \
    --solver ilp
```
`--solver ilp` = exact alignment (recommended, deterministic). `hillclimber` =
fast approximation.

## Notes
- smatchpp returns F1 on a 0–100 scale; the toolkit stores it as 0–1.
- Gold SBN is converted to Penman once per id (cached).
- **`make_scorer` passes `GenericStandardizer`.** Without it smatch++ does not
  de-invert `:X-of`, so every inverted role (`AttributeOf`, `PartOf`, `ThemeOf`,
  …) scores as a different edge — see the comment above `make_scorer`. Anything
  scored before Jul/2026 predates the fix and is understated. `GenericStandardizer`
  also lower-cases labels; if name casing must count, substitute a standardizer
  that runs `relabel_vars` + `deinvert_e` only.
- Feed predictions from the inference stage; the HF dataset
  (`Shrikes/self_repair_parsing_pilot_data`) stays a static input manifest.
