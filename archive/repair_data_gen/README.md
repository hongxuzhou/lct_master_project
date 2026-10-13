# Archived: repair data generation

Files moved out of `repair_data_gen/` on 2026-10-13, when the base train RSR
set was built. Nothing here is needed to rebuild that set. The full working
log of the period is the playground README at git tag `rsr-train-base-log`.

## `rule_pipeline/`

The first data source: a rule-based generator that substitutes one WordNet
neighbour on gold SBN and splices `CORRECTION`/`CONJUNCTION` around it. It
was replaced by LLM generation on 2026-10-04 because it reaches content words
only (see `FINDINGS.md`, "measured boundary"). The scripts import each other
and are kept as they were; they are not maintained.

| File | What it was |
|---|---|
| `generate_repairs.py`, `build_pool.py` | generator and candidate-pool builder |
| `wn_candidates.py`, `inflect_en.py` | WordNet candidates and inflection |
| `nli_filter.py`, `test_nli_filter.py`, `calib_*`, `sample_for_calibration.py` | NLI/surprisal filter and its calibration (not used as a filter, see `DECISIONS.md`) |
| `analyse_feasibility.py`, `nonfinite_probe.py`, `explore_lexical_resources.ipynb` | feasibility probes |
| `FINDINGS.md` (+ `FINDINGS_files/`) | write-up of the generator's results |
| `pool_*.tsv` | regenerable candidate pools (not in git) |

Still in use and therefore not archived: `repair_data_gen/repair_transform.py`
(oracle column of `analyse_run.py`), `sbn_lin.py`, `validate_with_pmb_parser.py`.

## `rsr_generation/`

Early LLM runs that contribute no item to the base set.

| File | What it was |
|---|---|
| `results/claude-opus-5-5.*`, `results/run_claude-opus-5-5.log` | Opus 5.5 pilot, 70 items (2026-10-01) |
| `results/meta-models_Muse-Glimmer-30B.tsv`, `.checked.tsv` | Muse Glimmer round 1 (2026-10-02) |
| `results/meta-models_Muse-Glimmer-30B.smoke.tsv` | cluster smoke run |
| `results/raw/…` | raw replies of these three runs |
| `samples/pinned_sample.tsv`, `hpc/run_full.sub`, `hpc/run_smoke.sub` | round-1 sample and jobs |
| `hpc/run_place_r5_pilot.sub` | place pilot job, never run (its documents went into the rerun) |
| `audit/r1_proto` | unlabelled prototype of the audit page |
| `dataset/quotes_review.tsv`, `dataset/onset_case_input.tsv` | review lists for the surface fixes; the decisions are in `dataset/onset_case_decisions.tsv` and `dataset/fix_log.tsv` |
