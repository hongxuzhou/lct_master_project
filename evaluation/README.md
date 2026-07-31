# Evaluation — repair-aware SBN parsing

Scoring for the main project. Two metrics, because a repair-aware parser can
fail in two unrelated ways and one number cannot separate them.

| | question it answers | module |
|---|---|---|
| **Metric A** | Did the parser find the repair, quarantine the right material, and merge into the right context? | `repair_metrics.py` |
| **Metric B** | With the repair scaffolding stripped, does the speaker's final semantics match? | `repair_strip.py` |

Why both: a parser that silently normalises the disfluency away scores **A = 0
and B = 1.0** — it understood the sentence but never noticed the repair. A
parser that flags the repair but garbles the content scores the reverse. Plain
smatch on the repair-aware graphs conflates the two into one middling number.
The example fixture reproduces exactly this (see *Worked example* below).

The two are **not independent**. Mislabelling a cross-turn denial as an
intra-turn repair is penalised in A, and again in B because stripping then
deletes content the speaker did commit to. Say so when reporting.

## Pipeline

```
annotations.tsv  --build_gold.py-->  gold.parquet
                                          |
predictions.tsv  --evaluate_repair.py-----+-->  scored.parquet + summary
```

```bash
python3 build_gold.py -i annotations.tsv -o gold.parquet --strict
python3 evaluate_repair.py -i preds_long.parquet -g gold.parquet \
    -o preds_scored.parquet --solver ilp
```

`--solver ilp` = exact alignment (deterministic, recommended); `hillclimber` =
fast approximation.

---

## The data contract

**This is what the synthesis script has to produce.** Everything else is
derived — deriving rather than annotating is the point, because gold and
prediction then pass through byte-identical code and any difference between
them is the parser's, not two annotators'
### 1. Annotate these three columns

| column | type | notes |
|---|---|---|
| `id` | str | unique; duplicates are a hard error |
| `sentence` | str | the disfluent utterance, **byte-identical to what the parser will be fed** |
| `sbn_repair` | str | gold repair-aware SBN, **single line** |

`sbn_repair` is parsed with `is_single_line=True`, so newlines are neither
needed nor wanted. Inline ` % ` comments do survive — `split_single` inserts
the line breaks first, so a comment lands on the separator it follows, exactly
as the notation prescribes. The `%` still needs whitespace on **both** sides or
it is tokenised as SBN and aborts the parse.

### 2. Add whatever you will slice on

Passed through untouched; the analysis notebook groups by these.

| column | example |
|---|---|
| `repair_type` | `subject` `verb` `object` `adjunct` `preposition` `tense` `none` |
| `device` | `1`–`4`, which repair-toolkit device the annotation uses |
| `interregnum` | `"I mean"`, empty when absent — **keep it as its own column**, not only in an SBN comment, so it can be sliced on |
| `position`, `condition`, … | whatever the experimental design varies |

### 3. Never annotate these — `build_gold.py` derives them

| column | meaning |
|---|---|
| `penman_repair` | gold graph in Penman; reference for the repair-aware baseline |
| `penman_clean` | gold graph with the repair stripped; reference for Metric B |
| `partition` | `core` (both metrics) or `challenge` (Metric A only) |
| `strip_reason` | which gate sent it to the challenge set |
| `n_repairs` | intra-turn repairs found |
| `gold_status` | `ok` / `parse_error` / `strip_error` |

Two consequences worth knowing before you generate data:

- **The clean reference is Penman, not SBN.** Stripping happens on the graph,
  where `+n`/`-n` indices have already been resolved to node references, so
  nothing needs renumbering. Serialising back to SBN *would* require
  recomputing every index and is deliberately out of scope.
- **`partition` is a property of the dataset, fixed at build time.** If the
  stripper's gates ever change, that shows up as a diff in the gold table —
  which is what you want — instead of silently moving items between metrics.
  Watch the challenge-set count while generating: it tells you how much of the
  data only gets one metric.

### 4. Predictions

LONG format, one row per (item, condition):

| column | notes |
|---|---|
| `id` | must match the gold table |
| `condition` | optional; drives the per-condition breakdown |
| `pred_sbn` | raw parser output, single line |

---

## Output columns

`evaluate_repair.py` writes the prediction rows plus:

| column | meaning |
|---|---|
| `parse_status` | `success` / `ill_formed` / `parse_error` / `no_gold` |
| `partition` | copied from gold |
| `a_gold_n`, `a_pred_n`, `a_detected` | Metric A, detection |
| `a_token_*`, `a_span_*`, `a_merge_*` | Metric A raw counts — micro-averaging needs them, so any slice can be re-aggregated without re-scoring |
| `a_token_f1`, `a_span_f1` | per-item, for macro averages |
| `b_status`, `b_f1` | Metric B; `b_f1` is NaN unless `b_status == success` |
| `full_status`, `full_f1` | plain repair-aware graph score, no stripping — the obvious baseline, reported so nobody has to ask |

**Reparandum extent is reported twice on purpose.** `token` is a
position-blind multiset, `span` is `(concept index, token)`. Token alone cannot
tell a correct parse from one that swapped reparandum and repair — in a
tense/aspect repair both contain exactly `time.n.08` and `go.v.01`. Span alone
is brittle: one dropped concept earlier in the sentence shifts every index.
Read them as a pair — high token with low span means *right material, wrong
place*.

**Metric B failure policy.** A prediction that will not parse, or whose own
structure falls outside the stripper's domain, gets `b_f1 = NaN` and a status
saying which. The summary prints both means; quote the **penalised** one
(failures = 0) as the headline. The success-only mean silently rewards a parser
for failing loudly.

## Where the SBN modules come from

There is exactly one authoritative copy, `data/pmb-5.1.0/src/sbn/`, resolved by
`sbn_env.py`. Nothing here vendors a second one: `sbn_spec.py` carries the
project's own `CORRECTION` separator, extra roles and `INVERTIBLE_ROLES`
additions, and a duplicate drifts the moment a role is added to one copy and
not the other — with a silent scoring difference, not an error, as the symptom.
Override with `$SBN_SRC` on HPC. `colloquium_prep/pilot_eval/sbn_lib/` is a
frozen snapshot kept only so the pilot's published numbers stay reproducible;
do not point new code at it.

## Tests

```bash
python3 tests/test_repair_strip.py -v   # 16 cases
python3 tests/test_repair_metrics.py    # 8 cases
```

Both are driven by the worked examples in
`documentation/knowledge_base/repair_sbn_notation.qmd` and double as a
regression check on the notation itself. The stripper suite asserts that
stripping a repair-aware annotation reproduces the *natural fluent-sentence*
SBN **exactly** (F1 = 1.00 on all 11 single-sentence cases) — that is the claim
Metric B rests on. The metrics suite is built entirely from predictions that
parse cleanly but mean something else, since those are the failures a
parse-success check cannot see.

## Worked example

`tests/fixtures/` holds a runnable six-item dataset in the contract format
above, plus predictions exercising each failure mode.

```bash
python3 build_gold.py -i tests/fixtures/annotations_example.tsv -o /tmp/gold.tsv
python3 evaluate_repair.py -i tests/fixtures/predictions_example.tsv \
    -g /tmp/gold.tsv -o /tmp/scored.tsv
```

The `missed_repair` condition is the one to look at: **Metric A F1 = 0.00,
Metric B F1 = 1.00**, repair-aware baseline 0.67. The parser normalised the
disfluency away — perfect final semantics, zero repair structure — and only the
two-metric split makes that visible.

## Open decisions

- **Dangling anaphora into the reparandum (clarification C4).** When an anaphor
  picks up both reparandum and repair, stripping leaves a dangling edge and
  there is no uncontroversial clean reading. Currently every such item is gated
  to the challenge set. Revisit once the synthesised data shows how often it
  actually occurs.

## Deps

`pip install smatchpp penman networkx pandas pyarrow`
