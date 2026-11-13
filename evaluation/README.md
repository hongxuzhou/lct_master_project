# Standalone basic scoring

`score_basic.py` checks graph validity and scores complete graphs with
SMATCH++. It does not select references or apply dataset-specific checks.

| Input | Required columns |
|---|---|
| References | `example_id`, `condition`, `gold_sbn` |
| Predictions | `example_id`, `condition`, `pred_sbn` |

Both inputs accept TSV, CSV or Parquet. Select the appropriate gold before
running the scorer. Each key must occur at most once per input. Run each
model/seed separately; reuse the same reference table where applicable.

```bash
python evaluation/score_basic.py --references refs.tsv \
  --predictions preds.tsv --output basic.jsonl
```

The output has one row per reference key. Precision, recall and F1 use the
same full-graph alignment. Precision and recall are null when no alignment
was run. A blank or invalid prediction
and an absent prediction row have separate `pred_status` values. Both get
F1 zero when the reference is valid. An invalid reference or scorer error
gets a null F1 and an explanation. A valid pair with F1 below one is still
`scored`. The original input strings can be retrieved by key.

For the archived pilot TSV, use `id` as `example_id`, the four variant
names (`head`, `mid`, `tail`, `interrug`) as conditions, `mr` as the
reference, and the corresponding `repair_*_sbn` as predictions. This
measures agreement with the fluent target only.

## Reader and scoring dependencies

The basic scorer calls the project's PMB-derived `SBNGraph.from_string()`
through `formal_graph.py`. The reader resolves concept and box indices using
`sbn_spec.py` and `graph_base.py`. `sbn_env.py` locates that reader;
`SBN_SRC` can point to the same reader revision on another machine.

The project checks graph validity and builds complete triples in
`formal_graph.py`. The external `smatchpp` package performs alignment and
triple matching with its ILP solver. The old PMB scoring, projection and
Penman export scripts are not called by basic scoring. However,
`sbn_smatch.py` imports `penman_model.py` and the legacy `smatch.py`
(which imports `amr.py`) when the reader loads. These are still needed to
import the current reader, even though basic scoring does not call their
scoring or export functions. Table I/O uses
pandas, with PyArrow for Parquet. Use the `repairsbn` environment.

## Which files to read

| Role | Files |
|---|---|
| Current basic scoring | `score_basic.py` is the entry point. It uses `formal_graph.py` for parsing and SMATCH++, `sbn_env.py` to locate the reader, and `tables.py` to read inputs. |
| Local checks under development | `formal_local.py` checks a supplied pattern; `formal_gold.py` extracts patterns from supported gold. Neither is needed for basic scoring. |
| Combined prototype | `score_formal.py` combines basic and local scoring. It still supports the original acceptance tests; it is not the entry point for the standalone workflow. |
| Historical Metrics A/B | [Archived scripts and tests](../archive/evaluation_metric_ab/README.md) reproduce the earlier scoring and pruning workflow. |

---

# Combined RepairSBN prototype

This prototype records the evaluation design. Use `score_basic.py` for
standalone basic scoring. To reproduce the combined prototype, supply the
released test tables and one prediction per
`(example_id, condition)`. Predictions can be Parquet, TSV or CSV and must
include `pred_sbn`. The selected gold tables and conditions define the panel;
missing prediction rows are visible and count as incomplete when the gold
supports the check.

```bash
python evaluation/score_formal.py \
  --default repair_data_gen/training_dataset/data/test.parquet \
  --hard-negative repair_data_gen/training_dataset/hard_negative/test.parquet \
  --later-reference repair_data_gen/training_dataset/later_reference/test.parquet \
  --predictions predictions.parquet --output items.jsonl --summary summary.json
```

Pass only the gold files you intend to score and use `--conditions` to restrict
the panel. `--core-c1` includes the optional C1 comparison on core RSR.
The summary records input and reader hashes, per-slice denominators, and
paired outcomes keyed by core `item_id` or later-reference `pair_id`.

| Slice | Full graph | Targeted checks |
|---|---|---|
| Fluent PMB and hard negatives | Ordinary gold for C1–C3 | False repairs among valid outputs; hard negatives also report gold relation recall |
| Core RSR | C2 uses `sbn_clean`; C3 uses `sbn_repair` | C3 repair correctness |
| Later reference, both sides | Each side's own `sbn_repair` for C2 and C3 | Exact member recovery for both; C3 RSR additionally checks repair and their joint correctness |

Full scoring uses SMATCH++ with ILP on the whole graph. Relation recall counts
gold semantic and discourse relations recovered under that same alignment,
excluding node labels, box membership and the root bookkeeping triple.
Local checks use a separate correspondence and preserve the scope of each
condition, including inverted roles. An invalid prediction gets full F1 zero
when the full gold is usable; it is an incomplete local task when that gold
pattern is usable. It receives no false-repair judgement or relation recall.
`review` reasons distinguish gold coverage, gold errors, ambiguous output
correspondences and scorer errors. Review counts are separate from evaluable
denominators. In the false-repair check, `fail` means a false repair is present,
so the summary reports the failure rate for that check.

Automatic core patterns cover simple lexical or value replacements where one
concept in each repair box isolates the contributions, metadata agrees with
the graph, and required links stay within supported scopes. Other constructions
need reviewed requirements. Supply those through `--reviewed-repairs` as a JSON
object keyed by `example_id`. Each entry contains `gold_sha256` (hash of the
UTF-8 gold SBN string), `concept_indices` (zero-based gold concept positions),
and `relations` (triples of gold source index, canonical role, and target
index or string constant). The scorer recovers condition scope from gold and
validates the pattern against it. These indices never address predictions.

Later-reference extraction uses each row's gold group and antecedent
indices. Member identity uses synsets, names and supported descriptive
modifiers without requiring any particular repair status. The later
predicate helps identify the group. If several candidates remain or the
group cannot be identified, the result goes to review. Quantity is outside
the exact-membership criterion. Automatic joint checking supports the
shared-event and anchored-subject repair constructions.

In a release audit (four base rows per core repair type), 8 of 28 core
patterns were ready and 20 needed review: 12 unsupported constructions,
7 cases where the boxes included continuation material, and 1 attachment
outside the supported scopes. Six later-reference pairs sampled across both
construction types supported all 12 member checks and all six RSR repair
patterns. These are sample counts, not release-wide coverage estimates.

Run the worked acceptance cards with
`python -m unittest discover -s evaluation/tests`. Use the `repairsbn`
environment or install the dependencies above, including an ILP backend.
