# Storm/weather: worked scoring example for the Experiment Design slides

Scored 2026-09-27 with the project evaluator (`evaluation/`, smatch++ ILP solver,
GenericStandardizer). One gold item, four constructed outputs. These are not
model predictions; they are hand-built outputs that stand for the typical
behaviour of each training condition.

## Gold

Source sentence: *What is the weather like in Dublin?* Repair site `weather.n.01`,
reparandum `storm.n.01` (WordNet, same POS). Gold SBN produced by
`repair_data_gen/repair_transform.build_repair` (strategy: anchor, device ⑤):

```
entity.n.01 EQU ? time.n.08 EQU now entity.n.01 CORRECTION <1 storm.n.01 EQU -1 CONJUNCTION <2 % I mean weather.n.01 EQU -2 like.a.01 Co-Theme -5 Time -4 Theme -3 Location +1 city.n.01 Name "Dublin"
```

`build_gold.py --strict`: status ok, partition core, 1 repair.

## Outputs

| id | Stands for | SBN (differences from the fluent SBN) |
|---|---|---|
| normalisation | RSR normalisation (deletion) | fluent SBN, *storm* dropped |
| both_asserted | parser trained on fluent SBN only | `storm.n.01` added as a free concept, no contexts |
| both_asserted_mean | same, pilot-style (cf. goat/cow) | `storm.n.01` plus *I mean* as `mean.v.01` event |
| repair_aware | repair-aware target | = gold |

## Scores

| Output | SMATCH++ vs repair-aware gold (`full_f1`) | SMATCH++ vs fluent gold (pilot-style) | Repair detected | Reparandum F1 (token / span) | Merge context | Accepted meaning (`b_f1`) |
|---|---:|---:|:-:|---:|:-:|---:|
| normalisation | 0.667 | 1.000 | ✗ | 0 / 0 | ✗ | 1.000 |
| both_asserted | 0.680 | 0.950 | ✗ | 0 / 0 | ✗ | 0.950 |
| both_asserted_mean | 0.557 | 0.745 | ✗ | 0 / 0 | ✗ | 0.745 |
| repair_aware | 1.000 | 0.667 | ✓ | 1 / 1 | ✓ | 1.000 |

Accepted meaning = the prediction with its withdrawn context stripped, scored against
the fluent gold graph; for outputs without a repair it equals the pilot-style score.

## Reproduce

Inputs are the SBN strings above, written to `annotations.tsv` (`id, sentence, sbn_repair`)
and `predictions.tsv` (`id, condition, pred_sbn`).

```bash
python3 evaluation/build_gold.py -i annotations.tsv -o gold.tsv --strict
python3 evaluation/evaluate_repair.py -i predictions.tsv -g gold.tsv -o scored.tsv --solver ilp
```

Full prediction strings:

```
normalisation       entity.n.01 EQU ? time.n.08 EQU now weather.n.01 like.a.01 Co-Theme -3 Time -2 Theme -1 Location +1 city.n.01 Name "Dublin"
both_asserted       entity.n.01 EQU ? time.n.08 EQU now storm.n.01 weather.n.01 like.a.01 Co-Theme -4 Time -3 Theme -1 Location +1 city.n.01 Name "Dublin"
both_asserted_mean  entity.n.01 EQU ? time.n.08 EQU now storm.n.01 person.n.01 EQU speaker mean.v.01 Theme -2 Agent -1 Time +1 time.n.08 EQU now weather.n.01 like.a.01 Co-Theme -7 Time -6 Theme -1 Location +1 city.n.01 Name "Dublin"
```
