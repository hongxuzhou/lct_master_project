# What predicts an SBN `be` node? — pattern analysis

- be-form samples analysed: **3483** (with be node: **947**, without: **2536**)

## Feature presence by class

Each cell = share of that class whose sample has the feature.

| feature | with be-node | without be-node |
|---|---:|---:|
| `sbn_has_adj` | 20.6% | 56.6% |
| `sbn_has_adv_pred` | 1.0% | 11.2% |
| `sbn_has_nonbe_verb` | 7.6% | 52.1% |
| `sbn_has_co_theme` | 65.2% | 7.8% |
| `sbn_has_attributeof` | 14.0% | 37.4% |
| `sbn_has_role` | 25.7% | 10.0% |
| `snt_existential` | 12.9% | 0.2% |
| `snt_wh_locative` | 1.7% | 0.5% |
| `snt_progressive` | 2.2% | 25.6% |
| `snt_passive_by` | 0.0% | 4.5% |

## Coarse construction type vs. be-node

| construction type | n | with be-node | be-node rate |
|---|---:|---:|---:|
| auxiliary (lexical verb present) | 1393 | 72 | 5.2% |
| adjectival — predicate adjective | 1017 | 30 | 2.9% |
| nominal / equative (predicate noun) | 582 | 558 | 95.9% |
| locative / existential / measure (no other predicate) | 314 | 285 | 90.8% |
| adjectival — comparative | 168 | 2 | 1.2% |
| adverbial — predicate adverb | 9 | 0 | 0.0% |

## Rule accuracy

**Rule:** a `be` node appears iff the predicate is nominal (predicate noun / identity) or locative / existential / measure — i.e. there is no adjective, adverb or lexical verb to carry the predicate.

| | actually has be-node | actually none |
|---|---:|---:|
| predicted be-node | 843 | 53 |
| predicted none | 104 | 2483 |

- accuracy: **95.5%**  ·  precision: **94.1%**  ·  recall: **89.0%**

## be-sense ↔ construction (within `has_be_node`)

| be sense | top construction types (count) |
|---|---|
| `be.v.01` | nominal / equative (predicate noun) (170), auxiliary (lexical verb present) (20), locative / existential / measure (no other predicate) (6) |
| `be.v.02` | nominal / equative (predicate noun) (260), auxiliary (lexical verb present) (31), adjectival — comparative (1) |
| `be.v.03` | locative / existential / measure (no other predicate) (267), adjectival — predicate adjective (29), auxiliary (lexical verb present) (13) |
| `be.v.04` | nominal / equative (predicate noun) (1), locative / existential / measure (no other predicate) (1) |
| `be.v.05` | nominal / equative (predicate noun) (1) |
| `be.v.06` | nominal / equative (predicate noun) (35), auxiliary (lexical verb present) (3), locative / existential / measure (no other predicate) (1) |
| `be.v.08` | nominal / equative (predicate noun) (86), auxiliary (lexical verb present) (5) |
| `be.v.13` | locative / existential / measure (no other predicate) (9) |

## Diagnostic residue

- adjectival-predicate samples that *do* carry a be node: **30** (expected ~0 under the hypothesis)
- nominal/locative/existential samples *without* a be node: **53**

Examples (adjective complement yet be-node present):

- `[p60/d1674]` My house is on the south bank of the Thames.
    - `person.n.01 EQU speaker house.n.01 User -1 be.v.03 Theme -1 Time +1 Location +3 time.n.08 EQU now south.a.01 Theme +1 bank.n.01 PartOf +1 river.n.01 Name "Thames"`
- `[p26/d1963]` The corporate headquarters is in Los Angeles.
    - `corporate.a.01 AttributeOf +1 headquarters.n.01 be.v.03 Theme -1 Time +1 Location +2 time.n.08 EQU now city.n.01 Name "Los Angeles"`
- `[p50/d2610]` There was a beautiful woman with black hair in the park.
    - `entity.n.01 be.v.03 Time +1 EQU -1 Theme +3 Location +6 time.n.08 TPR now beautiful.a.01 AttributeOf +1 woman.n.01 Part +2 black.a.01 hair.n.01 Colour -1 park.n.02`
- `[p90/d3545]` The little boy is at the zoo.
    - `little.a.01 AttributeOf +1 boy.n.01 be.v.03 Theme -1 Time +1 Location +2 time.n.08 EQU now zoo.n.01`
- `[p27/d2600]` There was a terrible accident on the freeway.
    - `entity.n.01 be.v.03 Time +1 EQU -1 Theme +3 Location +4 time.n.08 TPR now terrible.a.01 AttributeOf +1 accident.n.01 freeway.n.01`
- `[p21/d0879]` There were many rotten apples in the basket.
    - `entity.n.01 be.v.03 Time +1 EQU -1 Theme +3 Location +4 time.n.08 TPR now rotten.a.02 AttributeOf +1 apple.n.01 Quantity + basket.n.01`
- `[p40/d1393]` What continent is the world's largest desert on?
    - `continent.n.01 Name ? be.v.03 Location -1 Time +1 Theme +5 time.n.08 EQU now world.n.04 large.a.01 Degree +1 most.r.01 desert.n.01 PartOf -3 Attribute -2`
- `[p10/d0831]` You are on the wrong train.
    - `person.n.01 EQU hearer be.v.03 Theme -1 Time +1 Location +3 time.n.08 EQU now wrong.a.03 AttributeOf +1 train.n.01`
