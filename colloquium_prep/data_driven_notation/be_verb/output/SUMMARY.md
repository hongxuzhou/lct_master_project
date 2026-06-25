# `be` samples — extraction & be-node classification

- Source: `/Users/hongxuzhou/Documents/GitHub/lct_master_project/data/pmb-5.1.0/split/en/train/gold.sbn`
- Total samples parsed: **9552**
- Samples containing a be-form (be/am/is/are/being/was/were): **3483** (36.5% of corpus)

## Step 2 — classification by SBN be-node

| class | count | share of be-form samples |
|---|---:|---:|
| has `be.*` node | 947 | 27.2% |
| no be node | 2536 | 72.8% |

## be-node sense distribution (within `has_be_node`)

| sense | count |
|---|---:|
| `be.v.03` | 321 |
| `be.v.02` | 312 |
| `be.v.01` | 207 |
| `be.v.08` | 102 |
| `be.v.06` | 40 |
| `be.v.13` | 9 |
| `be.v.04` | 2 |
| `be.v.05` | 1 |

## Surface be-form vs. be-node presence

Per distinct surface form, how often samples containing it also carry an SBN be-node. High rates support the data-driven mapping.

| be-form | samples | with be-node | rate |
|---|---:|---:|---:|
| be | 68 | 16 | 23.5% |
| am | 144 | 21 | 14.6% |
| is | 1926 | 627 | 32.6% |
| are | 416 | 161 | 38.7% |
| being | 16 | 5 | 31.2% |
| was | 821 | 113 | 13.8% |
| were | 129 | 17 | 13.2% |
