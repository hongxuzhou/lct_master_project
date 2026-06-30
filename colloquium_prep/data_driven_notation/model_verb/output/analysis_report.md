# Modal verbs in PMB-5.1.0 gold.sbn — SBN correspondence

- Blocks scanned: **9552**
- Blocks containing >=1 core modal: **330** (3.5%)
- Core modal tokens detected: **333**
- Detection: spaCy `en_core_web_sm` PTB **MD** tag + NER guard + `'d`→`had` guard.

## Per-modal → SBN operator mapping

Counts are *blocks* in which the modal co-occurs with each SBN operator (a block may have several).

| modal | hits | POSSIBILITY | NECESSITY | TSU(future) | NEGATION | none of P/N/TSU |
|-------|-----:|-----------:|---------:|-----------:|--------:|---------------:|
| can | 96 | 94 | 0 | 0 | 58 | 2 |
| could | 14 | 14 | 0 | 0 | 6 | 0 |
| may | 8 | 8 | 0 | 0 | 1 | 0 |
| might | 3 | 3 | 0 | 0 | 0 | 0 |
| must | 16 | 0 | 16 | 4 | 3 | 0 |
| should | 19 | 0 | 19 | 2 | 4 | 0 |
| will | 169 | 0 | 0 | 169 | 33 | 0 |
| would | 8 | 0 | 0 | 8 | 1 | 0 |

**Reading of the mapping**

- `can / could / may / might` → **POSSIBILITY <1** (epistemic/ability/permission).
- `must / should / ought (to)` → **NECESSITY <1** (deontic/epistemic necessity).
- `will / would` → **time.n.08 TSU now** (future = *time successor*); they do *not* introduce a modal-force operator. A few `must`/`should` blocks also carry `TSU` when the obligation is future-oriented.
- Negated modals stack **NEGATION <1** on top, e.g. `can't` → `NEGATION <1 POSSIBILITY <1`.
- `shall` and `ought` have **0** occurrences in this split; `used to` / `dare` / `had better` are semi-modals (not MD) and are listed separately, not counted above.

## The separator question

All three modal devices (`POSSIBILITY`, `NECESSITY`, `TSU`) are realised as **in-box operators** scoping the next node via the `<1` index — they do **not** open a new box and are **not** discourse separators.

- Modal blocks that *also* contain a separator (CONTINUATION/CONJUNCTION/CONTRAST/CONSEQUENCE/…): **70/330**.
- In those blocks the separator is licensed by **clause structure** (subordination / coordination: *if…*, *…that…*, *…but…*, reported speech), **not** by the modal itself. The modal's own contribution stays in-box.

Example separator co-occurrences (modal operator stays inside its box):

- `p63/d2848` — I'll take back everything I said.
  - `person.n.01 EQU speaker NEGATION <1 NEGATION <1 time.n.08 TSU now take_back.v.05 Agent -2 Time -1 Theme +1 CONJUNCTION <2 entity.n.01 person.n.01 EQU speaker say.v.01 Topic -2 Agent -1 Time +1 time.n.08 TPR now`
- `p00/d3233` — The doctor told Tom that he should eat a lot of vegetables.
  - `person.n.01 Role +1 doctor.n.01 tell.v.02 Proposition >1 Agent -2 Time +1 Recipient +2 time.n.08 TPR now male.n.02 Name "Tom" CONTINUATION <0 male.n.02 ANA -1 NECESSITY <1 eat.v.01 Agent -1 Patient +1 entity.n.01 Quantity + EQU +1 vegetable.n.01`
- `p58/d1526` — If the ceiling fell, he would be crushed.
  - `ceiling.n.01 fall.v.01 Theme -1 Time +1 time.n.08 TPR now CONSEQUENCE <1 male.n.02 time.n.08 TSU now crush.v.02 Patient -2 Time -1`
- `p12/d2112` — If I don't have a bow, I can't play the violin.
  - `person.n.01 EQU speaker NEGATION <1 time.n.08 EQU now have.v.01 Pivot -2 Time -1 Theme +1 bow.n.02 CONSEQUENCE <1 person.n.01 EQU speaker NEGATION <1 POSSIBILITY <1 play.v.07 Agent -1 Theme +1 violin.n.01`
- `p09/d2150` — According to the paper, it will snow tomorrow.
  - `according.a.02 Proposition >1 Source +1 paper.n.03 CONTINUATION <0 entity.n.01 time.n.08 TSU now snow.v.01 EQU -2 Time -1 Time +1 time.n.08 TIN +2 day.n.03 TCT now TAB +1 day.n.03`
- `p27/d1965` — I like this song; it's got a strong beat and you can dance to it.
  - `person.n.01 EQU speaker like.v.02 Experiencer -1 Time +1 Stimulus +2 time.n.08 EQU now song.n.01 CONTINUATION <1 entity.n.01 ANA -1 have_got.v.01 Theme -1 Time +1 Pivot +3 time.n.08 EQU now strong.a.01 AttributeOf +1 beat.n.03 CONTINUATION <1 person.n.01 POSSIBILITY <1 dance.v.02 Agent -1 Theme +1 entity.n.01 ANA -8`

## Implication for the self-repair (CORRECTION + CONJUNCTION) design

Modals give a clean precedent: a *propositional-attitude / force* meaning is encoded **without a separator**, as an indexed in-box operator (`OP <1`). A self-repair marker that is *purely operator-like* could follow the same in-box pattern; a marker that genuinely **re-segments** the utterance (replaces or coordinates spans) is what should justify a separator — which is precisely the CONTRAST / CONJUNCTION family, not the modal family.

## Files

- `modal_samples.jsonl` — full per-block records.
- `modal_samples.csv` — one row per (block, modal) with operator flags.
- `modal_sbn_mapping.csv` — the matrix above.
- `semimodal_samples.csv` — need(n't)/dare/used-to/had-better (not in core counts).
- `review_possible_fp.csv` — audit list for **manual post-processing**: `'d`=had cases auto-excluded, plus kept-but-suspect noun homographs (`This can is leaking`, `thieves' cant`) to delete by hand if desired.
