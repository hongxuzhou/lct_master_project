# Your task

You are an annotator. You will be given an English sentence with its gold
SBN, and you will turn it into a sentence with a self-repair and write the
SBN for that new sentence by hand, following the SBN manual and the
self-repair document above.

## Input

Each request gives you:

- **Repair type**: one of `content_word`, `constant_name`,
  `constant_number`, `constant_temporal`, `tense`, `edge_label`,
  `forwarding` (self-repair document §4; the three `constant_` types are
  §4.2 with a name, a number, or a time or date).
- **Interregnum**: the words to put between reparandum and repair
  ("I mean", "no", "no wait", "actually"), or `none`, in which case
  the repair follows the reparandum directly after a comma.
- **Gold sentence** and **gold SBN**.

## Steps

1. **Number the concepts** of the gold SBN from 0, left to right. You will
   need these positions for every index.
2. **Choose the repair site**: a concept that matches the repair type
   (self-repair document §4), and not one of the places in §7. If the
   sentence has no such site, stop and report `no_site` (see the output
   format). Do not change the repair type.
3. **Choose the reparandum**: something the speaker could plausibly have
   said by mistake in that slot. It must differ in meaning from the repair;
   a synonym is not a self-repair.
   - For a new synset (content word, forwarding with a common noun), look
     it up before you use it (see "The WordNet lookup" below).
   - Placeholder concepts (`time.n.08`, `entity.n.01`, `person.n.01`,
     `male.n.02`, `female.n.02`, `quantity.n.01`, `location.n.01`) are not
     content words. Do not choose one as the repair site or as the
     reparandum of a content-word repair.
   - For a constant, keep the kind of value and keep the concept: a man's
     name on `male.n.02`, a woman's name on `female.n.02`, a country on
     `country.n.02`, a number for a number, a weekday for a weekday.
4. **Write the repaired sentence.** Insert the reparandum, the interregnum
   (if any) and the repair into the gold sentence, with commas around the
   interregnum. Inflect the reparandum to fit ("chases, I mean, chased").
   Repeat a determiner or preposition before the repair where a speaker
   would ("the gnat, I mean, the fly"). Leave the rest of the sentence as it
   is.
5. **Write the repaired SBN.** Start from the gold SBN. Insert
   `CORRECTION <1`, the reparandum, `CONJUNCTION <2` and the repair as the
   repair type requires. Apply the toolkit (§3). Then recompute every index
   and every box pointer (R1, R6).
6. **Check your SBN against R1–R7** before you answer. In particular,
   recount every index that crosses the inserted material, and confirm that
   the reparandum has an edge to another concept.
7. **List the devices** you used.

## The WordNet lookup

SBN concepts are written `lemma.pos.NN`, where `NN` is the sense number of
that lemma in WordNet 3.0. You cannot guess sense numbers, so look them up:

```
python3 wn_lookup.py senses <lemma> <pos>
python3 wn_lookup.py check <lemma.pos.NN>
```

`senses` lists every sense of a word with its definition; pick the one that
fits the sentence and copy its name exactly. `check` confirms a name exists.
`pos` is `n`, `v`, `a` or `r`. Use the lemma (`chase`, not `chased`).

## Output format

You may think step by step first. End your answer with exactly one JSON
object in a ```json block. Nothing may follow it.

```json
{
  "status": "ok",
  "reason": "",
  "site_index": 0,
  "site_synset": "fly.n.01",
  "reparandum_synset": "gnat.n.01",
  "reparandum_value": "",
  "reparandum_surface": "gnat",
  "nl_repair": "The gnat, I mean, the fly buzzes.",
  "sbn_repair": "entity.n.01 CORRECTION <1 gnat.n.01 EQU -1 CONJUNCTION <2 fly.n.01 EQU -2 buzz.v.01 Agent -3",
  "devices": ["d5_anchor_dummy"]
}
```

| Field | Content |
|---|---|
| `status` | `"ok"`, or `"no_site"` if the sentence offers no site of the requested type |
| `reason` | for `no_site`, one sentence on why; otherwise `""` |
| `site_index` | position (from 0) in the **gold** SBN of the repaired concept. For `tense` the verb, for `edge_label` the event, for `forwarding` the subject |
| `site_synset` | that concept's synset, as written in the gold SBN |
| `reparandum_synset` | the reparandum's synset. Same as `site_synset` for `constant_*` and `tense`; `entity.n.01` for `edge_label`; the new subject's synset for `forwarding` |
| `reparandum_value` | what differs, when the synset does not: the new constant (`"Jim"` with quotes for a name, `2`, `thursday`), the new tense operator (`TSU now`), or the wrong role (`Source`). Otherwise `""` |
| `reparandum_surface` | the reparandum's words as they appear in `nl_repair` |
| `nl_repair` | the repaired sentence |
| `sbn_repair` | the repaired SBN, on one line, no `%` comments, tokens separated by single spaces |
| `devices` | the devices used, from: `d1_drop_role` (①), `d2_invert_role` (②), `d3_dummy_equ` (③), `d4_movement` (④), `d5_anchor_dummy` (⑤). `[]` if none |

For `no_site`, fill `status` and `reason`, and leave every other field empty
(`""`, `-1` for `site_index`, `[]` for `devices`).
