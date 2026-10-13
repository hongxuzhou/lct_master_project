# Labelling guideline: place self-repairs (natural only)

Each item is an English sentence with a spoken self-repair of a **place**:
the speaker first says one spatial preposition, then corrects it to another,
for the same verb. Examples of the format:

- bare form: "The money is under, on the table." (first version = "under")
- full form: "The money is under the table, on the table."

Both forms are fine as such; do not prefer one. The interregnum ("I mean")
is left out on purpose; do not judge its absence. `nl_clean` is the
original sentence (the corrected version); `first_version` is the
abandoned phrase.

## The data's reading (fixed, not up for judgement)

The meaning representation always reads the first version as a place
relative to **the same thing** as the second one: "under, in the barn"
means "first: under the barn; corrected: in the barn". It never means
"under some other thing the speaker did not name". So judge only this
reading. Do **not** rescue an item by imagining an unsaid, different
landmark ("he meant under the shed, then switched to the barn").

## Question

Could a fluent English speaker naturally produce this self-repair, under
the reading above? Answer `yes`, `no` or `unsure`.

Work through these checks in order; the first that fails gives `no`.

1. **Restart.** The repair repeats the whole clause (subject and verb
   again: "The party was beside a room, the party was in a room").
   → `no`, code `restart`.
2. **Substitution test.** Build the sentence with the first preposition
   and the same thing, and read it alone ("The farmer parks the car under
   the garage"). If that is impossible or absurd for this thing and this
   event (a place that has no "under", "near" for something you are
   inside of, a body part, the sky, the world, hiding something above
   where it would be seen, a thing that cannot be under/beside what it is
   said to be), → `no`, code `first_impossible`.
3. **Verb or fixed phrase.** The second preposition belongs to the verb
   or to a fixed expression rather than choosing a place, so the first
   version is not an alternative place but a broken phrase: "arrive
   in/at <city>", "run into someone on the street", "get on the bus",
   "lean on", "wrap in", "spread on", "pat on the shoulder", "in the
   wash", "on her way". → `no`, code `verb_locked`.
4. **No real change.** The two versions describe practically the same
   place, or one includes the other ("at, in the building", "near, beside
   the beach", "inside, in the box"). → `no`, code `same_place`.
5. Otherwise the first version is a place the speaker could have meant
   and then corrected:
   - plausible situation → `yes`, code `ok`;
   - possible but clearly odd or unlikely, so you need a second thought
     ("put a stamp under the letter", "stayed near home") → `unsure`,
     code `odd`.

Be lenient: give `yes` when you find no concrete reason for `no` or
`unsure`. Odd-but-sensible scenes a person could plausibly blame on the
speaker ("I left your book under the car, in the car") are `yes`.
Use `no` with code `other` for anything else that makes it unnatural
(e.g. ungrammatical first version, the repair breaks the sentence), and
say why in the note.

Calibration (invented, not from the data):
- "Tom was waiting near, at the station." → yes, ok
- "She found the keys under the sofa, behind the sofa." → yes, ok
- "Birds were flying under, in the air." → no, first_impossible
- "We finally arrived near, in Paris." → no, verb_locked
- "He hid the letter above, under the mattress." → no, first_impossible
- "The cat sleeps near, in its basket." → yes, ok
- "He glued the label under the jar, on the jar." → unsure, odd

## Output

A TSV with header `audit_id	label	code	note`, one row per input item, in
input order. `label` ∈ yes/no/unsure; `code` ∈ ok, odd, restart,
first_impossible, verb_locked, same_place, other; `note` ≤ 15 words, may
be empty for `yes ok`. No tabs inside fields.
