# Representing Self-Repair in SBN

This document extends the SBN manual. Everything in the manual still holds;
this document adds one construction for self-repair and the rules for using
it.

A **self-repair** is a speaker replacing something they have just said:
"The gnat, I mean, the fly buzzes." The abandoned part ("gnat") is the
**reparandum**. The part that replaces it ("fly") is the **repair**. An
optional **interregnum** ("I mean", "no", "no wait", "actually") can sit
between them.

In this task you are given a gold sentence and its SBN. The gold sentence
contains the repair. You add a reparandum before it: something the speaker
could plausibly have said by mistake in the same slot. You will be told which
**type** of repair to make:

| Type | What the speaker gets wrong | Example | Section |
|---|---|---|---|
| content word | a word: one concept is replaced by a different concept | "The gnat, I mean, the fly buzzes." | §4.1 |
| constant | a name, number, time or date: the same concept with a different constant | "Jim, I mean, Tom kicked the can." | §4.2 |
| tense | the tense of a verb | "A dog chases, I mean, chased a cat." | §4.3 |
| edge label | a preposition: the same event with a different role to the same thing | "The bee flew from, I mean, to the flower." | §4.4 |
| forwarding | the start of the clause: the speaker restarts with a new subject and repeats the verb | "Jim bought, no, Tom bought a gun." | §4.5 |

---

## 1. The construction

```
… CORRECTION <1 reparandum … CONJUNCTION <2 repair … rest of the sentence
```

- `CORRECTION <1` opens a new box right before the reparandum. This box holds
  what the speaker took back. It attaches to the box that was live before
  it (the *parent box*).
- `CONJUNCTION <2` opens a second box right before the repair. `<2` counts
  back over the `CORRECTION` box to the same parent box. `CONJUNCTION` does
  not create a new scope: everything in its box is merged into the parent
  box, so the repair and the rest of the sentence count as asserted.
- The two new boxes are **siblings**: both hang off the parent box, neither
  is inside the other.
- Everything after the repair, up to the next separator, stays in the
  `CONJUNCTION` box. That is correct, because the box merges into the parent.

The reparandum is one concept for content-word, constant and edge-label
repairs, and a short span of concepts for tense and forwarding repairs (§4).

The connectors are always `CORRECTION <1` and `CONJUNCTION <2`.

The interregnum, if any, goes in the sentence only. It is not written in the
SBN.

---

## 2. Rules

**R1. Indices count across all boxes.** As in the manual: every concept in
the sequence counts, including the reparandum and concepts in closed boxes.
Inserting a reparandum therefore shifts every index that reaches across it.

**R2. Nothing outside may point into the `CORRECTION` box with a positive
index.** A concept before the `CORRECTION` box cannot hook forward onto the
reparandum.

**R3. The reparandum may only point backward, to the parent box or its
ancestors, or within its own box.** It may not point forward into the
`CONJUNCTION` box, because that box is its sibling.

**R4. Pointing *over* the `CORRECTION` box into the `CONJUNCTION` box is
allowed.** A concept before the `CORRECTION` box may hook forward onto the
repair or onto later material; the index simply counts the reparandum as one
of the positions it skips.

**R5. The reparandum must be connected.** It must carry at least one role or
operator that points to another concept. A constant alone (`Name "Jim"`)
does not connect it: a reparandum with no edge to another concept is not
linked to the sentence at all.

**R6. Box pointers after the repair must be updated.** The construction adds
two boxes. A separator *after* the repair whose connector reaches back to a
box *before* the repair now has to count over the two new boxes, so its
number goes up by 2 (e.g. `CONTRAST <1` becomes `CONTRAST <3`). The same holds
for `Proposition >n` pointing from before the repair to a box after it.

**R7. Keep the rest of the gold SBN unchanged.** Apart from the inserted
reparandum, the two separators, what the repair type and the devices below
require, and the index and box-pointer updates these cause, every concept,
role and constant of the gold SBN stays as it was.

---

## 3. The repair toolkit

R2, R3 and R5 often conflict with the edges the repaired concept has in the
gold SBN. The devices below resolve the conflict. Look at each edge that
touches the repaired concept and decide.

**Edges going out of the repaired concept** (its own roles):

| Situation | Device |
|---|---|
| target is before the `CORRECTION` box | the reparandum copies the role, pointing backward (allowed by R3) |
| target is after the repair | **① drop the role** on the reparandum. The speaker abandoned the word before saying that argument, so the reparandum never had it. The repair keeps it |

**Edges coming into the repaired concept** (another concept's role pointing
at it):

| Situation | Device |
|---|---|
| source is after the repair | the source points to the repair only |
| source is before the `CORRECTION` box | **② invert the role**: remove it from the source, and give both reparandum and repair the inverted role pointing backward to the source. Any role can be inverted (manual §2.3); a role that is already inverted is turned back (`AttributeOf` on the source becomes `Attribute` on both nouns). Operators are inverted the same way, by moving a symmetric operator or using the converse one (manual §2.4) |
| source is before the `CORRECTION` box, and the edge cannot be inverted (`Of`, `MadeOf`, `FeatureOf`, `ANA`, `TAB`, `STI`, `STO`, `SY2`, `SXY`) | the source keeps the edge and points **over** the `CORRECTION` box to the repair (R4). The reparandum then needs some other edge (R5) |

**When the reparandum would still have no edge to another concept** (R5):

| Situation | Device |
|---|---|
| the repaired concept is a noun (a referent) | **⑤ anchor**: insert `entity.n.01` right before `CORRECTION`. Every edge that came into the repaired concept now points to this anchor instead. Reparandum and repair each get `EQU` pointing back to the anchor |
| the repaired concept is an adjective, adverb or verb | **④ movement**: move the concepts it points forward to (and their own attachments) in front of the `CORRECTION` box, so that both reparandum and repair can point back at them. This breaks surface order, so use it only here |

Do not use ⑤ for adjectives, adverbs or verbs: `entity.n.01 EQU -1` on an
adjective or event makes no sense.

Several devices can combine in one repair (see example F).

---

## 4. Repair types

### 4.1 Content word

Pick one concept with a WordNet synset, other than a placeholder
(`time.n.08`, `entity.n.01`, `person.n.01`, `male.n.02`, `female.n.02`,
`quantity.n.01`, `location.n.01`). The reparandum is a **different**
concept of the same part of speech that fits the same slot: a word the
speaker could have said by mistake, not a synonym. Check that its synset
exists. Apply the toolkit (§3). Examples A–K.

### 4.2 Constant

Pick a concept that carries a constant: a name (`Name "Tom"`), a number
(`Quantity 3`), or a time or date (`ClockTime 22:00`, `DayOfWeek friday`,
`MonthOfYear`, `YearOfCentury`). The reparandum is a **copy of the same
concept with a different value** of the same kind: another number, another
weekday, or another name that fits the concept. A name on `male.n.02` or
`female.n.02` is a person's name of that gender; a name on `country.n.02`,
`city.n.01`, `company.n.01` and the like is another country, city or company. Everything else works as for a
content word, including the toolkit. Examples L–N.

### 4.3 Tense

Pick a verb whose tense is expressed by a `time.n.08` concept directly next
to it (`time.n.08 TPR now` for past, `EQU now` for present, `TSU now` for
future). The reparandum is a **copy of the verb and its `time.n.08`**, with
a different tense operator. Both go into the `CORRECTION` box; the verb and
its `time.n.08` in the `CONJUNCTION` box are the gold ones. The verb keeps
its synset. In the sentence, the verb is inflected for each tense ("chases,
I mean, chased"). Apply ① to the copied verb's forward roles. Example O.

### 4.4 Edge label

Pick an event with a role that a preposition expresses (`Destination` "to",
`Source` "from", `Location` "in"). The speaker first used a different
preposition. A role cannot stand alone in a box, so each version of the role
is carried by a dummy `entity.n.01` (**device ③**):

- remove the role from the event;
- make sure the role's target comes **before** the `CORRECTION` box; if it
  comes after, move it in front (④);
- reparandum: `entity.n.01 EQU <event> <wrong role> <target>`;
- repair: `entity.n.01 EQU <event> <gold role> <target>`.

This is the one place where `entity.n.01 EQU` points to an event. Example P.

### 4.5 Forwarding

The speaker starts the clause, stops after the verb, and restarts with a
different subject, repeating the verb. Pick a subject that carries a name or
is a noun, with its verb right after it. The reparandum is a **copy of the
subject, the verb and the verb's `time.n.08`**, with a different subject:

- insert an anchor `entity.n.01` right before `CORRECTION` (⑤); both subjects
  carry `EQU` back to it;
- the copied verb points to the copied subject, and drops every role that
  points after the repair (①);
- the gold subject, verb and `time.n.08` follow in the `CONJUNCTION` box.

Example Q.

---

## 5. Examples

Each example gives the sentence, the gold SBN, and the repaired SBN.

### Content word

**A. Verb, no device needed.** "The boy sneezed, I mean, coughed."

```
gold:   boy.n.01 time.n.08 TPR now cough.v.01 Agent -2 Time -1
repair: boy.n.01 time.n.08 TPR now CORRECTION <1 sneeze.v.01 Agent -2 Time -1 CONJUNCTION <2 cough.v.01 Agent -3 Time -2
```

All of `cough.v.01`'s roles point backward, so the reparandum copies them.
The repair's indices grow by 1, because the reparandum now sits between.

**B. Verb with an object: ①.** "The cat chases, actually, hunts the rat."

```
gold:   cat.n.01 time.n.08 EQU now hunt.v.01 Agent -2 Theme +1 Time -1 rat.n.01
repair: cat.n.01 time.n.08 EQU now CORRECTION <1 chase.v.01 Agent -2 Time -1 CONJUNCTION <2 hunt.v.01 Agent -3 Theme +1 Time -2 rat.n.01
```

`chase.v.01` has no `Theme`: the rat comes after the repair (R3), and the
speaker had not said it yet when abandoning "chases".

**C. Object: ②.** "Cats catch abrocomes, I mean, mice."

```
gold:   cat.n.01 catch.v.04 Agent -1 Theme +1 mouse.n.01
repair: cat.n.01 catch.v.04 Agent -1 CORRECTION <1 abrocome.n.01 ThemeOf -1 CONJUNCTION <2 mouse.n.01 ThemeOf -2
```

`catch.v.04 Theme +1` would point into the `CORRECTION` box (R2). It is
inverted: both nouns carry `ThemeOf` back to the verb.

**D. Subject: ⑤.** "The gnat, I mean, the fly buzzes."

```
gold:   fly.n.01 buzz.v.01 Agent -1
repair: entity.n.01 CORRECTION <1 gnat.n.01 EQU -1 CONJUNCTION <2 fly.n.01 EQU -2 buzz.v.01 Agent -3
```

`fly.n.01` has no roles of its own, and the only edge into it (`Agent`) comes
from after the repair, so without ⑤ the reparandum would have no edge. The
anchor carries the subject; `buzz.v.01 Agent` points to the anchor.

**E. Adjective: ④.** "How grey, I mean, blue the sky is!"

```
gold:   blue.a.01 ColourOf +1 Time +2 sky.n.01 time.n.08 EQU now
repair: sky.n.01 time.n.08 EQU now CORRECTION <1 grey.a.01 ColourOf -2 Time -1 CONJUNCTION <2 blue.a.01 ColourOf -3 Time -2
```

Both of `blue.a.01`'s roles point forward, so the reparandum would have no
edge, and ⑤ does not suit an adjective. `sky.n.01` and `time.n.08` are moved
in front, and both adjectives point back at them.

**F. Object with a modifier: ② twice + ①.** "The government has imposed a
new levy, I mean, tax on wine."

```
gold:   government.n.01 impose.v.03 Agent -1 Theme +2 new.a.01 AttributeOf +1 tax.n.01 Theme +1 wine.n.01
repair: government.n.01 impose.v.03 Agent -1 new.a.01 CORRECTION <1 levy.n.01 ThemeOf -2 Attribute -1 CONJUNCTION <2 tax.n.01 Theme +1 ThemeOf -3 Attribute -2 wine.n.01
```

Two edges come into `tax.n.01` from before the `CORRECTION` box, and both
are inverted (②): `impose.v.03 Theme` becomes `ThemeOf` on both nouns, and
`AttributeOf` on `new.a.01` is turned back into `Attribute` on both nouns.
The speaker said "new" before "levy", so the levy is new as well.
`tax.n.01 Theme +1 wine.n.01` points forward, so the reparandum drops it (①).

**G. Inside a negation.** "I didn't order a banana bread, I mean, an apple
pie."

```
gold:   person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 order.v.02 Agent -2 Time -1 Theme +1 apple_pie.n.01
repair: person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 order.v.02 Agent -2 Time -1 CORRECTION <1 banana_bread.n.01 ThemeOf -1 CONJUNCTION <2 apple_pie.n.01 ThemeOf -2
```

The boxes are: box 0, the `NEGATION` box 1, the `CORRECTION` box 2, the
`CONJUNCTION` box 3. `CONJUNCTION <2` counts back from box 3 to box 1, so the
repair merges into the **negated** box, which is what the sentence means.

**H. A separator after the repair: R6.** "Even though he was hungry, I mean,
tired, he worked."

```
gold:   male.n.02 time.n.08 TPR now tired.a.01 Experiencer -2 Time -1 CONTRAST <1 male.n.02 ANA -3 work.v.01 Agent -1 Time +1 time.n.08 TPR now
repair: male.n.02 time.n.08 TPR now CORRECTION <1 hungry.a.01 Experiencer -2 Time -1 CONJUNCTION <2 tired.a.01 Experiencer -3 Time -2 CONTRAST <3 male.n.02 ANA -4 work.v.01 Agent -1 Time +1 time.n.08 TPR now
```

`CONTRAST <1` pointed to box 0. Two boxes now sit in between, so it becomes
`CONTRAST <3`. `ANA -3` becomes `ANA -4` because it now counts over the
reparandum too (R1).

**I. Inside an embedded clause.** "Bill said you will put, I mean, drop the
ball on the table."

```
gold:   male.n.02 Name "Bill" say.v.01 Proposition >1 Agent -1 Time +1 time.n.08 TPR now CONTINUATION <0 person.n.01 EQU hearer time.n.08 TSU now drop.v.01 Agent -2 Time -1 Theme +1 Destination +2 ball.n.01 table.n.02
repair: male.n.02 Name "Bill" say.v.01 Proposition >1 Agent -1 Time +1 time.n.08 TPR now CONTINUATION <0 person.n.01 EQU hearer time.n.08 TSU now CORRECTION <1 put.v.01 Agent -2 Time -1 CONJUNCTION <2 drop.v.01 Agent -3 Time -2 Theme +1 Destination +2 ball.n.01 table.n.02
```

`Proposition >1` still points to the `CONTINUATION` box, which comes before
the repair, so it does not change. The reparandum drops `Theme` and
`Destination` (①).

**J. Operator: ②.** "It's under the desk, I mean, the table."

```
gold:   entity.n.01 be.v.03 Theme -1 Time +1 Location +2 time.n.08 EQU now location.n.01 SZN +1 table.n.02
repair: entity.n.01 be.v.03 Theme -1 Time +1 Location +2 time.n.08 EQU now location.n.01 CORRECTION <1 desk.n.01 SZP -1 CONJUNCTION <2 table.n.02 SZP -2
```

`location.n.01 SZN +1` ("the location is under the table") would point into
the `CORRECTION` box. It is replaced by its converse on both nouns: each is
`SZP` ("above") the location.

**K. Under universal quantification.** "Everybody loves, I mean, likes him."

```
gold:   NEGATION <1 person.n.01 NEGATION <1 like.v.03 Experiencer -1 Time +1 Stimulus +2 time.n.08 EQU now male.n.02
repair: NEGATION <1 person.n.01 NEGATION <1 CORRECTION <1 love.v.01 Experiencer -1 CONJUNCTION <2 like.v.03 Experiencer -2 Time +1 Stimulus +2 time.n.08 EQU now male.n.02
```

The repair sits in the second `NEGATION` box (box 2), so both new boxes hang
off box 2. `love.v.01 Experiencer -1` points back to `person.n.01` in box 1,
an ancestor, which is allowed. `Time` and `Stimulus` point forward and are
dropped (①).

### Constant

**L. Name: ⑤.** "Jim, I mean, Tom kicked the can."

```
gold:   male.n.02 Name "Tom" kick.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now can.n.01
repair: entity.n.01 CORRECTION <1 male.n.02 Name "Jim" EQU -1 CONJUNCTION <2 male.n.02 Name "Tom" EQU -2 kick.v.01 Agent -3 Time +1 Theme +2 time.n.08 TPR now can.n.01
```

`Name "Jim"` is a constant, so on its own it does not connect the reparandum
(R5). As with the subject in D, the anchor carries the subject.

**M. Number: ②.** "Tom has been arrested two, I mean, three times."

```
gold:   male.n.02 Name "Tom" time.n.08 TPR now arrest.v.01 Patient -2 Time -1 Frequency +1 time.n.01 Quantity 3
repair: male.n.02 Name "Tom" time.n.08 TPR now arrest.v.01 Patient -2 Time -1 CORRECTION <1 time.n.01 Quantity 2 FrequencyOf -1 CONJUNCTION <2 time.n.01 Quantity 3 FrequencyOf -2
```

**N. Weekday: ②.** "The accident occurred on Thursday, no, on Friday."

```
gold:   accident.n.01 occur.v.01 Theme -1 Time +1 Time +2 time.n.08 TPR now time.n.08 DayOfWeek friday
repair: accident.n.01 occur.v.01 Theme -1 Time +1 time.n.08 TPR now CORRECTION <1 time.n.08 DayOfWeek thursday TimeOf -2 CONJUNCTION <2 time.n.08 DayOfWeek friday TimeOf -3
```

The event has two `Time` roles. Only the one to the weekday is inverted; the
tense `Time +1` stays.

### Tense

**O.** "A dog chases, I mean, chased a cat."

```
gold:   dog.n.01 chase.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now cat.n.01
repair: dog.n.01 CORRECTION <1 chase.v.01 Agent -1 Time +1 time.n.08 EQU now CONJUNCTION <2 chase.v.01 Agent -3 Time +1 Theme +2 time.n.08 TPR now cat.n.01
```

The reparandum is `chase.v.01` with its own present-tense `time.n.08`. Its
`Time +1` stays inside the `CORRECTION` box, which is allowed; its `Theme`
points after the repair and is dropped (①).

### Edge label

**P.** "The bee flew from, I mean, to the flower."

```
gold:   bee.n.01 fly.v.01 Theme -1 Time +1 Destination +2 time.n.08 TPR now flower.n.01
repair: bee.n.01 fly.v.01 Theme -1 Time +1 time.n.08 TPR now flower.n.01 CORRECTION <1 entity.n.01 EQU -3 Source -1 CONJUNCTION <2 entity.n.01 EQU -4 Destination -2
```

`Destination` is removed from `fly.v.01`. `flower.n.01` already comes before
the new boxes once the dummies are placed at the end. The rejected dummy
bears `Source`, the committed one `Destination`; both are `EQU` to the event.

### Forwarding

**Q.** "Jim bought, no, Tom bought a gun."

```
gold:   male.n.02 Name "Tom" buy.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now gun.n.01
repair: entity.n.01 CORRECTION <1 male.n.02 Name "Jim" EQU -1 buy.v.01 Agent -1 Time +1 time.n.08 TPR now CONJUNCTION <2 male.n.02 Name "Tom" EQU -4 buy.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now gun.n.01
```

The abandoned clause "Jim bought" sits whole in the `CORRECTION` box. Its
verb points to Jim, keeps its tense, and has no `Theme`: the gun was never
said under that plan (①).

---

## 6. Errors to avoid

Each repaired SBN below is **wrong**. Most are read by the SBN reader without
complaint, so nothing will warn you.

| Wrong | Why | Right |
|---|---|---|
| (A) `… CORRECTION <1 sneeze.v.01 Agent -2 Time -1 CONJUNCTION <2 cough.v.01 Agent -2 Time -1` | the repair's indices were copied from the reparandum without shifting: `Agent -2` now lands on `time.n.08`, `Time -1` on `sneeze.v.01` | `cough.v.01 Agent -3 Time -2` |
| (A) `… CONJUNCTION <1 cough.v.01 …` | `<1` attaches the `CONJUNCTION` box to the `CORRECTION` box, so the repair is merged into what was taken back | `CONJUNCTION <2` |
| (G) `… CONJUNCTION <3 apple_pie.n.01 …` | merges the repair into box 0 instead of the negated box: "I did order an apple pie" | `CONJUNCTION <2` |
| (B) `… CORRECTION <1 chase.v.01 Agent -2 Theme +2 Time -1 CONJUNCTION <2 …` | the reparandum points forward into its sibling `CONJUNCTION` box (R3) | drop the `Theme` (①) |
| (C) `cat.n.01 catch.v.04 Agent -1 Theme +1 CORRECTION <1 abrocome.n.01 CONJUNCTION <2 mouse.n.01` | `Theme +1` points into the `CORRECTION` box (R2), and the reparandum has no edge (R5) | invert to `ThemeOf` (②) |
| (C) `cat.n.01 catch.v.04 Agent -1 Theme +1 CORRECTION <1 abrocome.n.01 ThemeOf -1 CONJUNCTION <2 mouse.n.01 ThemeOf -2` | the role was inverted but also left on the source: `Theme +1` still points into the `CORRECTION` box (R2), and together with `ThemeOf -1` it forms a cycle, so the graph is no longer a DAG | remove `Theme +1` from `catch.v.04` |
| (E) `entity.n.01 CORRECTION <1 grey.a.01 EQU -1 CONJUNCTION <2 blue.a.01 EQU -2 …` | ⑤ on an adjective: a colour is equated with a thing | ④ |
| (H) `… CONTRAST <1 male.n.02 ANA -4 …` | the connector was not updated (R6): `CONTRAST` now attaches to the `CONJUNCTION` box | `CONTRAST <3` |
| (L) `CORRECTION <1 male.n.02 Name "Jim" CONJUNCTION <2 male.n.02 Name "Tom" kick.v.01 Agent -1 …` | the reparandum has only a constant and no edge to another concept (R5): Jim is not linked to the kicking | ⑤ |
| (O) `dog.n.01 CORRECTION <1 chase.v.01 Agent -1 Time +2 CONJUNCTION <2 chase.v.01 Agent -2 Time +1 Theme +2 time.n.08 TPR now cat.n.01` | the reparandum has no `time.n.08` of its own and points to the repair's (R3), so both versions share one tense and nothing was corrected | give the reparandum its own `time.n.08` |
| (P) `bee.n.01 fly.v.01 Theme -1 Time +1 Destination +2 time.n.08 TPR now flower.n.01 CORRECTION <1 …` | `Destination` was left on the event, so "to the flower" is asserted on the event as well as by the dummy | remove the role from the event |

---

## 7. What cannot be repaired

The construction can only quarantine concepts and their edges. Do not place a
repair on:

- **a word expressed by a separator**: negation ("not", "no one"),
  quantifiers built from `NEGATION` ("every", "all"), modals expressed by
  `POSSIBILITY` / `NECESSITY` ("can", "must"), or discourse connectives
  ("because", "but", "if");
- **a repair that needs material between reparandum and repair**: the
  repair must follow the reparandum directly ("three apples from Lidl
  yesterday, actually, four" is out).
