# Multi-sentence / direct-speech candidates to pull from pilot set

> Read-only analysis of `pilot_dataset_cleaned.tsv` (865 rows). **No data was modified.**
> Segmentation: spaCy `en_core_web_sm` (statistical SBD, abbreviation-aware).
> Goal: separate samples that are NOT a single coherent syntactic unit, because the
> sequence-position (front/mid/tail) reparandum insertion is only methodologically
> stable within one sentence/clause-complex.

## Summary
| | count |
|---|---|
| Total rows | 865 |
| **Candidates to pull** | **29** |
| &nbsp;&nbsp;Cat 1 — multi-sentence (no dialogue) | 25 |
| &nbsp;&nbsp;Cat 2 — multi-sentence dialogue | 3 |
| &nbsp;&nbsp;Cat 3 — single-sentence direct speech | 1 |
| Remaining single-sentence set | 836 |

## Decision rule used
- **Criterion A — sentence boundary:** spaCy yields ≥ 2 sentences. (Correctly keeps
  `Mr./Mrs./Mt./Dr./St.`, decimals, etc. as single sentences; correctly splits the
  initial-then-newsentence case `…Baron von S. His wife…`.)
- **Criterion B — direct speech:** a quotation co-occurring with a reporting verb
  (said/asked/…) or paired quoted utterances each ending in terminal punctuation.
- Coordinated single sentences (`A, but B`) are KEPT — one syntactic unit.

---

## Cat 1 — Multi-sentence, no dialogue (25)
- **p11/d3265** — This stone is beautiful. Where did you find it?
    - [1] This stone is beautiful.
    - [2] Where did you find it?
- **p65/d0897** — In Esperanto, nouns end with "o". Plurals are formed with the addition of "j".
    - [1] In Esperanto, nouns end with "o".
    - [2] Plurals are formed with the addition of "j".
- **p73/d1730** — Cycling is good exercise. Moreover, it doesn't pollute the air.
    - [1] Cycling is good exercise.
    - [2] Moreover, it doesn't pollute the air.
- **p69/d0896** — The Titanic sank on her maiden voyage. She was a large ship.
    - [1] The Titanic sank on her maiden voyage.
    - [2] She was a large ship.
- **p56/d1404** — We tried to put out the fire but we were unsuccessful. We had to call the fire brigade.
    - [1] We tried to put out the fire but we were unsuccessful.
    - [2] We had to call the fire brigade.
- **p51/d0076** — Frank Lampard passed the ball to Beckham. Beckham kicked the ball and scored!
    - [1] Frank Lampard passed the ball to Beckham.
    - [2] Beckham kicked the ball and scored!
- **p02/d1539** — The weather is clearing up. I needn't have brought an umbrella.
    - [1] The weather is clearing up.
    - [2] I needn't have brought an umbrella.
- **p13/d1992** — Tom doesn't have a cat. However, Tom does have a dog, doesn't he?
    - [1] Tom doesn't have a cat.
    - [2] However, Tom does have a dog, doesn't he?
- **p67/d3028** — I have a Vietnamese friend. Her name is Tiên.
    - [1] I have a Vietnamese friend.
    - [2] Her name is Tiên.
- **p05/d1601** — She is called Mei. She is cooking in the kitchen.
    - [1] She is called Mei.
    - [2] She is cooking in the kitchen.
- **p55/d1232** — The indictment contains 875 pages. The trial is expected to last several years.
    - [1] The indictment contains 875 pages.
    - [2] The trial is expected to last several years.
- **p86/d3353** — Tom is by no means unintelligent. He is just lazy.
    - [1] Tom is by no means unintelligent.
    - [2] He is just lazy.
- **p50/d3043** — My wife has faults. None the less, I love her.
    - [1] My wife has faults.
    - [2] None the less, I love her.
- **p90/d3065** — I cannot wipe the table. I don't have a cloth.
    - [1] I cannot wipe the table.
    - [2] I don't have a cloth.
- **p00/d2718** — Tom can't speak French. Tom can't speak Spanish either.
    - [1] Tom can't speak French.
    - [2] Tom can't speak Spanish either.
- **p07/d3318** — The concert starts at seven. We must not be late.
    - [1] The concert starts at seven.
    - [2] We must not be late.
- **p60/d3120** — Cat Stevens is not a cat. He's a singer.
    - [1] Cat Stevens is not a cat.
    - [2] He's a singer.
- **p18/d1810** — Tom is kind of tired. He wants to go home.
    - [1] Tom is kind of tired.
    - [2] He wants to go home.
- **p31/d1973** — Queen Anne knighted Newton in 1705. He was the first scientist to be knighted for his work.
    - [1] Queen Anne knighted Newton in 1705.
    - [2] He was the first scientist to be knighted for his work.
- **p80/d2373** — She stopped studying. She left the university without a degree.
    - [1] She stopped studying.
    - [2] She left the university without a degree.
- **p07/d0133** — No one is working. Everyone's watching the World Cup.
    - [1] No one is working.
    - [2] Everyone's watching the World Cup.
- **p60/d0775** — I'm very sensitive to cold. May I have another blanket?
    - [1] I'm very sensitive to cold.
    - [2] May I have another blanket?
- **p95/d0707** — Bush returned home as a war hero. He married Barbara and enrolled at Yale.
    - [1] Bush returned home as a war hero.
    - [2] He married Barbara and enrolled at Yale.
- **p15/d2660** — My pen is old. I want a new one.
    - [1] My pen is old.
    - [2] I want a new one.
- **p95/d1663** — Who fights can lose. Who doesn't fight has already lost.
    - [1] Who fights can lose.
    - [2] Who doesn't fight has already lost.

## Cat 2 — Multi-sentence dialogue (3)
- **p78/d0807** — "How old is she?" "She is twelve years old."
    - [1] "How old is she?"
    - [2] "She is twelve years old."
- **p09/d3109** — "Who is Norman Finkelstein?" "He is an American political scientist."
    - [1] "Who is Norman Finkelstein?"
    - [2] "He is an American political scientist."
- **p02/d1309** — "Terrible weather," said Baron von S. His wife looked nervous.
    - [1] "Terrible weather," said Baron von S.
    - [2] His wife looked nervous.

## Cat 3 — Single-sentence direct speech (1)
- **p23/d2513** — "Are you still cold," she asked, as she kissed him on the forehead.
    - [1] "Are you still cold," she asked, as she kissed him on the forehead.

---

## Deliberately KEPT (principled exclusions — bring these to the advisor too)

These look superficially like the targets but are single syntactic units; flagging
them as "decisions" shows the rule is not naive string matching.

### Title / scare quotes — single sentence, quoted proper noun, NO dialogue (11)
- **p83/d1387** — What ship did the oil tanker "New World" crash into?
- **p04/d1391** — Who played the title role in the movie "Ben-Hur"?
- **p56/d1383** — What war is connected with the book "Charge of the Light Brigade"?
- **p74/d3278** — The famous song "Ave Maria" was composed by Schubert.
- **p47/d0715** — "The Lord of the Rings" was directed by Peter Jackson.
- **p99/d0877** — "The Old Man and the Sea" is a novel by Hemingway.
- **p57/d1385** — Who is the author of the poem "The Midnight Ride of Paul Revere?"
- **p09/d1086** — Scrooge shivered, and wiped the perspiration from his brow."
- **p45/d1381** — What is the name of the heroine in "Gone with the Wind"?
- **p74/d3377** — Goethe's poem "Mignon" is widely read in Japan in Mori Ogai's excellent translation.
- **p25/d1381** — What was the name of the high school in "Grease"?

### Citation-marker noise — `[n]` artifacts, actually one sentence (1)
- **p00/d0317** — Padalecki was born in San Antonio, Texas, to Gerald and Sherri Padalecki.

### Borderline you should confirm with the advisor
- **p57/d1385** — `Who is the author of the poem "The Midnight Ride of Paul Revere?"`
  Single question; the `?` sits inside the closing quote (US convention), NOT dialogue. Kept.
- **p52/d0808** — `My brother is two years older than I, but he is three centimeters shorter.`
  Coordinated single sentence. Kept (matches your own judgement).
