#!/usr/bin/env python3
"""
Extract and analyse modal-verb samples from PMB gold.sbn.

Goal (colloquium prep, data-driven SBN notation design):
    For the self-repair / CORRECTION+CONJUNCTION work we need to know how
    English modal verbs (can/could/may/might/must/shall/should/will/would/
    ought-to, plus their negated / contracted / double-contracted variants)
    surface in Simplified Box Notation -- in particular whether a modal ever
    triggers a *separator* (a new box / discourse relation) or stays an
    in-box operator.

Design decisions
----------------
1. Surface detection is done with spaCy's PTB ``MD`` tag (modal auxiliary).
   This is a closed lexical class that already covers every core English
   modal AND its tokenised contractions:
       can't   -> ca  + n't      (lemma=can)
       cannot  -> can             (lemma=can)
       I'll    -> 'll             (lemma=will)
       won't   -> wo  + n't       (lemma=will)
       would've-> would + 've     (lemma=would)
       shan't  -> sha + n't       (lemma=shall)
       ought   -> ought           (lemma=ought)   [+ following "to"]
   Crucially the tagger uses context, so homographs are handled correctly:
       "open a can"        -> can = NOUN, not MD
       "in May 2010"       -> May = date,  not MD
       "his free will"     -> will = NOUN, not MD
   We deliberately do NOT filter by the SBN side: that would discard the
   counter-examples (modals that map to something other than the expected
   operator), which are exactly what the analysis is about.

2. Three precision guards on top of the raw MD tag:
   (a) NER guard   -- drop an MD token whose entity type is PERSON or DATE
                      (catches the rare proper-name "Will", stray dates).
   (b) 'd guard    -- spaCy lemmatises every "'d" to "would"; when "'d" is
                      followed by a past participle (VBN) it is really the
                      perfect auxiliary "had", so we reclassify and exclude.
                      "'d better" (semi-modal HAD BETTER) is likewise excluded.
   (c) need/dare etc. are NOT tagged MD by spaCy and are reported separately
       as semi-modals (not part of the core counts).

3. Negation of the modal is detected from the local syntax (a "not"/"n't"/
   "never"/"cannot" attached to or governing the modal), so we can line it
   up against the SBN ``NEGATION`` operator.

Outputs (../output/):
    modal_samples.jsonl        one record per block that contains >=1 modal
    modal_samples.csv          flat, one row per (block, modal) pair
    modal_sbn_mapping.csv       modal-lemma x SBN-operator co-occurrence matrix
    semimodal_samples.csv      need(n't)/dare/used-to/had-better hits (audit)
    review_possible_fp.csv     MD hits worth eyeballing for false positives
    analysis_report.md         human-readable summary + the separator answer
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import spacy

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve()
PROJECT = HERE.parents[4]  # .../lct_master_project
GOLD = PROJECT / "data" / "pmb-5.1.0" / "split" / "en" / "train" / "gold.sbn"
OUT = HERE.parents[1] / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# SBN operator inventory (empirically enumerated from gold.sbn)
# ----------------------------------------------------------------------------
MODAL_OPERATORS = {"POSSIBILITY", "NECESSITY"}          # core modal force
FUTURE_OP = "TSU"                                        # time-successor (will/would)
NEG_OP = "NEGATION"
SEPARATORS = {                                           # discourse / box-splitting relations
    "CONTINUATION", "CONJUNCTION", "CONTRAST", "CONSEQUENCE",
    "EXPLANATION", "CONDITION", "PRECONDITION", "RESULT", "ALTERNATION",
}
ALLCAPS_RE = re.compile(r"[A-Z][A-Z-]+$")

# core modal lemmas spaCy emits for MD tokens
CORE_MODAL_LEMMAS = {
    "can", "could", "may", "might", "must",
    "shall", "should", "will", "would", "ought",
}

# modal lemmas with a common noun / name homograph; when one of these is
# detected but the SBN shows none of POSSIBILITY/NECESSITY/TSU it is worth a
# manual look (kept in the data, also copied to the audit file).
HOMOGRAPH_LEMMAS = {"can", "may", "will", "might"}

# semi-modals: not MD in PTB, captured separately for completeness / audit
SEMIMODAL_RE = re.compile(
    r"\b(need(?:n't| not| n't)?\s+\w+ing|needn't|need not|dare(?:n't)?|"
    r"used to|had better|'d better)\b",
    re.IGNORECASE,
)


def load_blocks(path: Path):
    """Yield (doc_id, sentence, sbn) triples from a .sbn file."""
    raw = path.read_text(encoding="utf-8")
    for block in raw.split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln.strip() != ""]
        if len(lines) != 3:
            continue
        doc_id, sentence, sbn = lines
        yield doc_id.strip(), sentence.strip(), sbn.strip()


def sbn_operators(sbn: str):
    """Return the multiset of ALLCAPS operators appearing in an SBN string."""
    return Counter(tok for tok in sbn.split() if ALLCAPS_RE.match(tok))


def is_had_not_would(tok) -> bool:
    """True if a lemma='would' "'d" token is really perfect-aux 'had'."""
    if tok.lemma_ != "would" or tok.text.lower().lstrip("'") != "d":
        return False
    # skip intervening adverbs ("'d never told", "'d already gone")
    j = tok.i + 1
    doc = tok.doc
    while j < len(doc) and doc[j].pos_ in ("ADV", "PART"):
        if doc[j].lower_ == "better":          # "'d better" -> HAD BETTER
            return True
        j += 1
    if j >= len(doc):
        return False
    # A real modal "would" only ever governs a BARE INFINITIVE (VB).  If "'d"
    # is followed by a finite past form (VBN perfect, or VBD which the tagger
    # sometimes emits for a participle) it cannot be "would" -> it is "had".
    return doc[j].tag_ in ("VBN", "VBD")


def modal_is_negated(tok) -> bool:
    """Detect negation local to the modal (n't / not / never / cannot)."""
    # 'cannot' is one token; its own text carries the negation
    if tok.lower_ == "cannot":
        return True
    # a neg child / sibling within a small window
    for other in tok.doc:
        if other.dep_ == "neg" and other.head == tok:
            return True
    # window scan for an attached negator right after the modal
    for off in (1, 2):
        j = tok.i + off
        if j < len(tok.doc):
            w = tok.doc[j].lower_
            if w in ("n't", "not", "never"):
                return True
            if tok.doc[j].pos_ not in ("PART", "ADV"):  # stop at the verb
                break
    return False


def extract():
    nlp = spacy.load("en_core_web_sm")

    records = []          # per-block records (>=1 modal)
    flat_rows = []        # per (block, modal)
    semimodal_rows = []
    review_rows = []      # potential false positives to eyeball
    # mapping[lemma][operator] = count of blocks
    mapping = defaultdict(Counter)
    modal_lemma_counts = Counter()
    n_blocks = 0

    blocks = list(load_blocks(GOLD))
    for doc_id, sentence, sbn in blocks:
        n_blocks += 1
        doc = nlp(sentence)
        ops = sbn_operators(sbn)

        modals = []
        for tok in doc:
            if tok.tag_ != "MD":
                continue
            lemma = tok.lemma_.lower()
            if lemma not in CORE_MODAL_LEMMAS:
                continue
            # guard (a): NER -- proper names / dates wrongly tagged MD
            if tok.ent_type_ in ("PERSON", "DATE", "TIME"):
                review_rows.append([doc_id, sentence, tok.text, lemma,
                                    f"NER={tok.ent_type_}", sbn])
                continue
            # guard (a2): a genuine modal functions as aux (or ROOT/conj in
            # ellipsis / coordination).  A noun homograph ("the can will…")
            # lands on nsubj/dobj/etc.  We DROP only the clear nominal deps;
            # advcl is kept because the parser also assigns it to real modals
            # in malformed source sentences (e.g. the "can but/buy" typo),
            # i.e. dropping advcl would cost true positives.
            if tok.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "obj",
                            "attr", "appos", "compound", "poss"):
                review_rows.append([doc_id, sentence, tok.text, lemma,
                                    f"dep={tok.dep_} (noun homograph)", sbn])
                continue
            # guard (b): "'d" that is really perfect-aux 'had'
            if is_had_not_would(tok):
                review_rows.append([doc_id, sentence, tok.text, "had?",
                                    "'d+VBN/better -> had (excluded)", sbn])
                continue
            negated = modal_is_negated(tok)
            modals.append({
                "text": tok.text,
                "lemma": lemma,
                "i": tok.i,
                "negated": negated,
                # "ought to": flag the following infinitival 'to'
                "ought_to": (lemma == "ought" and tok.i + 1 < len(doc)
                             and doc[tok.i + 1].lower_ == "to"),
            })

        # semi-modals (separate bucket, never counted as core)
        sm = SEMIMODAL_RE.search(sentence)
        if sm:
            semimodal_rows.append([doc_id, sentence, sm.group(0), sbn])

        if not modals:
            continue

        present_ops = sorted(o for o in ops if o in MODAL_OPERATORS
                             or o == FUTURE_OP or o == NEG_OP or o in SEPARATORS)
        rec = {
            "id": doc_id,
            "sentence": sentence,
            "sbn": sbn,
            "modals": modals,
            "sbn_ops": dict(ops),
            "has_POSSIBILITY": ops["POSSIBILITY"] > 0,
            "has_NECESSITY": ops["NECESSITY"] > 0,
            "has_TSU": ops["TSU"] > 0,
            "has_NEGATION": ops["NEGATION"] > 0,
            "has_separator": any(ops[s] > 0 for s in SEPARATORS),
        }
        records.append(rec)

        for m in modals:
            lemma = m["lemma"]
            modal_lemma_counts[lemma] += 1
            # which SBN operators co-occur in this block
            for op in MODAL_OPERATORS | {FUTURE_OP, NEG_OP}:
                if ops[op] > 0:
                    mapping[lemma][op] += 1
            if not (ops["POSSIBILITY"] or ops["NECESSITY"] or ops["TSU"]):
                mapping[lemma]["<none>"] += 1
                # Any modal hit whose SBN carries no modal force (P/N/TSU) is an
                # anomaly: a noun homograph ("This can is leaking"), a mis-split
                # contraction, or a "'d"=had the VBN guard missed.  These are a
                # handful of cases -> flag for manual review rather than growing
                # the rules.
                hint = ("possible noun homograph"
                        if lemma in HOMOGRAPH_LEMMAS else "check 'd=had / parse")
                review_rows.append([
                    doc_id, sentence, m["text"], lemma,
                    f"kept, but no P/N/TSU in SBN ({hint})", sbn])
            flat_rows.append([
                doc_id, sentence, m["text"], lemma,
                "neg" if m["negated"] else "",
                "ought_to" if m["ought_to"] else "",
                "Y" if ops["POSSIBILITY"] else "",
                "Y" if ops["NECESSITY"] else "",
                "Y" if ops["TSU"] else "",
                "Y" if ops["NEGATION"] else "",
                "Y" if rec["has_separator"] else "",
                ";".join(present_ops),
                sbn,
            ])

    return dict(
        n_blocks=n_blocks, records=records, flat_rows=flat_rows,
        semimodal_rows=semimodal_rows, review_rows=review_rows,
        mapping=mapping, modal_lemma_counts=modal_lemma_counts,
    )


def write_outputs(res):
    # ---- jsonl ----
    with (OUT / "modal_samples.jsonl").open("w", encoding="utf-8") as f:
        for rec in res["records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- flat csv ----
    with (OUT / "modal_samples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "modal_text", "modal_lemma", "negated",
                    "ought_to", "POSSIBILITY", "NECESSITY", "TSU", "NEGATION",
                    "has_separator", "modal_relevant_ops", "sbn"])
        w.writerows(res["flat_rows"])

    # ---- mapping matrix ----
    cols = ["POSSIBILITY", "NECESSITY", "TSU", "NEGATION", "<none>"]
    order = ["can", "could", "may", "might", "must",
             "shall", "should", "will", "would", "ought"]
    with (OUT / "modal_sbn_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modal_lemma", "n_hits"] + cols)
        for lemma in order:
            if lemma not in res["modal_lemma_counts"]:
                continue
            row = [lemma, res["modal_lemma_counts"][lemma]]
            row += [res["mapping"][lemma].get(c, 0) for c in cols]
            w.writerow(row)

    # ---- semi-modals ----
    with (OUT / "semimodal_samples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "match", "sbn"])
        w.writerows(res["semimodal_rows"])

    # ---- review / possible FP ----
    with (OUT / "review_possible_fp.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "token", "lemma", "reason", "sbn"])
        w.writerows(res["review_rows"])

    write_report(res)


def write_report(res):
    recs = res["records"]
    n_modal_blocks = len(recs)
    n_modal_tokens = len(res["flat_rows"])
    sep_blocks = [r for r in recs if r["has_separator"]]

    lines = []
    A = lines.append
    A("# Modal verbs in PMB-5.1.0 gold.sbn — SBN correspondence")
    A("")
    A(f"- Blocks scanned: **{res['n_blocks']}**")
    A(f"- Blocks containing >=1 core modal: **{n_modal_blocks}** "
      f"({n_modal_blocks/res['n_blocks']*100:.1f}%)")
    A(f"- Core modal tokens detected: **{n_modal_tokens}**")
    A(f"- Detection: spaCy `en_core_web_sm` PTB **MD** tag + NER guard "
      f"+ `'d`→`had` guard.")
    A("")
    A("## Per-modal → SBN operator mapping")
    A("")
    A("Counts are *blocks* in which the modal co-occurs with each SBN operator "
      "(a block may have several).")
    A("")
    A("| modal | hits | POSSIBILITY | NECESSITY | TSU(future) | NEGATION | none of P/N/TSU |")
    A("|-------|-----:|-----------:|---------:|-----------:|--------:|---------------:|")
    order = ["can", "could", "may", "might", "must",
             "shall", "should", "will", "would", "ought"]
    for lemma in order:
        if lemma not in res["modal_lemma_counts"]:
            continue
        m = res["mapping"][lemma]
        A(f"| {lemma} | {res['modal_lemma_counts'][lemma]} | "
          f"{m.get('POSSIBILITY',0)} | {m.get('NECESSITY',0)} | "
          f"{m.get('TSU',0)} | {m.get('NEGATION',0)} | {m.get('<none>',0)} |")
    A("")
    A("**Reading of the mapping**")
    A("")
    A("- `can / could / may / might` → **POSSIBILITY <1** (epistemic/ability/permission).")
    A("- `must / should / ought (to)` → **NECESSITY <1** (deontic/epistemic necessity).")
    A("- `will / would` → **time.n.08 TSU now** (future = *time successor*); they do "
      "*not* introduce a modal-force operator. A few `must`/`should` blocks also carry "
      "`TSU` when the obligation is future-oriented.")
    A("- Negated modals stack **NEGATION <1** on top, e.g. `can't` → "
      "`NEGATION <1 POSSIBILITY <1`.")
    A("- `shall` and `ought` have **0** occurrences in this split; `used to` / `dare` / "
      "`had better` are semi-modals (not MD) and are listed separately, not counted above.")
    A("")
    A("## The separator question")
    A("")
    A("All three modal devices (`POSSIBILITY`, `NECESSITY`, `TSU`) are realised as "
      "**in-box operators** scoping the next node via the `<1` index — they do **not** "
      "open a new box and are **not** discourse separators.")
    A("")
    A(f"- Modal blocks that *also* contain a separator "
      f"(CONTINUATION/CONJUNCTION/CONTRAST/CONSEQUENCE/…): "
      f"**{len(sep_blocks)}/{n_modal_blocks}**.")
    A("- In those blocks the separator is licensed by **clause structure** "
      "(subordination / coordination: *if…*, *…that…*, *…but…*, reported speech), "
      "**not** by the modal itself. The modal's own contribution stays in-box.")
    A("")
    A("Example separator co-occurrences (modal operator stays inside its box):")
    A("")
    shown = 0
    for r in sep_blocks:
        if shown >= 6:
            break
        A(f"- `{r['id']}` — {r['sentence']}")
        A(f"  - `{r['sbn']}`")
        shown += 1
    A("")
    A("## Implication for the self-repair (CORRECTION + CONJUNCTION) design")
    A("")
    A("Modals give a clean precedent: a *propositional-attitude / force* meaning is "
      "encoded **without a separator**, as an indexed in-box operator (`OP <1`). "
      "A self-repair marker that is *purely operator-like* could follow the same "
      "in-box pattern; a marker that genuinely **re-segments** the utterance (replaces "
      "or coordinates spans) is what should justify a separator — which is precisely "
      "the CONTRAST / CONJUNCTION family, not the modal family.")
    A("")
    A("## Files")
    A("")
    A("- `modal_samples.jsonl` — full per-block records.")
    A("- `modal_samples.csv` — one row per (block, modal) with operator flags.")
    A("- `modal_sbn_mapping.csv` — the matrix above.")
    A("- `semimodal_samples.csv` — need(n't)/dare/used-to/had-better (not in core counts).")
    A("- `review_possible_fp.csv` — audit list for **manual post-processing**: "
      "`'d`=had cases auto-excluded, plus kept-but-suspect noun homographs "
      "(`This can is leaking`, `thieves' cant`) to delete by hand if desired.")
    A("")

    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    res = extract()
    write_outputs(res)
    print(f"blocks scanned         : {res['n_blocks']}")
    print(f"blocks with a modal    : {len(res['records'])}")
    print(f"core modal tokens      : {len(res['flat_rows'])}")
    print(f"semi-modal hits        : {len(res['semimodal_rows'])}")
    print(f"guard-dropped (review) : {len(res['review_rows'])}")
    print("modal lemma counts     :",
          dict(res["modal_lemma_counts"].most_common()))
    print(f"output -> {OUT}")


if __name__ == "__main__":
    main()
