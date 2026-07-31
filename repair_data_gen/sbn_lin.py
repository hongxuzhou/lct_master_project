"""
Linear-order SBN parser + repair-insertion transform.

Deliberately *not* the PMB graph parser: for repair synthesis we need the raw
linear concept order (indices are offsets over that order, see C1 of the
notation doc), which the networkx graph throws away.

Everything here works on the one-line SBN format used in
data/pmb-5.1.0/split/en/*/*.sbn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

SEPARATORS = {
    "ALTERNATION", "ATTRIBUTION", "CONDITION", "CONSEQUENCE", "CONTINUATION",
    "CONTRAST", "EXPLANATION", "NECESSITY", "NEGATION", "POSSIBILITY",
    "PRECONDITION", "RESULT", "SOURCE", "CONJUNCTION", "ELABORATION",
    "CORRECTION",
}

DRS_OPERATORS = {
    "TSU", "MOR", "BOT", "TOP", "ESU", "EPR", "EQU", "NEQ", "APX", "LES",
    "LEQ", "TPR", "TAB", "TIN", "SZP", "SZN", "SXP", "SXN", "STI", "STO",
    "SY1", "SY2", "SXY", "ANA", "TCT",
}

ROLES = {
    "InstanceOf", "Proposition", "Name", "Agent", "AgentOf", "Asset",
    "Attribute", "AttributeOf", "Affectee", "Beneficiary", "Causer",
    "Co-Agent", "Co-Patient", "Co-Theme", "Co-ThemeOf", "Consumer",
    "Destination", "Duration", "Experiencer", "Finish", "Frequency", "Goal",
    "Instrument", "Instance", "Location", "Manner", "MannerOf", "Material",
    "Path", "Patient", "Pivot", "Product", "Recipient", "Result", "Source",
    "Start", "Stimulus", "Theme", "ThemeOf", "Time", "TimeOf", "Topic",
    "Value", "Bearer", "Colour", "ColourOf", "ContentOf", "Content", "Creator",
    "Degree", "MadeOf", "Of", "Operand", "Owner", "Part", "PartOf", "Player",
    "Quantity", "Role", "Sub", "SubOf", "Title", "Unit", "User", "ClockTime",
    "DayOfMonth", "DayOfWeek", "Decade", "MonthOfYear", "YearOfCentury",
    "Affector", "Context", "Equal", "Extent", "Precondition", "Measure",
    "Cause", "Order", "Participant", "FeatureOf",
    # Attested in gold but missing from SBNSpec.ROLES -> the stock parser
    # aborts on p36/d2375 and p30/d2358. Added here so parsing is lossless.
    "CauserOf",
}

# Roles whose inverse to_penman_string() will normalise back to the base role.
# NB: this is the *extended* set (project additions), cf. C6 of the notation doc.
INVERTIBLE_ROLES = {
    "InstanceOf", "AttributeOf", "ColourOf", "ContentOf", "PartOf", "SubOf",
    "ThemeOf", "TimeOf", "AgentOf", "MannerOf", "Co-ThemeOf",
}
# Base role -> inverted form, restricted to inverses the scorer understands.
INVERSE_OF = {r[: -len("Of")]: r for r in INVERTIBLE_ROLES if r.endswith("Of")}
INVERSE_OF["Instance"] = "InstanceOf"

SYNSET_PATTERN = re.compile(r"^(.+)\.(n|v|a|r|x)\.(\d+)$")
IDX_PATTERN = re.compile(r"^([-+])(\d+)$")
BOX_PTR_PATTERN = re.compile(r"^([<>])(\d+)$")


@dataclass
class Concept:
    pos: int                 # index in the linear concept sequence
    synset: str
    box: int                 # id of the box this concept lives in
    roles: list[tuple[str, str]] = field(default_factory=list)

    @property
    def lemma(self) -> str:
        m = SYNSET_PATTERN.match(self.synset)
        return m.group(1) if m else self.synset

    @property
    def pos_tag(self) -> str:
        m = SYNSET_PATTERN.match(self.synset)
        return m.group(2) if m else "?"


@dataclass
class Separator:
    after_concept: int       # number of concepts emitted before this separator
    name: str
    ptr: str                 # e.g. "<1"
    new_box: int
    target_box: int | None


@dataclass
class SBNDoc:
    doc_id: str
    sentence: str
    raw: str
    concepts: list[Concept]
    separators: list[Separator]

    def edges(self) -> Iterator[tuple[int, str, int]]:
        """(source concept pos, role, target concept pos) for index-valued roles."""
        for c in self.concepts:
            for role, val in c.roles:
                m = IDX_PATTERN.match(val)
                if not m:
                    continue
                off = int(val)
                tgt = c.pos + off
                if 0 <= tgt < len(self.concepts):
                    yield (c.pos, role, tgt)

    def seps_after(self, concept_pos: int) -> list[Separator]:
        return [s for s in self.separators if s.after_concept > concept_pos]


class SBNParseError(Exception):
    pass


def parse_sbn_line(raw: str, doc_id: str = "", sentence: str = "") -> SBNDoc:
    tokens = raw.split()
    concepts: list[Concept] = []
    separators: list[Separator] = []
    cur_box = 0
    n_boxes = 1
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in SEPARATORS:
            if i + 1 >= len(tokens):
                raise SBNParseError(f"{doc_id}: separator {t} without pointer")
            ptr = tokens[i + 1]
            m = BOX_PTR_PATTERN.match(ptr)
            if not m:
                raise SBNParseError(f"{doc_id}: bad box pointer {ptr!r} after {t}")
            new_box = n_boxes
            n_boxes += 1
            k = int(m.group(2))
            # mirrors sbn_smatch.py: <k on the newly created box targets box new_box-k
            target = None if (m.group(1) == "<" and k == 0) else new_box - k
            separators.append(Separator(len(concepts), t, ptr, new_box, target))
            cur_box = new_box
            i += 2
            continue
        if SYNSET_PATTERN.match(t):
            concepts.append(Concept(len(concepts), t, cur_box))
            i += 1
            continue
        if t in ROLES or t in DRS_OPERATORS:
            if i + 1 >= len(tokens):
                raise SBNParseError(f"{doc_id}: role {t} without value")
            if not concepts:
                raise SBNParseError(f"{doc_id}: role {t} before any concept")
            val = tokens[i + 1]
            if val.startswith('"') and not val.endswith('"'):
                j = i + 1
                while j + 1 < len(tokens) and not tokens[j].endswith('"'):
                    j += 1
                val = " ".join(tokens[i + 1 : j + 1])
                i = j + 1
            else:
                i += 2
            concepts[-1].roles.append((t, val))
            continue
        raise SBNParseError(f"{doc_id}: unparsable token {t!r}")
    return SBNDoc(doc_id, sentence, raw, concepts, separators)


def read_split(path: str | Path) -> list[SBNDoc]:
    """Read a PMB split .sbn file (id / sentence / sbn / blank blocks)."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    docs, skipped = [], []
    i = 0
    id_re = re.compile(r"^p\d{2}/d\d{4}$")
    while i + 2 < len(lines):
        if id_re.match(lines[i].strip()):
            doc_id, sent, sbn = (lines[i].strip(), lines[i + 1].strip(),
                                 lines[i + 2].strip())
            try:
                docs.append(parse_sbn_line(sbn, doc_id, sent))
            except SBNParseError as e:
                skipped.append((doc_id, str(e)))
            i += 3
        else:
            i += 1
    read_split.skipped = skipped  # type: ignore[attr-defined]
    return docs


def serialise(concepts: list[Concept], separators: list[Separator]) -> str:
    """Render concepts+separators back to a one-line SBN string."""
    out: list[str] = []
    sep_by_pos: dict[int, list[Separator]] = {}
    for s in separators:
        sep_by_pos.setdefault(s.after_concept, []).append(s)
    for k in range(len(concepts) + 1):
        for s in sep_by_pos.get(k, []):
            out += [s.name, s.ptr]
        if k < len(concepts):
            c = concepts[k]
            out.append(c.synset)
            for role, val in c.roles:
                out += [role, val]
    return " ".join(out)
