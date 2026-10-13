"""
Does prompts/base/sbn_manual.md document every kind of token that PMB train
gold uses? The manual's inventories were built from gold by kind (roles,
operators, separators, constants by class), so a kind nobody thought to
count can be missing without notice: `?` was, until the Muse Glimmer smoke
run showed a model guessing at it.

    python3 check_manual_coverage.py [--min N]

Every gold token is classified as concept, index, connector, separator,
operator, role or constant. Roles, operators and separators count as
documented when the manual writes them in backticks (an inverse `ROf` also
when `R` is documented, as §2.3 states the rule). Constants count as
documented when they match a class in the manual's §2.2 table. Everything
else is printed with its gold frequency and one example sentence.
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLD = HERE.parents[1] / "data/pmb-5.1.0/split/en/train/gold.sbn"
MANUAL = HERE / "prompts/base/sbn_manual.md"

CONCEPT = re.compile(r"^\S+\.[nvar]\.\d\d$")
INDEX = re.compile(r"^[+-]\d+$")
CONNECTOR = re.compile(r"^[<>]\d+$")
SEPARATOR = re.compile(r"^[A-Z]{4,}$")
OPERATOR = re.compile(r"^[A-Z]{2}[A-Z0-9]$")   # SY1, SY2 have a digit
TOKEN = re.compile(r'"[^"]*"|\S+')              # names may contain spaces
ROLE = re.compile(r"^[A-Z][a-z]+(-[A-Z][a-z]+)*(Of)?$|^[A-Z][a-z]*[A-Z][A-Za-z]*$")

# Constant classes as §2.2 of the manual describes them. Each entry is
# (class name in the report, pattern, the §2.2 row it corresponds to).
CONSTANT_CLASSES = [
    ("deictic", re.compile(r"^(now|speaker|hearer)$")),
    ("name string", re.compile(r'^".*"$')),
    ("number", re.compile(r"^\d+(\.\d+)?$")),
    ("plus/minus", re.compile(r"^[+-]$")),
    ("clock time", re.compile(r"^\d{1,2}:\d\d$")),
    ("weekday", re.compile(r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$")),
    ("single capital letter", re.compile(r"^[A-Z]$")),
    ("decade", re.compile(r"^'\d{3}X'$")),
    ("question", re.compile(r"^\?$")),
    ("score", re.compile(r"^\d+-\d+$")),
    ("fraction", re.compile(r"^\d+/\d+$")),
]


def documented_names(manual: str) -> set[str]:
    return set(re.findall(r"`([A-Za-z][A-Za-z0-9-]*)`", manual))


def constant_class(tok: str, manual: str) -> str | None:
    for name, pat in CONSTANT_CLASSES:
        if pat.match(tok):
            # a class counts only if the manual's constants section mentions it
            return name if f"| {name} |" in manual else None
    return None


def read_gold():
    for block in GOLD.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.split("\n")
        if len(lines) >= 3:
            yield lines[0], lines[1].strip(), TOKEN.findall(lines[2].split("%")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=1, help="hide items rarer than this")
    args = ap.parse_args()
    manual = MANUAL.read_text(encoding="utf-8")
    names = documented_names(manual)

    missing = collections.Counter()
    example: dict[tuple[str, str], str] = {}
    for doc_id, nl, toks in read_gold():
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if CONCEPT.match(t) or INDEX.match(t) or CONNECTOR.match(t):
                continue
            if SEPARATOR.match(t) and CONNECTOR.match(nxt):
                kind, ok = "separator", t in names
            elif OPERATOR.match(t):
                kind, ok = "operator", t in names
            elif ROLE.match(t):
                base = t[:-2] if t.endswith("Of") and t not in names else t
                kind, ok = "role", t in names or base in names
            else:
                kind, ok = "constant", constant_class(t, manual) is not None
            if not ok:
                key = (kind, t if kind != "constant" else _shape(t))
                missing[key] += 1
                example.setdefault(key, f"{doc_id}  {nl}")

    shown = [(k, n) for k, n in missing.most_common() if n >= args.min]
    if not shown:
        print("every gold token kind is documented in sbn_manual.md")
        return
    print(f"{'kind':10s} {'token / shape':24s} {'gold':>5s}  example")
    for (kind, tok), n in shown:
        print(f"{kind:10s} {tok:24s} {n:5d}  {example[(kind, tok)]}")


def _shape(tok: str) -> str:
    """Group constants by form, so 300 different years show as one line."""
    if re.match(r"^'\d{4}'$", tok):
        return "'NNNN' (quoted year)"
    if re.match(r"^-?\d+(\.\d+)?$", tok):
        return "number"
    return tok


if __name__ == "__main__":
    main()
