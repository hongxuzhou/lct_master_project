"""
prepare_pmb_data.py
────────────────────────────────────────────────────────────────────
Reads PMB 5.1.0 split/*.sbn files, validates every record, and
writes one JSON file per split.

Expected input directory layout (the `en/` folder you already have):
    split/en/
        train/gold.sbn
        train/silver.sbn
        train/bronze.sbn
        dev/standard.sbn
        test/standard.sbn
        test/long.sbn

Output JSON layout (one file per source split):
    processed/
        train_gold.json
        train_silver.json
        train_bronze.json
        dev_standard.json
        test_standard.json
        test_long.json

Each JSON file is a list of records:
    {
        "id":       "p50/d0779",
        "sentence": "The movie starts at ten o'clock.",
        "sbn":      "movie.n.01 start.v.01 Theme -1 ...",
        "messages": [
            {"role": "user",      "content": "<instruction>\n<sentence>"},
            {"role": "assistant", "content": "<sbn>"}
        ]
    }

The `messages` field uses the Gemma 2 Instruct convention and is
what TRL's SFTTrainer expects when you set
    tokenizer.apply_chat_template(..., tokenize=False)
The instruction string can be adjusted here before training if needed.

Usage:
    python prepare_pmb_data.py --pmb_en /path/to/split/en \
                               --out_dir ./processed
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter


# ── Instruction string ────────────────────────────────────────────
INSTRUCTION = (
    "Please parse the following sentence to discourse representation structure:"
)

# ── Pattern: a valid SBN synset node  e.g. movie.n.01, start.v.01 ─
SYNSET_RE = re.compile(r"\b[a-z_]+\.[nvasr]\.\d{2}\b")

# ── Pattern: doc_id  e.g. p50/d0779 ──────────────────────────────
DOC_ID_RE = re.compile(r"^p\d+/d\d+$")


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class Record:
    doc_id:   str
    sentence: str
    sbn:      str


@dataclass
class ValidationResult:
    ok:      list[Record] = field(default_factory=list)
    issues:  list[dict]   = field(default_factory=list)   # {"record": ..., "reasons": [...]}


# ─────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────

def parse_sbn_file(path: Path) -> list[Record]:
    """
    Read a .sbn file and return a list of (doc_id, sentence, sbn) records.

    The file format is strict 3-line groups:
        p50/d0779
        The movie starts at ten o'clock.
        movie.n.01 start.v.01 Theme -1 ...
        p36/d2375
        ...

    Raises ValueError if the total line count is not a multiple of 3
    (after stripping trailing blank lines).
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    # Drop trailing empty lines (common in Unix text files)
    while raw_lines and raw_lines[-1].strip() == "":
        raw_lines.pop()

    if len(raw_lines) % 3 != 0:
        raise ValueError(
            f"{path}: line count {len(raw_lines)} is not a multiple of 3. "
            "Check for unexpected blank lines or truncated records."
        )

    records = []
    for i in range(0, len(raw_lines), 3):
        doc_id   = raw_lines[i].strip()
        sentence = raw_lines[i + 1].strip()
        sbn      = raw_lines[i + 2].strip()
        records.append(Record(doc_id=doc_id, sentence=sentence, sbn=sbn))

    return records


# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_records(records: list[Record], source_label: str) -> ValidationResult:
    """
    Check every record against basic sanity criteria and return a
    ValidationResult separating clean records from problematic ones.

    Checks performed:
        1. doc_id matches pNN/dNNNN pattern
        2. sentence is non-empty
        3. SBN is non-empty
        4. SBN contains at least one synset node (word.pos.NN)
        5. No stray Unicode control characters in sentence or SBN
    """
    result = ValidationResult()
    control_char_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    for rec in records:
        reasons = []

        if not DOC_ID_RE.match(rec.doc_id):
            reasons.append(f"doc_id '{rec.doc_id}' does not match pNN/dNNNN")

        if not rec.sentence:
            reasons.append("sentence is empty")

        if not rec.sbn:
            reasons.append("SBN is empty")
        else:
            if not SYNSET_RE.search(rec.sbn):
                reasons.append("SBN contains no synset node (word.pos.NN)")
            if control_char_re.search(rec.sbn):
                reasons.append("SBN contains control characters")

        if rec.sentence and control_char_re.search(rec.sentence):
            reasons.append("sentence contains control characters")

        if reasons:
            result.issues.append({"record": rec, "reasons": reasons})
        else:
            result.ok.append(rec)

    return result


# ─────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────

def print_stats(label: str, records: list[Record]) -> None:
    """Print length distribution statistics for a list of records."""
    if not records:
        print(f"  [!] {label}: no records to report")
        return

    sent_lens = [len(rec.sentence.split()) for rec in records]
    sbn_lens  = [len(rec.sbn.split())      for rec in records]

    def quartiles(vals):
        vals = sorted(vals)
        n = len(vals)
        return {
            "min":    vals[0],
            "q25":    vals[n // 4],
            "median": vals[n // 2],
            "q75":    vals[3 * n // 4],
            "max":    vals[-1],
            "mean":   round(sum(vals) / n, 1),
        }

    sq = quartiles(sent_lens)
    bq = quartiles(sbn_lens)

    print(f"\n  ── {label} ({len(records):,} records) ──")
    print(f"  Sentence length (words):  "
          f"min={sq['min']}  q25={sq['q25']}  median={sq['median']}  "
          f"q75={sq['q75']}  max={sq['max']}  mean={sq['mean']}")
    print(f"  SBN length    (tokens):   "
          f"min={bq['min']}  q25={bq['q25']}  median={bq['median']}  "
          f"q75={bq['q75']}  max={bq['max']}  mean={bq['mean']}")

    # Count synsets per record
    synset_counts = [len(SYNSET_RE.findall(rec.sbn)) for rec in records]
    sq2 = quartiles(synset_counts)
    print(f"  Synset nodes per record:  "
          f"min={sq2['min']}  median={sq2['median']}  max={sq2['max']}  mean={sq2['mean']}")


def print_duplicate_check(records: list[Record], label: str) -> None:
    """Warn if any sentence appears more than once in the same split."""
    counter = Counter(rec.sentence for rec in records)
    dupes = {s: c for s, c in counter.items() if c > 1}
    if dupes:
        print(f"  [!] {label}: {len(dupes)} duplicate sentence(s) detected "
              f"(e.g. '{next(iter(dupes))[:60]}...'  ×{next(iter(dupes.values()))})")
    else:
        print(f"  [✓] {label}: no duplicate sentences")


# ─────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────

def record_to_dict(rec: Record) -> dict:
    """
    Convert a Record to the JSON dict that the LoRA training script
    will consume.  The `messages` field is what TRL / HuggingFace
    apply_chat_template expects.
    """
    user_content = f"{INSTRUCTION}\n{rec.sentence}"
    return {
        "id":       rec.doc_id,
        "sentence": rec.sentence,
        "sbn":      rec.sbn,
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": rec.sbn},
        ],
    }


def save_json(records: list[Record], out_path: Path) -> None:
    data = [record_to_dict(r) for r in records]
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  → saved {len(data):,} records to {out_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

# Files to process: (relative path under pmb_en, output stem)
SPLITS = [
    ("train/gold.sbn",    "train_gold"),
    ("train/silver.sbn",  "train_silver"),
    ("train/bronze.sbn",  "train_bronze"),
    ("dev/standard.sbn",  "dev_standard"),
    ("test/standard.sbn", "test_standard"),
    ("test/long.sbn",     "test_long"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Process and validate PMB 5.1.0 SBN split files."
    )
    parser.add_argument(
        "--pmb_en",
        type=Path,
        required=True,
        help="Path to the split/en directory (contains train/, dev/, test/).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("processed"),
        help="Output directory for JSON files (default: ./processed).",
    )
    parser.add_argument(
        "--skip_silver_bronze",
        action="store_true",
        help="Only process gold train, dev, and test; skip silver and bronze.",
    )
    args = parser.parse_args()

    if not args.pmb_en.is_dir():
        sys.exit(f"[ERROR] --pmb_en path not found: {args.pmb_en}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PMB 5.1.0 Data Processing & Validation")
    print(f"  Source : {args.pmb_en}")
    print(f"  Output : {args.out_dir}")
    print("=" * 60)

    total_ok = 0
    total_issues = 0

    for rel_path, stem in SPLITS:
        if args.skip_silver_bronze and stem in ("train_silver", "train_bronze"):
            print(f"\n[SKIP] {rel_path}")
            continue

        src = args.pmb_en / rel_path
        if not src.exists():
            print(f"\n[MISSING] {rel_path} – skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"[{stem}]  {src}")

        # 1. Parse
        try:
            records = parse_sbn_file(src)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            total_issues += 1
            continue

        print(f"  Parsed {len(records):,} raw records")

        # 2. Validate
        vr = validate_records(records, stem)
        n_ok     = len(vr.ok)
        n_issues = len(vr.issues)
        total_ok     += n_ok
        total_issues += n_issues

        if n_issues == 0:
            print(f"  [✓] All records passed validation")
        else:
            print(f"  [!] {n_issues} record(s) failed validation:")
            for item in vr.issues[:10]:   # show first 10 only
                rec = item["record"]
                print(f"      {rec.doc_id}: {item['reasons']}")
            if n_issues > 10:
                print(f"      ... and {n_issues - 10} more")

        # 3. Statistics (on clean records only)
        print_stats(stem, vr.ok)
        print_duplicate_check(vr.ok, stem)

        # 4. Save clean records
        out_path = args.out_dir / f"{stem}.json"
        save_json(vr.ok, out_path)

    print(f"\n{'='*60}")
    print(f"DONE.  Total clean records: {total_ok:,}  |  "
          f"Total flagged: {total_issues:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()