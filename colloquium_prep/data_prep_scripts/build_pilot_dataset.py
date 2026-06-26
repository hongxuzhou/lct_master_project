"""
Build the static pilot dataset for HF upload (Shrikes/self_repair_parsing_pilot_data).

Pipeline (deterministic, no manual judgement except the pre-decided multi-sentence list):
  1. Read pilot_dataset_cleaned.tsv with the DEFAULT csv reader (field boundaries are
     correct this way: 11 cols / 865 rows).
  2. Strip the accumulated backslash-escaping artifacts. Diagnosed empirically:
     ALL 694 backslash chars sit in runs immediately before a double-quote, so the
     rule  \+(?=")  ->  ''  is provably safe (legitimate quotes are preserved).
  3. Regenerate ALL THREE interregnum variants from the CLEANED repair columns,
     reusing insert_interregnum() from interregnum_insertion.py. Only "I mean," is
     used (pilot study). NOTE: head/tail were NOT manually checked -- the comma
     heuristic may mis-place the editing phrase on some rows. Accepted as pilot risk.
  4. Drop the 29 multi-sentence / direct-speech rows (parsed from the Cat 1/2/3
     sections of multi_sentence_candidates.md; the "KEPT" section is never touched).
  5. Keep only the 9 input-side columns -- all parser-output *_sbn columns are dropped.
     The dataset is a STATIC input manifest; parser outputs are stored separately.
  6. Write pilot_dataset.parquet (binary columnar -> no escaping concept at all).

Run from anywhere; paths are absolute to the repo root.
"""

import re
import difflib
from pathlib import Path

import pandas as pd


def insert_interregnum(original, repaired, interregnum="I mean,"):
    """Insert the editing phrase between reparandum and repair.

    Extends the original interregnum_insertion.py logic: it handled only 'insert'
    opcodes (mid/tail-style reparanda that ADD tokens). Head-style reparanda that
    REPLACE the opening word (e.g. "My house" -> "Your house, my house") surface as
    a 'replace' opcode and were silently skipped (~23% of head rows collapsed onto
    the plain head repair). Here 'replace' chunks are handled identically: the
    disfluency comma inside the repaired chunk marks the reparandum boundary, so we
    insert "I mean," right after it. A comma is REQUIRED for the replace branch (no
    end-attachment fallback), so genuine word-substitutions without a comma are left
    untouched -- this only ADDS correct insertions, never invents boundaries.
    """
    if pd.isna(original) or pd.isna(repaired) or not str(repaired).strip():
        return ""

    tokens_orig = str(original).split()
    tokens_rep = str(repaired).split()
    matcher = difflib.SequenceMatcher(None, tokens_orig, tokens_rep)
    result_tokens = []

    def insert_after_comma(chunk, require_comma):
        comma_idx = next((i for i, t in enumerate(chunk) if ',' in t), -1)
        if comma_idx != -1:
            result_tokens.extend(chunk[:comma_idx + 1])
            result_tokens.append(interregnum)
            result_tokens.extend(chunk[comma_idx + 1:])
        elif not require_comma:
            result_tokens.extend(chunk)
            result_tokens.append(interregnum)
        else:
            result_tokens.extend(chunk)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result_tokens.extend(tokens_rep[j1:j2])
        elif tag == 'insert':
            insert_after_comma(tokens_rep[j1:j2], require_comma=False)
        elif tag == 'replace':
            insert_after_comma(tokens_rep[j1:j2], require_comma=True)
        elif tag == 'delete':
            pass

    return " ".join(result_tokens)

# Manually authored repairs that override defective Qwen augmentations. Each entry
# maps id -> {column: surface}. Only the plain repair cell is set; the matching
# *_interrug_nl is regenerated downstream by insert_interregnum() like every other
# row. All surfaces follow the corpus convention "<reparandum>, <gold-restatement>
# <rest>" so the retained (post-repair) reading equals the gold `mr`.
#
# Two defect classes, both from the original augmentation:
#   (a) head no-op: Qwen emitted repair_head_nl == nl (3 rows). An emptiness audit
#       misses these because the cell is non-empty.
#   (b) pure substitution: Qwen replaced a content word WITHOUT restating the gold
#       token (no reparandum kept), so the surface carried no self-repair and its
#       meaning diverged from gold (4 rows, e.g. all->none).
MANUAL_REPAIRS = {
    # (a) head no-op rows
    "p90/d2399": {"repair_head_nl": "I walked, ran away from home when I was thirteen."},
    "p67/d2005": {"repair_head_nl": "Wealth, money and I are strangers; in other words, I am poor."},
    "p16/d1602": {"repair_head_nl": "It took my ears, my eyes a moment to adjust to the darkness."},
    # (b) pure-substitution rows
    "p72/d2539": {"repair_head_nl": "There was, is a pond in the middle of the park."},
    "p36/d3158": {"repair_head_nl": "One, some of the apples in the box were rotten."},
    "p01/d1880": {"repair_mid_nl": "Two small squirrels, small rabbits, a white rabbit and a black rabbit, lived in a large forest."},
    "p41/d3041": {"repair_mid_nl": "Then Robert woke up again, and none, all of his limbs were hurting."},
}

REPO = Path("/Users/hongxuzhou/Documents/GitHub/lct_master_project")
SRC_TSV = REPO / "pilot_dataset_cleaned.tsv"
MULTI_MD = REPO / "colloquium_prep" / "multi_sentence_candidates.md"
OUT_PARQUET = REPO / "colloquium_prep" / "pilot_dataset.parquet"

FINAL_COLS = [
    "id", "nl", "mr",
    "repair_head_nl", "repair_mid_nl", "repair_tail_nl",
    "repair_head_interrug_nl", "repair_mid_interrug_nl", "repair_tail_interrug_nl",
]


def strip_escaping(s: str) -> str:
    """Remove backslash-runs that precede a double quote (the only artifact present)."""
    if not isinstance(s, str):
        return s
    return re.sub(r'\\+(?=")', '', s).strip()


def parse_multisentence_ids(md_path: Path) -> list[str]:
    """Extract the **pXX/dXXXX** ids that appear between the Cat 1 header and the
    'Deliberately KEPT' header -- i.e. the 29 rows to remove, never the KEPT ones."""
    text = md_path.read_text(encoding="utf-8")
    start = text.index("## Cat 1")
    end = text.index("## Deliberately KEPT")
    section = text[start:end]
    ids = re.findall(r"\*\*(p\d+/d\d+)\*\*", section)
    return ids


def main():
    print(f"Reading {SRC_TSV.name} (default reader, boundaries intact)...")
    df = pd.read_csv(SRC_TSV, sep="\t", dtype=str).fillna("")
    print(f"  loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # 1. strip escaping artifacts across every string cell
    df = df.map(strip_escaping)
    print("  backslash artifacts stripped")

    # 1b. inject manually authored repairs overriding defective augmentations.
    # Applied before interregnum regen so the editing-phrase variant is produced
    # by the same code path as every other row (no hand-written interregnum).
    n_cells = 0
    for rid, overrides in MANUAL_REPAIRS.items():
        sel = df["id"] == rid
        assert sel.sum() == 1, f"expected exactly 1 row for {rid}, got {sel.sum()}"
        for col, surface in overrides.items():
            current = df.loc[sel, col].iloc[0].strip()
            assert current != surface.strip(), (
                f"{rid}/{col}: already equals the manual surface -- upstream data "
                "changed, re-verify this override is still needed"
            )
            df.loc[sel, col] = surface
            n_cells += 1
    print(f"  injected {n_cells} manual repair overrides across {len(MANUAL_REPAIRS)} rows")

    # 2. regenerate all three interregnum variants from cleaned repair columns
    df["repair_head_interrug_nl"] = df.apply(
        lambda r: insert_interregnum(r["nl"], r["repair_head_nl"]), axis=1)
    df["repair_mid_interrug_nl"] = df.apply(
        lambda r: insert_interregnum(r["nl"], r["repair_mid_nl"]), axis=1)
    df["repair_tail_interrug_nl"] = df.apply(
        lambda r: insert_interregnum(r["nl"], r["repair_tail_nl"]), axis=1)
    print("  interregnum variants regenerated (head/mid/tail)")

    # 3. drop multi-sentence rows
    drop_ids = parse_multisentence_ids(MULTI_MD)
    assert len(drop_ids) == 29, f"expected 29 multi-sentence ids, got {len(drop_ids)}"
    missing = [i for i in drop_ids if i not in set(df["id"])]
    assert not missing, f"ids not found in TSV (format mismatch?): {missing}"
    before = len(df)
    df = df[~df["id"].isin(drop_ids)].reset_index(drop=True)
    print(f"  dropped {before - len(df)} multi-sentence rows -> {len(df)} rows")

    # 4. keep only the input-side columns
    df = df[FINAL_COLS]

    # 5. write parquet
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET}")
    print(f"  final shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")

    # self-check: no backslashes survive, no empty required fields
    bs = sum(df[c].str.contains("\\\\", regex=True).sum() for c in FINAL_COLS)
    empty_core = (df["nl"].str.len() == 0).sum() + (df["mr"].str.len() == 0).sum()
    print(f"  self-check: residual backslashes={bs}, empty nl/mr={empty_core}")


if __name__ == "__main__":
    main()