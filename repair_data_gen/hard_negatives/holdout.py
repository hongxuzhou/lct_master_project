"""The train documents held out for hard negatives, and the one way to read them.

    python3 holdout.py build    # train_comma_labels.tsv -> train_holdout.tsv
    python3 holdout.py check    # list matches its recorded sha256

Every script that builds training data from PMB train gold calls `load()`
and leaves these documents out: the RSR source pool (`select_docs.py`), the
RSR set (`assemble.py`) and the fluent PMB part of every condition.

`load()` stops if the list differs from `SHA256`, so the list cannot change
without this file changing too.
"""
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = HERE / "train_comma_labels.tsv"
HOLDOUT = HERE / "train_holdout.tsv"
# Labels held out of training. contrast_correction and borderline are held
# out too: whether they enter the test set is decided later, but once trained
# on they cannot be used (decided 2026-10-16).
HELD = ("apposition", "asyndetic", "list", "contrast_correction", "borderline")
SHA256 = "9a29eab567953f7e1df42194328e8ebbf58690314c2f118dd3612546bce91f7d"


def build() -> None:
    with open(LABELS, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["label"] in HELD]
    with open(HOLDOUT, "w", encoding="utf-8") as f:
        f.write("doc_id\tlabel\tsentence\n")
        for r in sorted(rows, key=lambda r: r["doc_id"]):
            f.write(f"{r['doc_id']}\t{r['label']}\t{r['sentence']}\n")
    print(f"{len(rows)} documents -> {HOLDOUT.name}, sha256 {sha()}")


def sha() -> str:
    return hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()


def load() -> frozenset[str]:
    """Doc ids held out of every training set built from PMB train gold."""
    if sha() != SHA256:
        sys.exit(f"{HOLDOUT} does not match its recorded sha256; "
                 "rebuild with holdout.py build and update SHA256 on purpose")
    with open(HOLDOUT, encoding="utf-8") as f:
        return frozenset(r["doc_id"] for r in csv.DictReader(f, delimiter="\t"))


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    elif sys.argv[1:] == ["check"]:
        print(f"{len(load())} documents, sha256 ok")
    else:
        sys.exit(__doc__)
