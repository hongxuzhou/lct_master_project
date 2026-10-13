"""
Round-trip repaired SBN through the project's own SBN reader
(data/pmb-5.1.0/src/sbn/sbn_smatch.py), which is what the evaluation pipeline
uses.  A splice that our arithmetic calls "legal" is worthless if SBNGraph
refuses it, silently loses an edge, or produces a non-DAG.

Checks per SBN string, in `check()`:
  parse      -- SBNGraph.from_string does not raise
  dag        -- graph is acyclic
  index      -- every index resolves to a concept.  The reader does not drop
                an out-of-range index: it turns `Theme +3` into an edge to a
                constant node "+3", so this has to be looked for explicitly
  magnitude  -- no index of 10 or more; SBNSpec.INDEX_PATTERN reads one digit
                only, so `-12` is silently read as `-1`
  edges      -- distinct (source, target) edges written == edges in the graph
                (two roles between the same pair collapse into one edge in a
                DiGraph; that happens in gold too, e.g. p12/d3490, so pairs,
                not roles, are counted)
  boxes      -- CORRECTION and CONJUNCTION end up as *siblings* off the same
                parent box, which is what the notation requires

Run:
  python3 validate_with_pmb_parser.py [n_samples]
      generate repairs with build_repair and check them (the original use)
  python3 validate_with_pmb_parser.py --tsv FILE [--column sbn_repair] [--out FILE]
      check every row of a TSV, e.g. the LLM playground's output or
      generate_repairs.py's; --out writes one result row per input row
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/pmb-5.1.0/src/sbn"))

import networkx as nx  # noqa: E402

from sbn_smatch import SBNGraph, SBNSource  # noqa: E402
from sbn_spec import (  # noqa: E402
    SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNError, split_comments, split_single,
)

from sbn_lin import BOX_PTR_PATTERN, IDX_PATTERN, parse_sbn_line, read_split  # noqa: E402

GOLD = ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"
SIGNED_DIGITS = re.compile(r"^[+-]\d+$")
MAX_INDEX_MAGNITUDE = 9

# Result keys, in report order.  A key counts the rows where that outcome
# happened; the *_ok keys count rows that passed the check.
KEYS = ("parsed", "parse_error", "crash", "not_dag", "unresolved_index",
        "index_overflow", "edge_loss", "edges_ok", "missing_box_edge",
        "not_siblings", "boxes_ok")


def strip_comments(sbn: str) -> str:
    """The SBN without `%` comments, cut the way the reader cuts them."""
    return " ".join(line for line, _ in split_comments(split_single(sbn)))


def expected_edges(sbn: str) -> tuple[int, int]:
    """(distinct role/operator edges written, largest |index|)."""
    doc = parse_sbn_line(strip_comments(sbn))
    pairs: set[tuple[int, int]] = set()
    other = 0
    max_abs = 0
    for c in doc.concepts:
        for _, val in c.roles:
            if BOX_PTR_PATTERN.match(val):
                continue                      # syn-box edge, not counted here
            if IDX_PATTERN.match(val):
                max_abs = max(max_abs, abs(int(val)))
                tgt = c.pos + int(val)
                if 0 <= tgt < len(doc.concepts):
                    pairs.add((c.pos, tgt))
                    continue
            other += 1                        # constant, or out-of-range index
    return len(pairs) + other, max_abs


def box_shape(G) -> tuple[tuple, ...]:
    """Box-box connector labels with their parent and child box ids."""
    out = []
    for u, v, data in G.edges(data=True):
        if data.get("type") == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            out.append((data["token"], u[1], v[1]))
    return tuple(sorted(out, key=lambda x: x[2]))


def check(sbn: str) -> dict:
    """Run every check on one SBN string.

    Returns a dict with one key per check (`parse`, `dag`, `index`,
    `magnitude`, `edges`, `boxes`), each "ok" or the name of the failure,
    plus `detail` for the first failure found.  Checks after a failed parse
    are None.
    """
    r = {"parse": None, "dag": None, "index": None, "magnitude": None,
         "edges": None, "boxes": None, "detail": ""}
    try:
        G = SBNGraph(source=SBNSource.INFERENCE).from_string(sbn, is_single_line=True)
    except SBNError as e:
        r["parse"], r["detail"] = "parse_error", str(e)
        return r
    except Exception as e:  # noqa: BLE001
        r["parse"], r["detail"] = "crash", f"{type(e).__name__}: {e}"
        return r
    r["parse"] = "ok"
    details = []

    r["dag"] = "ok" if nx.is_directed_acyclic_graph(G) else "not_dag"
    if r["dag"] != "ok":
        details.append("cycle")

    role_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                  if d.get("type") in (SBN_EDGE_TYPE.ROLE, SBN_EDGE_TYPE.DRS_OPERATOR)]
    dangling = [f"{G.nodes[u]['token']} {d['token']} {G.nodes[v]['token']}"
                for u, v, d in role_edges
                if G.nodes[v].get("type") == SBN_NODE_TYPE.CONSTANT
                and SIGNED_DIGITS.match(str(G.nodes[v].get("token", "")))]
    r["index"] = "ok" if not dangling else "unresolved_index"
    if dangling:
        details.append("index points past the sequence: " + "; ".join(dangling))

    try:
        want, max_abs = expected_edges(sbn)
    except Exception as e:  # noqa: BLE001 -- sbn_lin is stricter than the reader
        want, max_abs = None, 0
        details.append(f"could not count written edges: {e}")
    r["magnitude"] = "ok" if max_abs <= MAX_INDEX_MAGNITUDE else "index_overflow"
    if r["magnitude"] != "ok":
        details.append(f"index of magnitude {max_abs}")
    if want is not None:
        got = len(role_edges)
        r["edges"] = "ok" if got == want else "edge_loss"
        if got != want:
            details.append(f"wrote {want} edges, graph has {got}")

    shape = box_shape(G)
    corr = next(((par, new) for tok, par, new in shape if tok == "CORRECTION"), None)
    # our CONJUNCTION is the box created immediately after the CORRECTION
    conj = next(((par, new) for tok, par, new in shape
                 if tok == "CONJUNCTION" and corr and new == corr[1] + 1), None)
    if corr is None or conj is None:
        r["boxes"] = "missing_box_edge"
        details.append(f"CORRECTION={corr} CONJUNCTION={conj}")
    elif corr[0] != conj[0]:
        r["boxes"] = "not_siblings"
        details.append(f"CORRECTION parent box {corr[0]}, CONJUNCTION parent {conj[0]}")
    else:
        r["boxes"] = "ok"

    r["detail"] = " | ".join(details)
    return r


def tally(results: list[dict]) -> collections.Counter:
    """Count outcomes over many check() results, under the KEYS names."""
    c = collections.Counter()
    for r in results:
        if r["parse"] != "ok":
            c[r["parse"]] += 1
            continue
        c["parsed"] += 1
        if r["dag"] != "ok":
            c["not_dag"] += 1
        if r["index"] != "ok":
            c["unresolved_index"] += 1
        if r["magnitude"] != "ok":
            c["index_overflow"] += 1
        if r["edges"] is not None:
            c["edges_ok" if r["edges"] == "ok" else "edge_loss"] += 1
        c["boxes_ok" if r["boxes"] == "ok" else r["boxes"]] += 1
    return c


def report(results: list[tuple[str, str, dict]]) -> None:
    """Print the tally and up to 5 examples per failure."""
    counts = tally([r for _, _, r in results])
    print("=== results ===")
    for k in KEYS:
        if counts[k]:
            print(f"  {k:20s} {counts[k]:6d}")
    print()
    failures = collections.defaultdict(list)
    for row_id, sbn, r in results:
        for key in ("parse", "dag", "index", "magnitude", "edges", "boxes"):
            if r[key] not in (None, "ok"):
                failures[r[key]].append((row_id, r["detail"], sbn))
    for k, v in failures.items():
        print(f"=== {k} (showing up to 5 of {len(v)}) ===")
        for row_id, msg, line in v[:5]:
            print(f"  {row_id}: {msg}")
            print(f"    {line}")
        print()


def generated_samples(n_samples: int) -> list[tuple[str, str]]:
    """(doc_id, repaired SBN) from build_repair, as the script always did.

    Needs analyse_feasibility and wn_candidates, which moved to
    archive/repair_data_gen/rule_pipeline/ on 2026-10-13 (add it to sys.path).
    """
    from analyse_feasibility import NON_LEXICAL, POS_OK, WORD_RE, surface_forms, token_match
    from repair_transform import build_repair
    from wn_candidates import candidates

    random.seed(0)
    out = []
    for doc in read_split(GOLD):
        lowered = [t.lower() for t in WORD_RE.findall(doc.sentence)]
        for c in doc.concepts:
            if c.pos_tag not in POS_OK or c.synset in NON_LEXICAL:
                continue
            if token_match(surface_forms(c.synset), lowered) is None:
                continue
            cands = candidates(c.synset)
            if not cands:
                continue
            res = build_repair(doc, c.pos, cands[0])
            if res.ok:
                out.append((doc.doc_id, res.sbn))
    random.shuffle(out)
    return out[:n_samples]


def check_tsv(path: Path, column: str, out: Path | None) -> None:
    """Check `column` of every row in a TSV; optionally write results."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if rows and column not in rows[0]:
        sys.exit(f"{path}: no column {column!r}; columns are {list(rows[0])}")
    results, out_rows = [], []
    skipped = 0
    for k, row in enumerate(rows):
        sbn = (row.get(column) or "").strip()
        row_id = row.get("doc_id") or str(k)
        if not sbn:                            # e.g. a no_site answer
            skipped += 1
            continue
        r = check(sbn)
        results.append((row_id, sbn, r))
        out_rows.append({"row": k, "doc_id": row.get("doc_id", ""),
                         **{key: ("" if v is None else v) for key, v in r.items()}})
    print(f"checking {len(results)} rows of {path} (column {column!r}); "
          f"{skipped} rows with an empty SBN skipped\n")
    report(results)
    if out:
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n_samples", nargs="?", type=int, default=2000,
                    help="how many build_repair samples to check (without --tsv)")
    ap.add_argument("--tsv", type=Path, help="check the SBN column of this TSV instead")
    ap.add_argument("--column", default="sbn_repair")
    ap.add_argument("--out", type=Path, help="with --tsv: write per-row results here")
    args = ap.parse_args()

    if args.tsv:
        check_tsv(args.tsv, args.column, args.out)
        return
    samples = generated_samples(args.n_samples)
    print(f"validating {len(samples)} generated repairs against sbn_smatch.SBNGraph\n")
    report([(doc_id, sbn, check(sbn)) for doc_id, sbn in samples])


if __name__ == "__main__":
    main()
