"""
Human audit of LLM-written repairs: two questions per item.

    python3 audit.py sample results/<model>.checked.tsv --out audit/<name>
    python3 audit.py serve audit/<name> [--annotator hz] [--port 8765]
    python3 audit.py stats audit/<name>

`sample` draws items stratified by repair type from the rows that pass every
automatic check (`auto_pass`), so the audit estimates the errors the checks
cannot see. Run it again with a larger `--per-type` (and `--types`) to top up
some types: items already drawn are kept, new ones go to the end.

`serve` opens a local page that shows one item at a time, without its type or
check results. Each save appends one line to `labels.jsonl`; a later save for
the same item replaces the earlier one when read.

`stats` joins labels to items and prints, per type and overall, the share of
"yes" among yes/no answers ("unsure" left out) with a 95% Wilson interval,
against the thresholds in README.md. It also writes `merged.tsv`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sbn_lin import SBNParseError, parse_sbn_line  # noqa: E402

csv.field_size_limit(sys.maxsize)

TYPES = ("content_word", "constant_name", "constant_number",
         "constant_temporal", "tense", "edge_label", "forwarding")
CHECKS = ("parse", "dag", "index", "magnitude", "edges", "boxes")
ITEM_FIELDS = ("audit_id", "request_id", "doc_id", "repair_type", "nl_clean", "nl_repair",
               "repair_onset", "reparandum_surface", "sbn_clean", "sbn_repair",
               "source")
QUESTIONS = {"meaning": 0.90, "natural": 0.80}   # README.md, round 2 thresholds
ANSWERS = ("yes", "no", "unsure")


def auto_pass(r: dict) -> bool:
    """Answered, and passes every check analyse_run.py writes.

    `oracle differs` is not counted as a failure: a choice other than
    build_repair's can still be right, and the audit is where that shows.
    """
    return (r.get("reply") == "ok" and r.get("status") == "ok"
            and bool(r.get("sbn_repair"))
            and all(r.get(k) == "ok" for k in CHECKS)
            and r.get("wn_ok") == "ok"
            and r.get("untouched") == "ok" and r.get("repair_part") == "ok"
            and r.get("withdraw") == "ok" and r.get("site_ok") == "ok"
            and r.get("place_inside") != "yes"
            and r.get("wn_relation") != "same_synset")


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_labels(d: Path) -> dict[str, dict]:
    """Last saved label per audit_id."""
    out: dict[str, dict] = {}
    p = d / "labels.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                out[rec["audit_id"]] = rec
    return out


def source_key(row: dict) -> str:
    """Stable generated-request key, with legacy fallback."""
    return row.get("request_id") or f"legacy::{row['repair_type']}::{row['doc_id']}"


# ---------------------------------------------------------------- sample

def cmd_sample(args: argparse.Namespace) -> None:
    rows = read_tsv(args.results)
    if "wn_ok" not in rows[0]:
        sys.exit(f"{args.results} has no check columns; run analyse_run.py --out first")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    items_path = out / "items.tsv"
    items = read_tsv(items_path) if items_path.exists() else []
    have = {source_key(i) for i in items}

    rng = random.Random(args.seed)
    types = args.types or TYPES
    new = []
    print(f"{'type':18s} {'answered':>9s} {'auto_pass':>9s} {'had':>5s} {'added':>5s}")
    for t in types:
        rs = [r for r in rows if r["repair_type"] == t]
        pool = sorted((r for r in rs if auto_pass(r)), key=lambda r: r["doc_id"])
        rng.shuffle(pool)
        had = sum(i["repair_type"] == t for i in items)
        fresh = [r for r in pool if source_key(r) not in have]
        take = fresh[:max(0, args.per_type - had)]
        new += take
        answered = sum(r.get("status") == "ok" for r in rs)
        print(f"{t:18s} {answered:9d} {len(pool):9d} {had:5d} {len(take):5d}"
              + ("   (pool exhausted)" if had + len(take) < args.per_type else ""))

    rng.shuffle(new)   # type order must not be guessable from position
    start = len(items)
    for k, r in enumerate(new):
        items.append({**{f: r.get(f, "") for f in ITEM_FIELDS},
                      "audit_id": f"a{start + k:04d}", "source": args.results.name})
    with open(items_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(items)
    print(f"\n{len(items)} items in {items_path}")


# ---------------------------------------------------------------- serve

def render_sbn(raw: str) -> dict:
    """Boxes with their concepts; index roles resolved to the target concept."""
    try:
        doc = parse_sbn_line(raw)
    except SBNParseError as e:
        return {"error": str(e), "raw": raw}
    cs = doc.concepts
    boxes = [{"id": 0, "op": None, "target": None, "concepts": []}]
    for s in doc.separators:
        boxes.append({"id": s.new_box, "op": s.name, "ptr": s.ptr,
                      "target": s.target_box, "concepts": []})
    for c in cs:
        roles = []
        for role, val in c.roles:
            ent = {"role": role, "val": val}
            if val[:1] in "+-" and val[1:].isdigit():
                j = c.pos + int(val)
                ent["to"] = j
                ent["to_name"] = cs[j].synset if 0 <= j < len(cs) else "OUT OF RANGE"
            roles.append(ent)
        boxes[c.box]["concepts"].append({"pos": c.pos, "synset": c.synset, "roles": roles})
    return {"boxes": boxes, "raw": raw}


def public_item(i: dict) -> dict:
    """What the page sees: no repair type, no check results."""
    return {"audit_id": i["audit_id"], "doc_id": i["doc_id"],
            "nl_clean": i["nl_clean"], "nl_repair": i["nl_repair"],
            "repair_onset": i["repair_onset"],
            "reparandum_surface": i["reparandum_surface"],
            "sbn_clean": render_sbn(i["sbn_clean"]),
            "sbn_repair": render_sbn(i["sbn_repair"])}


def cmd_serve(args: argparse.Namespace) -> None:
    d: Path = args.dir
    items = [public_item(i) for i in read_tsv(d / "items.tsv")]
    ids = {i["audit_id"] for i in items}
    page = (HERE / "audit_page.html").read_bytes()
    labels_path = d / "labels.jsonl"

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                labels = {k: v for k, v in read_labels(d).items()
                          if v.get("annotator") == args.annotator}
                self._json({"items": items, "labels": labels,
                            "annotator": args.annotator, "dir": str(d),
                            "questions": args.questions})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            rec = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if rec.get("audit_id") not in ids:
                return self._json({"error": "unknown audit_id"}, 400)
            for q in QUESTIONS:
                if rec.get(q) not in ANSWERS + ("",):
                    return self._json({"error": f"bad answer for {q}"}, 400)
            rec = {"audit_id": rec["audit_id"],
                   **{q: rec.get(q, "") for q in QUESTIONS},
                   "note": rec.get("note", "").strip(),
                   "annotator": args.annotator,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(labels_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._json(rec)

        def log_message(self, *a) -> None:  # keep the terminal quiet
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"{len(items)} items, labels -> {labels_path}")
    print(f"open http://127.0.0.1:{args.port}/   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- stats

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def cmd_stats(args: argparse.Namespace) -> None:
    d: Path = args.dir
    items = read_tsv(d / "items.tsv")
    labels = {k: v for k, v in read_labels(d).items()
              if args.annotator is None or v.get("annotator") == args.annotator}
    merged = [{**i, **{q: labels.get(i["audit_id"], {}).get(q, "") for q in QUESTIONS},
               "note": labels.get(i["audit_id"], {}).get("note", "")} for i in items]

    done = sum(bool(m["meaning"] and m["natural"]) for m in merged)
    print(f"{done}/{len(merged)} items have both answers; "
          + ", ".join(f"{q} {sum(bool(m[q]) for m in merged)}" for q in QUESTIONS) + "\n")
    for q, thr in QUESTIONS.items():
        print(f"{q}  (threshold {thr:.0%}; share of yes among yes+no)")
        print(f"  {'type':18s} {'yes':>4s} {'no':>4s} {'unsure':>6s} {'rate':>6s}  {'95% CI':>13s}")
        for t in TYPES + ("ALL",):
            ms = [m for m in merged if t == "ALL" or m["repair_type"] == t]
            if not ms:
                continue
            y = sum(m[q] == "yes" for m in ms)
            n = sum(m[q] == "no" for m in ms)
            u = sum(m[q] == "unsure" for m in ms)
            if y + n == 0:
                print(f"  {t:18s} {y:4d} {n:4d} {u:6d}      -")
                continue
            lo, hi = wilson(y, y + n)
            flag = ("below" if hi < thr else "under threshold, CI overlaps"
                    if y / (y + n) < thr else "")
            print(f"  {t:18s} {y:4d} {n:4d} {u:6d} {y / (y + n):6.0%}  [{lo:.2f}, {hi:.2f}]  {flag}")
        print()

    notes = [m for m in merged if m["note"]]
    if notes:
        print("notes:")
        for m in notes:
            print(f"  {m['doc_id']} [{m['repair_type']}] meaning={m['meaning'] or '-'} "
                  f"natural={m['natural'] or '-'}: {m['note']}")

    fields = list(ITEM_FIELDS) + list(QUESTIONS) + ["note"]
    with open(d / "merged.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(merged)
    print(f"\nwrote {d / 'merged.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("results", type=Path, help="a .checked.tsv from analyse_run.py")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--per-type", type=int, default=15)
    s.add_argument("--types", nargs="+", choices=TYPES)
    s.add_argument("--seed", type=int, default=13)
    s.set_defaults(fn=cmd_sample)

    v = sub.add_parser("serve")
    v.add_argument("dir", type=Path)
    v.add_argument("--annotator", default="hz")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--questions", nargs="+", choices=list(QUESTIONS), default=list(QUESTIONS),
                   help="questions shown on the page (default both); e.g. `natural` "
                        "for the edge-label selection, where the checks already cover meaning")
    v.set_defaults(fn=cmd_serve)

    t = sub.add_parser("stats")
    t.add_argument("dir", type=Path)
    t.add_argument("--annotator", help="only this annotator's labels (default: the latest label from anyone)")
    t.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
