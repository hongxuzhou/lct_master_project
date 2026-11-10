"""Build the confirmed PMB test bases and the pending marked-pair review."""
from __future__ import annotations

from collections import Counter
import csv
import importlib.util
import io
import json
from pathlib import Path
import random

from .analyse_split import ROOT, GEN, read, sha, holdout, read_split
from audit import ITEM_FIELDS
import check_withdraw
from validate_with_pmb_parser import check

WORK = ROOT / "repair_data_gen/splits/test"
QUEUES = ("test_full_review_v1", "test_supplement_v1")
COUNTS = dict(content_word=110, constant_name=118, constant_number=70,
              constant_temporal=67, tense=70, edge_label=55, forwarding=69)
FIELDS = (
    "item_id", "doc_id", "repair_type", "request_id", "site_index", "nl_clean",
    "nl_repair", "repair_onset", "sbn_clean", "sbn_repair", "reparandum_surface",
    "reparandum_synset", "reparandum_value", "repair_synset", "ambiguous",
    "wn_relation", "devices", "strategy", "interregnum", "audit_id", "source",
    "stage", "recovery_id", "label_source",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique(rows, field):
    keyed = {r[field]: r for r in rows}
    require(len(rows) == len(keyed) and "" not in keyed, f"Invalid or repeated {field}")
    return keyed


def save(path, data):
    """Allow exact rebuilds, never silently replace an existing export."""
    if path.exists():
        require(path.read_bytes() == data, f"{path} differs; use a new version")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(data)


def tsv(rows, fields):
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue().encode()


def save_json(path, payload):
    save(path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode())


def base_rows():
    path = WORK / "manifests/base_adjudication_v1.json"
    manifest = json.loads(path.read_bytes())
    selection = json.loads((WORK / "selection_policy.json").read_bytes())
    require(selection["version"] == manifest["policy"] and
            selection["selection"]["base_selection"] == "retain_all_eligible",
            "Unexpected selection policy")
    for name, digest in manifest["input_sha256"].items():
        require(sha(ROOT / name) == digest, f"Adjudication input changed: {name}")
    decisions = manifest["decisions"]
    used_decisions = set()
    rows = []
    for queue in QUEUES:
        audit = WORK / "audit" / queue
        queue_manifest = json.loads((audit / "manifest.json").read_bytes())
        for name, key in (("items.tsv", "items_sha256"), ("review_pool.tsv", "review_pool_sha256")):
            require(sha(audit / name) == queue_manifest[key], f"Queue changed: {queue}/{name}")
        pool = unique(read(audit / "review_pool.tsv"), "audit_id")
        items = unique(read(audit / "items.tsv"), "audit_id")
        labels = {}
        for line in (audit / "labels.jsonl").read_text().splitlines():
            label = json.loads(line)
            if label["annotator"] == manifest["annotator"]:
                labels[label["audit_id"]] = label
        require(items.keys() == pool.keys() == labels.keys(), f"Incomplete review: {queue}")
        for audit_id, row in sorted(pool.items()):
            require(all(row[k] == items[audit_id][k] for k in ITEM_FIELDS if k != "source"),
                    f"Review/graph mismatch: {audit_id}")
            label = labels[audit_id]
            state = label["meaning"] + "/" + label["natural"]
            if state == "yes/yes":
                require(audit_id not in decisions, f"Original double yes overwritten: {audit_id}")
                provenance = "human_double_yes"
            else:
                decision = decisions[audit_id]
                require(decision["original_label"] == state, f"Original label changed: {audit_id}")
                used_decisions.add(audit_id)
                if decision["decision"] != "include":
                    continue
                provenance = "confirmed_adjudication"
            new = {k: row.get(k, "") for k in FIELDS}
            new.update(item_id="rsr_test_" + audit_id.removeprefix("test"),
                       label_source=provenance)
            rows.append(new)
    require(used_decisions == decisions.keys(), "Unused adjudication records")
    require(Counter(r["repair_type"] for r in rows) == COUNTS, "Unexpected base counts")
    for field in ("item_id", "audit_id", "request_id", "nl_repair"):
        unique(rows, field)
    require(len({(r["doc_id"], r["repair_type"], r["site_index"]) for r in rows}) == len(rows),
            "Repeated source/type/site")
    return rows


def verify_bases(rows):
    split = ROOT / "data/pmb-5.1.0/split/en"
    paths = [split / name for name in ("test/standard.sbn", "train/gold.sbn", "dev/standard.sbn")]
    gold = {doc.doc_id: doc for doc in read_split(paths[0])}
    require(len(gold) == 1194 and not read_split.skipped, "PMB test inventory changed")
    held_ids = set(holdout.load())
    held_texts = set()
    for path in paths[1:]:
        other = read_split(path)
        require(not read_split.skipped, f"Skipped documents in {path}")
        held_ids.update(d.doc_id for d in other)
        held_texts.update(d.sentence for d in other)
    for r in rows:
        doc = gold[r["doc_id"]]
        require((r["nl_clean"], r["sbn_clean"]) == (doc.sentence, doc.raw),
                f"PMB source mismatch: {r['audit_id']}")
        require(r["doc_id"] not in held_ids and r["nl_clean"] not in held_texts,
                f"Source overlap: {r['audit_id']}")
        onset = int(r["repair_onset"])
        text = r["nl_repair"]
        require(r["interregnum"] == "none" and 2 <= onset < len(text) and
                text[onset-2:onset] == ", " and len(text.encode("utf-16-le")) == len(text)*2,
                f"Invalid insertion boundary: {r['audit_id']}")
        require(all(check(r["sbn_repair"])[key] == "ok"
                    for key in ("parse", "dag", "index", "magnitude", "edges", "boxes")),
                f"Graph failed checks: {r['audit_id']}")
        require(not any(check_withdraw.diff(r)), f"Withdrawal differs from PMB gold: {r['audit_id']}")
    return paths


def build_pairs(rows, policy):
    markers = policy["assignment_markers"]
    require(policy["mode"] == "paired" and policy["marked_variants_per_base"] == 1 and
            policy["retain_all_base"] and len(markers) == 5 and
            set(markers) == {"I mean", "no", "no wait", "actually", "sorry"} and
            policy["held_out_markers"] == ["sorry"] and
            policy["assignment"] ==
            "shuffle_base_ids_within_repair_type_then_cycle_five_markers_with_global_cursor",
            "Unexpected marker policy")
    helper_path = GEN / "dataset/insert_editing_expressions.py"
    spec = importlib.util.spec_from_file_location("test_marker_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rng = random.Random(policy["seed"])
    assignment = {}
    cursor = 0
    for kind in sorted(COUNTS):
        ids = sorted(r["item_id"] for r in rows if r["repair_type"] == kind)
        rng.shuffle(ids)
        for i in ids:
            assignment[i] = markers[cursor % len(markers)]
            cursor += 1
    base = [helper.derive(r) for r in rows]
    marked = [helper.derive(r, assignment[r["item_id"]]) for r in rows]
    for r, m in zip(rows, marked, strict=True):
        m["selection_reason"] = "paired_test"
        marker = m["editing_expression"]
        start, end = int(m["editing_start"]), int(m["repair_onset"])
        require(start == int(r["repair_onset"]) and int(m["editing_end"]) == start+len(marker) and
                m["nl_repair"][start:end] == marker+", " and
                m["nl_repair"][:start]+m["nl_repair"][end:] == r["nl_repair"],
                f"Invalid marker insertion: {r['item_id']}")
        require(all(m[key] == r[key] for key in ("sbn_clean", "sbn_repair", "doc_id", "request_id", "site_index")),
                f"Marker changed target or identity: {r['item_id']}")
    paired = base + marked
    unique(paired, "variant_id")
    unique(paired, "nl_repair")
    require(Counter(r["item_id"] for r in paired) == Counter({r["item_id"]: 2 for r in rows}),
            "Base/marked rows not paired")
    counts = Counter(r["editing_expression"] for r in marked)
    by_type = {}
    for kind in sorted(COUNTS):
        c = Counter(r["editing_expression"] for r in marked if r["repair_type"] == kind)
        require(set(c) == set(markers) and max(c.values())-min(c.values()) <= 1,
                f"Marker counts unbalanced for {kind}")
        by_type[kind] = {m: c[m] for m in markers}
    require(max(counts.values())-min(counts.values()) <= 1, "Global marker counts unbalanced")
    return marked, paired, dict(counts), by_type, assignment, helper_path


def main():
    rows = base_rows()
    sources = verify_bases(rows)
    policy_path = WORK / "marker_policy.json"
    policy = json.loads(policy_path.read_bytes())
    marked, paired, counts, per_type, assignments, helper = build_pairs(rows, policy)
    base_path = WORK / "dataset/test_rsr.base.tsv"
    marked_path = WORK / "dataset/test_rsr.marked.tsv"
    paired_path = WORK / "dataset/test_rsr.paired.tsv"
    outputs = {}
    for path, data in ((base_path, rows), (marked_path, marked), (paired_path, paired)):
        save(path, tsv(data, list(data[0])))
        require(read(path) == data, f"TSV round trip changed {path.name}")
        outputs[path.name] = dict(rows=len(data), sha256=sha(path))
    audit = WORK / "audit/test_marked_review_v1"
    order = sorted(marked, key=lambda r: r["item_id"])
    random.Random(policy["seed"] + 1).shuffle(order)
    items, mapping = [], {}
    for n, row in enumerate(order, 1):
        key = f"testmark{n:04d}"
        items.append({**{k: row[k] for k in ITEM_FIELDS}, "audit_id": key})
        mapping[key] = dict(item_id=row["item_id"], variant_id=row["variant_id"],
                            base_audit_id=row["audit_id"], marker=row["editing_expression"])
    save(audit / "items.tsv", tsv(items, ITEM_FIELDS))
    save_json(audit / "manifest.json", dict(
        status="assigned marked forms awaiting naturalness review",
        rows=len(items), items_sha256=sha(audit / "items.tsv"),
        marked_sha256=sha(marked_path), mapping=mapping,
    ))
    inputs = [WORK / "selection_policy.json", WORK / "manifests/base_adjudication_v1.json",
              policy_path, helper, *sources]
    save_json(WORK / "manifests/base_and_marked_v1.json", dict(
        status="base frozen; marked forms generated, pending naturalness review",
        base_count=len(rows), source_documents=len({r["doc_id"] for r in rows}),
        per_type=COUNTS, marker_counts=counts, markers_per_type=per_type,
        assignments=assignments, outputs=outputs,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        code_sha256=sha(Path(__file__)),
    ))
    print(json.dumps(dict(base=len(rows), marked=len(marked), paired=len(paired),
                          markers=counts, per_type=per_type, review=str(audit)), indent=2))


if __name__ == "__main__":
    main()
