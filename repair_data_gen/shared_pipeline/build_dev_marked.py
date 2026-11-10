"""Append one marked version for every identified ambiguous dev base item."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import random
from pathlib import Path

from .analyse_split import ROOT, GEN, read, sha, write
from .build_dev_base import require, unique


WORKSPACE = ROOT / "repair_data_gen/splits/dev"


def main():
    policy_path = WORKSPACE / "marker_policy.json"
    policy = json.loads(policy_path.read_text())
    require(policy["mode"] == "cover_identified_ambiguity" and not policy["random_topup"],
            "unsupported marker selection policy")
    build_path = WORKSPACE / "manifests/base_build.json"
    build = json.loads(build_path.read_text())
    for path, digest in build["input_sha256"].items():
        require(sha(path) == digest, f"validated base input changed: {path}")
    base_path = ROOT / build["file"]
    require(sha(base_path) == build["sha256"], "base export changed")
    rows = read(base_path)
    unique(rows, "item_id")
    require(len(rows) == policy["base_items"] == 390, "unexpected base count")
    selected = [r for r in rows if r["ambiguous"]]
    require(len(selected) == policy["target_marked_items"] == 122, "ambiguity count differs from policy")
    require(all(r["marker_priority"] != "True" or r["ambiguous"] for r in rows),
            "marker priority item omitted from ambiguity pool")
    markers = policy["markers"]
    require(set(markers) == {"I mean", "no", "no wait", "actually"}, "unexpected markers")
    require(not set(markers) & set(policy["held_out_markers"]), "held-out marker selected")
    helper_path = GEN / "dataset/insert_editing_expressions.py"
    spec = importlib.util.spec_from_file_location("train_marker_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rng, assignment, cursor = random.Random(policy["seed"] + 1), {}, 0
    for repair_type in sorted({r["repair_type"] for r in rows}):
        ids = sorted(r["item_id"] for r in selected if r["repair_type"] == repair_type)
        rng.shuffle(ids)
        for item_id in ids:
            assignment[item_id] = markers[cursor % len(markers)]
            cursor += 1
    base_variants = [helper.derive(r) for r in rows]
    marked = [helper.derive(r, assignment[r["item_id"]]) for r in selected]
    by_id = {r["item_id"]: r for r in rows}
    for row in marked:
        original = by_id[row["item_id"]]
        start, end = int(row["editing_start"]), int(row["repair_onset"])
        require(start == int(original["repair_onset"]), "insertion begins at wrong position")
        require(int(row["editing_end"]) == start + len(row["editing_expression"]), "marker span differs")
        require(row["nl_repair"][start:end] == row["editing_expression"] + ", ", "inserted segment differs")
        require(row["nl_repair"][:start] + row["nl_repair"][end:] == original["nl_repair"],
                "marked text does not recover the exact base")
        require(all(row[k] == original[k] for k in ("sbn_clean", "sbn_repair", "site_index", "request_id")),
                "marker changed a semantic target or identity")
    counts = Counter(r["editing_expression"] for r in marked)
    require(max(counts.values()) - min(counts.values()) <= 1, "unbalanced global markers")
    per_type = {}
    for t in sorted({r["repair_type"] for r in rows}):
        subset = [r for r in marked if r["repair_type"] == t]
        tc = Counter(r["editing_expression"] for r in subset)
        require(max((tc[m] for m in markers), default=0) - min((tc[m] for m in markers), default=0) <= 1,
                f"unbalanced markers for {t}")
        per_type[t] = dict(base=sum(r["repair_type"] == t for r in rows), marked=len(subset),
                           markers={m: tc[m] for m in markers})
    augmented = base_variants + marked
    unique(augmented, "variant_id")
    require(len(augmented) == 512, "unexpected augmented count")
    require(Counter(r["item_id"] for r in augmented) ==
            Counter({r["item_id"]: 2 if r["item_id"] in assignment else 1 for r in rows}),
            "unexpected base/marked row weighting")
    columns = list(dict.fromkeys([k for k in rows[0] if k != "interregnum"] + list(helper.EXTRA_COLUMNS)))
    outputs = []
    for filename, data in [("dev_rsr.marked.tsv", marked), ("dev_rsr.augmented.tsv", augmented)]:
        path = WORKSPACE / "dataset" / filename
        write(path, data, columns)
        require(read(path) == data, "TSV round trip changed marked data")
        outputs.append(dict(file=str(path.relative_to(ROOT)), rows=len(data), sha256=sha(path)))
    require(sha(base_path) == build["sha256"], "marker build changed base")
    manifest = dict(status="marked RSR built and validated; fluent component not yet assembled",
                    base_items=390, marked_items=122, augmented_items=512,
                    marked_share_of_base=122/390, covered_ambiguity_items=122,
                    random_topup_items=0, marker_counts=dict(counts), per_type=per_type,
                    checks=dict(exact_base_recovery=122, unchanged_semantic_targets=122,
                                unique_variants=512, held_out_marker_occurrences=0),
                    outputs=outputs, assignments=assignment,
                    input_sha256={str(p): sha(p) for p in [base_path, build_path, policy_path, helper_path]},
                    code_sha256={str(Path(__file__)): sha(Path(__file__))})
    path = WORKSPACE / "manifests/marked_build.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"assignments", "input_sha256", "code_sha256"}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
