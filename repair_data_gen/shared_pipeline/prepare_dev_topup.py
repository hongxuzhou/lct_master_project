"""Prepare missing-request retries and a second temporal dev pass locally.

No cluster submission or modification of existing samples/results occurs.
First-pass timeouts and temporal pass 2 use separate tags and raw directories.
"""
from collections import Counter
import json
from pathlib import Path
import random

from .analyse_split import ROOT, read, write, sha
from .split_config import load_config
from .select_candidates import common_row
from .prepare_dev_review import GEN
from sample_temporal_extra import used_site
import select_docs as selection


def main():
    workspace = ROOT / "repair_data_gen/splits/dev"
    config = load_config(workspace / "config.json", check_inputs=True)
    missing_path = workspace / "analysis/first_pass/missing.tsv"
    pool_path = workspace / "analysis/dev_review_v1/review_pool.tsv"
    pool = read(pool_path)
    missing = read(missing_path)
    samples = workspace / "samples"
    outputs, stages = [], []

    def save(tag, rows, template, prompt):
        if not rows:
            return
        path = samples / (tag + ".tsv")
        if path.exists():
            assert read(path) == rows, f"existing prepared sample differs: {path}"
        else:
            write(path, rows)
        outputs.append(dict(path=str(path.relative_to(ROOT)), rows=len(rows), sha256=sha(path)))
        text = (workspace / "hpc" / template).read_text()
        old = "dev_base" if prompt == "base" else "dev_temporal_p1"
        text = text.replace(old, tag)
        text = "# Prepared follow-up; submit separately from the original DAG.\n" + text
        if tag == "dev_base_retry1":
            text = text.replace(
                "# Five base-prompt types, 75 candidates each. Output is isolated from the\n# frozen train lineage under ../splits/dev/results/.",
                "# Retry the four missing base requests only; keep original outputs unchanged.")
        else:
            comment = ("# Retry the two missing first-pass temporal requests only." if tag.endswith("retry1")
                       else "# Second temporal pass: five new sites and 42 same-site alternative values.")
            text = text.replace("# Every eligible PMB dev temporal document, with the first site pinned.", comment)
        sub = workspace / "hpc" / ("run_" + tag + ".sub")
        if sub.exists():
            assert sub.read_text() == text, f"existing submit file differs: {sub}"
        else:
            sub.write_text(text)
        stages.append((tag, sub))

    for stage, tag, template, prompt in [
        ("dev_base", "dev_base_retry1", "run_dev_base.sub", "base"),
        ("dev_temporal_p1", "dev_temporal_p1_retry1", "run_dev_temporal_p1.sub", "temporal_r2"),
    ]:
        rows = [{k:v for k,v in row.items() if k != "stage" and v != ""}
                for row in missing if row["stage"] == stage]
        # Use the original schema and values exactly, rather than fields
        # introduced by the union schema in missing.tsv.
        original = {r["request_id"]:r for r in read(samples / (stage + ".tsv"))}
        rows = [original[r["request_id"]] for r in rows]
        save(tag, rows, template, prompt)

    docs = {d.doc_id:d for d in selection.read_split(config.pmb_source)}
    passed = [r for r in pool if r["repair_type"] == "constant_temporal"]
    assert len({r["doc_id"] for r in passed}) == len(passed)
    temporal, kinds = [], Counter()
    for first in sorted(passed, key=lambda r:r["doc_id"]):
        doc = docs[first["doc_id"]]
        sites = sorted(set(selection.site_temporal(doc)))
        used = used_site(sites, first)
        assert used is not None, first["request_id"]
        free = [s for s in sites if s != used]
        site, kind, avoid = ((free[0], "new_site", "none") if free else
                             (used, "same_site", first["reparandum_value"]))
        pos, value = site
        row = common_row(doc, "constant_temporal", sites,
                         f"dev_temporal_p2::constant_temporal::{doc.doc_id}::site_{pos}")
        row = {k:str(v) for k,v in row.items()}
        row.update(pinned_site=f"{pos} {doc.concepts[pos].synset} {value}",
                   avoid_value=avoid, kind=kind, first_pass_request_id=first["request_id"])
        temporal.append(row)
        kinds[kind] += 1
    random.Random(config.seed).shuffle(temporal)
    save("dev_temporal_p2", temporal, "run_dev_temporal_p1.sub", "temporal_r2")
    cluster_root = "/nethome/honzhou/thesis_zone/splits/dev/hpc/"
    dag = ["# Prepared follow-up DAG; no jobs have been submitted by this builder."]
    for tag, path in stages:
        dag.append(f"JOB {tag.upper()} {cluster_root}{path.name}")
    for (before, _), (after, _) in zip(stages, stages[1:]):
        dag.append(f"PARENT {before.upper()} CHILD {after.upper()}")
    (workspace / "hpc/run_dev_followup.dag").write_text("\n".join(dag) + "\n")
    manifest = dict(status="prepared locally; not submitted", split="dev", seed=config.seed,
                    config_sha256=sha(config.path), review_policy_sha256=sha(workspace / "review_policy.json"),
                    source_pool_sha256=sha(pool_path), missing_sha256=sha(missing_path),
                    builder_sha256=sha(__file__), temporal_kinds=dict(kinds),
                    total_requests=sum(o["rows"] for o in outputs), outputs=outputs,
                    prompt_versions=dict(base="base", temporal="temporal_r2"),
                    dag="repair_data_gen/splits/dev/hpc/run_dev_followup.dag")
    (workspace / "manifests/followup_samples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
