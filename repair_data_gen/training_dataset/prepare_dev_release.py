"""Stage an explicit allowlist for publishing the accepted dev snapshot.

    python -m repair_data_gen.training_dataset.prepare_dev_release --stage-dir PATH

This prepares local files only. Uploading and remote readback are separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .build import HERE, ROOT, sha256


def main():
    if (HERE/'fluent_group.acceptance.json').exists():
        raise ValueError('Schema-v2 train/dev must be staged together; use training_dataset.prepare_release')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    report_dir = ROOT / "repair_data_gen/splits/dev/analysis/unified_validation"
    report_path = report_dir / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    build = json.loads((HERE / "validation.manifest.json").read_text(encoding="utf-8"))
    if report["status"] != "passed" or report["errors"]:
        raise ValueError("development acceptance has not passed")
    for path, expected in report["input_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"accepted input changed: {path}")
    for filename, expected in report["outputs"].items():
        if sha256(report_dir / filename) != expected:
            raise ValueError(f"acceptance output changed: {filename}")
    if sha256(HERE / build["file"]) != report["validation_sha256"]:
        raise ValueError("validation Parquet differs from accepted data")

    acceptance = {
        "status": "passed", "split": "validation", "file": build["file"],
        "sha256": report["validation_sha256"], "rows": build["rows"],
        "assembly_manifest_sha256": sha256(HERE / "validation.manifest.json"),
        "local_acceptance_report": str(report_path.relative_to(ROOT)),
        "local_acceptance_report_sha256": sha256(report_path),
        "checks": report["checks"], "tokenization": report["tokenization"],
        "versions": report["versions"],
        "limitations": {key: value for key, value in report["notes"].items()
                        if key not in {"freeze", "hub_publication"}},
        "sources": build["sources"],
        "acceptance_validator": {
            "path": "repair_data_gen/training_dataset/validate_dev.py",
            "sha256": sha256(HERE / "validate_dev.py"),
        },
        "release_preparer": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__)),
        },
    }
    record = HERE / "validation.acceptance.json"
    record.write_text(json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stage = args.stage_dir.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    if any(stage.iterdir()):
        raise ValueError("stage directory must be empty")
    files = {"README.md": HERE / "README.md", build["file"]: HERE / build["file"],
             "validation.manifest.json": HERE / "validation.manifest.json",
             "validation.acceptance.json": record}
    for relative, source in files.items():
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != sha256(source):
            raise ValueError(f"staging changed {relative}")
    print(json.dumps({"stage": str(stage), "files": {name: sha256(source) for name, source in files.items()}}, indent=2))


if __name__ == "__main__":
    main()
