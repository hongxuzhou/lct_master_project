"""Load and validate reproducible configuration for one PMB split.

Paths in a config are repository-relative. Loading a config never creates
files or changes data. Use ``--check-inputs`` when the local PMB checkout is
available and input existence should be verified.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPAIR_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPAIR_ROOT.parent
REPAIR_TYPES = (
    "content_word",
    "constant_name",
    "constant_number",
    "constant_temporal",
    "tense",
    "edge_label",
    "forwarding",
)
STATUSES = {"working", "frozen", "disabled"}
AUDIT_POLICIES = {"all_auto_pass", "sample", "none"}


def _require_keys(obj: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - obj.keys())
    if missing:
        raise ValueError(f"{where} is missing: {', '.join(missing)}")


def _repo_path(value: str, field: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a repository-relative path: {value!r}")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository: {value!r}") from exc
    return resolved


@dataclass(frozen=True)
class MarkerPolicy:
    mode: str
    share_of_base: float
    markers: tuple[str, ...]
    held_out_markers: tuple[str, ...]


@dataclass(frozen=True)
class RepairSplitConfig:
    path: Path
    schema_version: int
    status: str
    split: str
    pmb_source: Path
    item_id_prefix: str
    seed: int
    base_cap_per_type: dict[str, int]
    candidate_plan: dict[str, Any]
    prompt_versions: dict[str, str]
    marker_policy: MarkerPolicy
    audit_policy: str
    pair_exclusion_sources: tuple[Path, ...]

    @property
    def workspace(self) -> Path:
        return self.path.parent

    def summary(self) -> dict[str, Any]:
        rel = lambda p: str(p.relative_to(PROJECT_ROOT))
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "split": self.split,
            "workspace": rel(self.workspace),
            "pmb_source": rel(self.pmb_source),
            "item_id_prefix": self.item_id_prefix,
            "seed": self.seed,
            "base_cap_per_type": self.base_cap_per_type,
            "candidate_plan": self.candidate_plan,
            "prompt_versions": self.prompt_versions,
            "marker_policy": {
                "mode": self.marker_policy.mode,
                "share_of_base": self.marker_policy.share_of_base,
                "markers": list(self.marker_policy.markers),
                "held_out_markers": list(self.marker_policy.held_out_markers),
            },
            "audit_policy": self.audit_policy,
            "pair_exclusion_sources": [rel(p) for p in self.pair_exclusion_sources],
        }


def load_config(path: str | Path, *, check_inputs: bool = False) -> RepairSplitConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    _require_keys(
        raw,
        {
            "schema_version",
            "status",
            "split",
            "pmb_source",
            "item_id_prefix",
            "seed",
            "base_cap_per_type",
            "candidate_plan",
            "prompt_versions",
            "marker_policy",
            "audit_policy",
            "pair_exclusion_sources",
        },
        "config",
    )
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version: {raw['schema_version']!r}")
    if raw["status"] not in STATUSES:
        raise ValueError(f"unknown status: {raw['status']!r}")
    if raw["split"] not in {"train", "dev", "test"}:
        raise ValueError(f"unknown split: {raw['split']!r}")
    if config_path.parent.name != raw["split"]:
        raise ValueError("config split must match its parent directory")
    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
        raise ValueError("seed must be an integer")

    caps = raw["base_cap_per_type"]
    if set(caps) != set(REPAIR_TYPES):
        raise ValueError("base_cap_per_type must contain exactly the seven repair types")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in caps.values()):
        raise ValueError("every base cap must be a non-negative integer")

    candidate_plan = raw["candidate_plan"]
    _require_keys(
        candidate_plan,
        {"base_per_type", "constant_value_caps", "temporal", "edge_direction",
         "edge_place"},
        "candidate_plan",
    )
    base_types = {
        "content_word", "constant_name", "constant_number", "tense", "forwarding"
    }
    if set(candidate_plan["base_per_type"]) != base_types:
        raise ValueError("candidate_plan.base_per_type must contain the five base-prompt types")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
           for v in candidate_plan["base_per_type"].values()):
        raise ValueError("candidate base quotas must be non-negative integers")
    constant_types = {"constant_name", "constant_number"}
    if set(candidate_plan["constant_value_caps"]) != constant_types:
        raise ValueError("candidate_plan.constant_value_caps must contain name and number")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
           for v in candidate_plan["constant_value_caps"].values()):
        raise ValueError("constant value caps must be non-negative integers")
    expected_modes = {
        "temporal": "all_eligible_first_site",
        "edge_direction": "all_eligible",
        "edge_place": "all_eligible_place_only",
    }
    for key, value in expected_modes.items():
        if candidate_plan[key] != value:
            raise ValueError(f"candidate_plan.{key} must be {value!r}")

    marker_raw = raw["marker_policy"]
    _require_keys(
        marker_raw,
        {"mode", "share_of_base", "markers", "held_out_markers"},
        "marker_policy",
    )
    if marker_raw["mode"] not in {"append_share", "paired", "none"}:
        raise ValueError(f"unknown marker mode: {marker_raw['mode']!r}")
    share = marker_raw["share_of_base"]
    if not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 <= share <= 1:
        raise ValueError("marker share_of_base must be between 0 and 1")
    markers = tuple(marker_raw["markers"])
    held = tuple(marker_raw["held_out_markers"])
    if len(set(markers)) != len(markers) or len(set(held)) != len(held):
        raise ValueError("marker lists must not contain duplicates")
    if set(markers) & set(held):
        raise ValueError("used and held-out markers must be disjoint")
    if raw["audit_policy"] not in AUDIT_POLICIES:
        raise ValueError(f"unknown audit_policy: {raw['audit_policy']!r}")

    pmb_source = _repo_path(raw["pmb_source"], "pmb_source")
    exclusions = tuple(
        _repo_path(value, "pair_exclusion_sources")
        for value in raw["pair_exclusion_sources"]
    )
    if check_inputs:
        missing = [p for p in (pmb_source, *exclusions) if not p.is_file()]
        if missing:
            names = ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing)
            raise FileNotFoundError(f"missing configured inputs: {names}")

    return RepairSplitConfig(
        path=config_path,
        schema_version=raw["schema_version"],
        status=raw["status"],
        split=raw["split"],
        pmb_source=pmb_source,
        item_id_prefix=raw["item_id_prefix"],
        seed=raw["seed"],
        base_cap_per_type={name: caps[name] for name in REPAIR_TYPES},
        candidate_plan=candidate_plan,
        prompt_versions=dict(raw["prompt_versions"]),
        marker_policy=MarkerPolicy(
            mode=marker_raw["mode"],
            share_of_base=float(share),
            markers=markers,
            held_out_markers=held,
        ),
        audit_policy=raw["audit_policy"],
        pair_exclusion_sources=exclusions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, check_inputs=args.check_inputs)
    print(json.dumps(config.summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
