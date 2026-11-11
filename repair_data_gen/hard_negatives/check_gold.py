"""Check the original PMB gold graphs for the frozen hard-negative hold-out."""

import sys
from collections import Counter
from pathlib import Path

from holdout import load

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from sbn_env import ensure_on_path

ensure_on_path()

from sbn_smatch import SBNGraph
from sbn_spec import SBNError


def check_gold(path: Path) -> tuple[str, str]:
    try:
        graph = SBNGraph().from_path(path)
    except SBNError as exc:
        return "parse_error", str(exc)
    if not graph.is_dag:
        return "cycle", ""
    missing = [f"{kind.value}[{index}]" for (kind, index), data in graph.nodes(data=True)
               if "type" not in data]
    if missing:
        return "unresolved_index", ", ".join(missing)
    if graph.is_possibly_ill_formed:
        return "unresolved_index", "reader treated an out-of-range index as a constant"
    try:
        graph.to_penman_string()
    except SBNError as exc:
        return "penman_error", str(exc)
    return "ok", ""


def main() -> None:
    gold = ROOT / "data/pmb-5.1.0/data/en/gold"
    counts = Counter()
    for doc_id in sorted(load()):
        status, detail = check_gold(gold / doc_id / "en.drs.sbn")
        counts[status] += 1
        if status != "ok":
            print(f"{doc_id}: {status}" + (f": {detail}" if detail else ""))
    print(f"{counts['ok']}/{counts.total()} pass structural checks")


if __name__ == "__main__":
    main()
