"""Build a small inference-only LST bundle using standard Python."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile
import tempfile
import shutil

from panel import HERE, digest, read_panel
from adapters import selected_entry

ROOT = HERE.parents[2]
NAME = 'repairsbn-restart-probe-v1'


def build(output):
    read_panel()
    frozen = ROOT / 'evaluation/runs/formal-test-20261113/frozen-habrok.json'
    rows = [selected_entry(frozen, seed) for seed in (24, 42, 4224)]
    names = ('run.py', 'panel.py', 'adapters.py', 'stimuli.json', 'lst_preflight.py',
             'lst_worker.sh', 'lst.sub', 'README.md')
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / NAME
        files = [HERE / name for name in names] + [ROOT / 'model/lora/data.py']
        for path in files:
            target = stage / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (stage / 'frozen-9B-C3.json').write_text(json.dumps(rows, indent=2) + '\n')
        (stage / 'logs').mkdir()
        hashes = {p.relative_to(stage).as_posix(): digest(p)
                  for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage / 'bundle.json').write_text(json.dumps(dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_frozen_manifest_sha256=digest(frozen), files=hashes), indent=2) + '\n')
        # Python tarfile avoids macOS resource-fork headers. No datasets or weights.
        with output.open('xb') as stream:
            with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.PAX_FORMAT) as archive:
                archive.add(stage, arcname=NAME)
    print(f'{output}: {output.stat().st_size} bytes; sha256={digest(output)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
