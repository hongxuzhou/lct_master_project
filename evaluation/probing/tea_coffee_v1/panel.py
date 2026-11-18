"""Read and validate the shared-prefix stimulus tree without ML dependencies."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_panel(path=HERE / 'stimuli.json'):
    panel = json.loads(Path(path).read_text())
    rows = panel['checkpoints']
    by_id = {r['id']: r for r in rows}
    if len(rows) != 16 or len(by_id) != 16 or len({r['text'] for r in rows}) != 16:
        raise ValueError('Expected 16 unique checkpoint IDs and texts')
    seen = set()
    for row in rows:
        parent = row['parent']
        if parent is None:
            if seen or row['text'] != 'I ordered tea':
                raise ValueError('Unexpected initial input')
        elif parent not in seen or row['text'] != by_id[parent]['text'] + row['added']:
            raise ValueError(f"Invalid cumulative update: {row['id']}")
        seen.add(row['id'])
    memberships = {key: [] for key in by_id}
    for name, path_ids in panel['paths'].items():
        previous = None
        for key in path_ids:
            if key not in by_id or by_id[key]['parent'] != previous:
                raise ValueError(f'Invalid path: {name}')
            memberships[key].append(name)
            previous = key
    if any(not names for names in memberships.values()):
        raise ValueError('Checkpoint without a path')
    return panel, [{**row, 'paths': memberships[row['id']]} for row in rows]
