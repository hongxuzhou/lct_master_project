"""Prepare only the selected adapter in a persistent, revision-pinned store."""
import json
import os
from pathlib import Path
import re
import tempfile

from panel import digest


def selected_entry(manifest, seed):
    matches = [r for r in json.loads(Path(manifest).read_text())
               if (r['model_size'], r['condition'], str(r['seed'])) == ('9B', 'C3', str(seed))]
    if len(matches) != 1:
        raise ValueError('Manifest must identify exactly one 9B C3 checkpoint for this seed')
    return matches[0]


def prepare_adapter(manifest, seed, store, repo, revision=None, allow_download=True):
    entry = selected_entry(manifest, seed)
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo):
        raise ValueError('Expected owner/repository')
    if revision is not None and not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('--adapter-revision must be a full commit SHA')
    root = Path(store) / repo.replace('/', '--')
    pin_path = root / 'revision.json'
    if revision is None:
        if pin_path.is_file():
            pin = json.loads(pin_path.read_text())
            if pin['repo'] != repo:
                raise ValueError('Adapter store repository mismatch')
            revision = pin['revision']
        elif not allow_download:
            raise FileNotFoundError(f'No local revision pin: {pin_path}')
        else:
            from huggingface_hub import HfApi
            revision = HfApi().model_info(repo).sha
            if not re.fullmatch(r'[a-f0-9]{40}', revision or ''):
                raise ValueError('Hub did not return a full commit SHA')
            root.mkdir(parents=True, exist_ok=True)
            # Concurrent seeds share the first successfully written pin.
            with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({'repo': repo, 'revision': revision}, stream)
            try:
                try:
                    os.link(temporary, pin_path)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink()
            pin = json.loads(pin_path.read_text())
            if pin['repo'] != repo:
                raise ValueError('Adapter store repository mismatch')
            revision = pin['revision']
    if not re.fullmatch(r'[a-f0-9]{40}', revision or ''):
        raise ValueError('Invalid local adapter revision pin')
    run_id = entry['run_id']
    if not re.fullmatch(r'formal-habrok-[\w-]+', run_id):
        raise ValueError('Invalid frozen run ID')
    checkpoint_name = f"checkpoint-{int(entry['best_step'])}"
    target = root / revision
    checkpoint = target / run_id / checkpoint_name
    config = checkpoint.parent / 'config.json'
    files = [(config, entry['config_sha256'])]
    for name, sha in entry['files'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid frozen checkpoint filename')
        files.append((checkpoint / name, sha))
    missing = []
    for path, sha in files:
        if path.exists():
            if not path.is_file() or digest(path) != sha:
                raise ValueError(f'Existing adapter file hash mismatch; preserved: {path}')
        else:
            missing.append((path, sha))
    if missing and not allow_download:
        raise FileNotFoundError(f'Missing adapter file: {missing[0][0]}')
    for path, sha in missing:
        from huggingface_hub import hf_hub_download
        filename = path.relative_to(target).as_posix()
        print(f'Downloading {repo}@{revision}: {filename}', flush=True)
        hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                        repo_type='model', local_dir=target)
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f'Downloaded adapter file hash mismatch: {path}')
    return checkpoint, config, dict(repo=repo, revision=revision, store=str(root.resolve()),
                                    downloaded_files=[p.relative_to(target).as_posix() for p, _ in missing])
