"""Publish the explicit challenge allowlist atomically, then verify readback."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = 'Shrikes/repair-sbn'
ALLOWLIST = {
    'README.md', 'SCHEMA.md', 'later_reference/README.md',
    'later_reference/test.parquet', 'later_reference/manifest.json',
    'later_reference/validation.json', 'later_reference/adjudications.json',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage_dir.resolve()
    manifest = json.loads((stage / 'stage.manifest.json').read_text())
    assert manifest['repo_id'] == REPO
    assert manifest['remote_head_verified']
    hashes = manifest['files_sha256']
    assert set(hashes) == ALLOWLIST
    for name, expected in hashes.items():
        assert sha(stage / name) == expected, name
        assert sha(HERE / name) == expected, ('Working file changed', name)
    validation = json.loads((stage / 'later_reference/validation.json').read_text())
    assert validation['status'] == 'passed'
    for name, expected in validation['input_sha256'].items():
        assert sha(ROOT / name) == expected, ('Validated input changed', name)
    parent = manifest['verified_parent_revision']
    api = HfApi()
    record_path = HERE / 'later_reference.publication.json'
    pending_path = HERE / 'later_reference.publication.pending.json'
    pending = json.loads(pending_path.read_text()) if pending_path.exists() else None
    if pending:
        assert pending['files_sha256'] == hashes, 'Existing publication differs from stage'
        revision = pending['revision']
    else:
        head = api.repo_info(REPO, repo_type='dataset').sha
        assert head == parent, ('Hub parent changed', head, parent)
        for name, expected in manifest['hub_baseline']['files_sha256'].items():
            remote = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=parent))
            assert sha(remote) == expected, ('Hub baseline changed', name)
        commit = api.create_commit(
            repo_id=REPO, repo_type='dataset', parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(stage / name)) for name in sorted(ALLOWLIST)],
            commit_message='Add paired later-reference challenge as a test-only configuration',
            commit_description='Add 104 fluent/marked-RSR quasi-minimal pairs with full SBN gold and reference indices. Include challenge card, schema, adjudications and validation. Preserve the existing default train/validation files.',
        )
        revision = commit.oid
        pending = dict(status='commit_created_readback_pending', repo_id=REPO,
                       revision=revision, parent_revision=parent, files_sha256=hashes)
        save(pending_path, pending)
        print('Commit created: ' + revision, flush=True)
    # Preserve every pre-existing non-document file, not only the two splits.
    old_files = {f.rfilename for f in api.repo_info(REPO, repo_type='dataset', revision=parent).siblings}
    new_files = {f.rfilename for f in api.repo_info(REPO, repo_type='dataset', revision=revision).siblings}
    assert old_files <= new_files
    verified = {}
    for name in sorted(old_files | ALLOWLIST):
        remote = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=revision))
        if name in hashes:
            expected = hashes[name]
        else:
            prior = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=parent))
            expected = sha(prior)
        assert sha(remote) == expected, ('Remote readback mismatch', name)
        verified[name] = sha(remote)
    remote_parquet = Path(hf_hub_download(REPO, 'later_reference/test.parquet', repo_type='dataset', revision=revision))
    assert pq.read_table(remote_parquet).equals(pq.read_table(stage / 'later_reference/test.parquet'), check_metadata=True)
    from datasets import load_dataset, get_dataset_config_names
    configs = get_dataset_config_names(REPO, revision=revision)
    assert 'default' in configs and 'later_reference' in configs
    loaded = load_dataset(REPO, 'later_reference', split='test', revision=revision,
                          cache_dir='/tmp/repairsbn-later-reference-remote-check')
    expected_rows = pq.read_table(stage / 'later_reference/test.parquet').to_pylist()
    assert len(loaded) == 208 and loaded.to_list() == expected_rows
    record = dict(
        pending, status='published_and_verified', config='later_reference', split='test',
        dataset_version='later-reference-v1', rows=208, pairs=104, schema_fields=18,
        commit_url=f'https://huggingface.co/datasets/{REPO}/commit/{revision}',
        verified_files_sha256=verified,
        all_preexisting_non_document_files_unchanged=True,
        hf_config_discovery_verified=True, hf_dataset_load_exact_readback=True,
        publisher_sha256=sha(Path(__file__)),
    )
    save(record_path, record)
    save(pending_path, {**record, 'status': 'readback_completed'})
    # Keep the default train/dev data revision pinned to its existing release,
    # while teaching documentation-only updates about the new Hub head.
    docs_path = HERE / 'documentation.publication.json'
    previous_docs = json.loads(docs_path.read_text())
    save(docs_path, dict(
        status='published_and_verified', repo_id=REPO, revision=revision,
        parent_revision=parent,
        data_release_revision=previous_docs['data_release_revision'],
        documentation_sha256={name: hashes[name] for name in ('README.md', 'SCHEMA.md')},
        commit_url=record['commit_url'], verified_files=len(verified),
        data_files_unchanged=False, default_data_files_unchanged=True,
        added_configuration='later_reference',
        challenge_publication='later_reference.publication.json',
        publisher_sha256=sha(Path(__file__)),
    ))
    print(json.dumps({key: record[key] for key in ('status', 'revision', 'commit_url', 'rows', 'pairs')}, indent=2))


if __name__ == '__main__':
    main()
