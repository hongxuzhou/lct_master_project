"""Build and stage only the later-reference test and its public documentation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARKED = HERE.parent / 'later_reference_challenge_marked_v1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(MARKED))
from validate_current import main as validate_sources
from pipeline import build, parse_sbn_line

SCHEMA = pa.schema([
    (name, pa.string()) for name in (
        'example_id', 'pair_id', 'family_id', 'partition_unit', 'text',
        'sbn_repair', 'language_condition', 'variant', 'editing_expression',
        'data_origin', 'construction_type', 'referent_type', 'source', 'label_source',
    )
] + [
    ('reference_group_index', pa.int64()),
    ('antecedent_indices', pa.list_(pa.field('element', pa.int64()))),
    ('withdrawn_antecedent_indices', pa.list_(pa.field('element', pa.int64()))),
    ('quantity', pa.int64()),
])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir', type=Path, required=True)
    parser.add_argument('--hub-baseline', type=Path, help='Read-only Hub verification receipt with revision and four file hashes')
    args = parser.parse_args()
    stage = args.stage_dir.resolve()
    allowed = {
        'README.md', 'SCHEMA.md', 'later_reference/README.md',
        'later_reference/test.parquet', 'later_reference/manifest.json',
        'later_reference/validation.json', 'later_reference/adjudications.json',
    }
    # Publication needs a separate explicit command. This script has no Hub writes.
    if stage == HERE or HERE in stage.parents and stage.name in {'data', 'later_reference'}:
        raise ValueError('Choose a separate staging directory')
    if stage.exists():
        extras = {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()} - allowed - {'stage.manifest.json'}
        if extras:
            raise ValueError('Unexpected staged files: ' + str(extras))
    baseline = {f'data/{s}.parquet': sha(HERE / 'data' / f'{s}.parquet') for s in ('train', 'validation')}
    validate_sources()
    validation_path = MARKED / 'derived/fresh_validation.json'
    assert json.loads(validation_path.read_text())['summary']['failed_rows'] == 0
    specs = [s for f in ('objects.json', 'subjects.json', 'names.json') for s in json.loads((MARKED / f).read_text())]
    spec_map = {s['id']: s for s in specs}
    annotations = [json.loads(line) for line in (MARKED / 'derived/annotations.jsonl').read_text().splitlines()]
    rows = []
    prefix = 'later_reference_v1::'
    for item in annotations:
        s = spec_map[item['pair_id']]
        variant = item['language_condition']
        draft = build(s, variant)
        positions = {node['id']: i for i, node in enumerate(draft.nodes())}
        assert draft.serialize() == item['sbn_repair']
        members = [positions['a'], positions['b']]
        group = positions['group']
        parsed = parse_sbn_line(item['sbn_repair'])
        actual = [group + int(value) for role, value in parsed.concepts[group].roles if role == 'Sub']
        assert sorted(actual) == sorted(members)
        rows.append(dict(
            example_id=prefix + item['id'], pair_id=prefix + s['id'],
            family_id=prefix + s['family_id'], partition_unit=prefix + s['id'],
            text=item['sentence'], sbn_repair=item['sbn_repair'],
            language_condition=variant,
            variant='rsr_editing_expression' if variant == 'rsr' else 'fluent_split_reference',
            editing_expression=s['editing_expression'] if variant == 'rsr' else None,
            data_origin='synthetic_rsr' if variant == 'rsr' else 'synthetic_fluent',
            construction_type='separate_subjects' if s['builder'] == 'subject' else 'separate_objects',
            referent_type=s['referent_type'],
            source='independently_authored_later_reference_v1',
            label_source='synthetic_annotation; author_surface_review' + ('; user_approved_item_adjudication' if 'adjudication' in s else ''),
            reference_group_index=group, antecedent_indices=members,
            withdrawn_antecedent_indices=[members[0]] if variant == 'rsr' else [],
            quantity=s['quantity'],
        ))
    assert len(rows) == len({r['example_id'] for r in rows}) == 208
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['pair_id']].append(row)
    assert len(pairs) == 104
    for pair in pairs.values():
        assert sorted(r['language_condition'] for r in pair) == ['fluent', 'rsr']
        assert len({r['family_id'] for r in pair}) == len({r['partition_unit'] for r in pair}) == 1
    norm = lambda text: ' '.join(re.findall(r'\w+', text.casefold()))
    existing, existing_ids = {}, set()
    for split in ('train', 'validation'):
        table = pq.read_table(HERE / 'data' / f'{split}.parquet')
        for r in table.select(['text', 'example_id', 'family_id', 'partition_unit']).to_pylist():
            existing.setdefault(norm(r['text']), []).append([split, r['example_id']])
            existing_ids.update(r[k] for k in ('example_id', 'family_id', 'partition_unit') if r[k])
    overlaps = [{'example_id': r['example_id'], 'matches': existing[norm(r['text'])]} for r in rows if norm(r['text']) in existing]
    assert not overlaps
    assert all(r[k] not in existing_ids for r in rows for k in ('example_id', 'family_id', 'partition_unit'))
    folder = HERE / 'later_reference'
    folder.mkdir(exist_ok=True)
    output = folder / 'test.parquet'
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, output, compression='zstd')
    assert pq.read_table(output).equals(table, check_metadata=True)
    from datasets import load_dataset
    loaded = load_dataset('parquet', data_files={'test': str(output)}, split='test', cache_dir='/tmp/repairsbn-later-reference-hf-check')
    assert len(loaded) == 208 and loaded.to_list() == rows
    frontmatter = (HERE / 'README.md').read_text().split('---', 2)[1]
    config = yaml.safe_load(frontmatter)['configs']
    assert next(c for c in config if c['config_name'] == 'default')['default'] is True
    assert next(c for c in config if c['config_name'] == 'later_reference')['data_files'] == [dict(split='test', path='later_reference/test.parquet')]
    assert baseline == {name: sha(HERE / name) for name in baseline}
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [
        *(MARKED / f for f in ('objects.json', 'subjects.json', 'names.json')),
        MARKED / 'derived/annotations.jsonl', MARKED / 'derived/graphs.json',
        MARKED / 'adjudications.json', validation_path, Path(__file__),
    ]}
    report = dict(
        status='passed', validation_scope='local_prepublication', config='later_reference', split='test',
        rows=208, pairs=104, schema_fields=len(SCHEMA),
        marker_counts=dict(Counter(s['editing_expression'] for s in specs)),
        current_export_failed_graphs=0, normalised_full_input_overlap=overlaps,
        train_validation_preserved_sha256=baseline,
        id_family_partition_overlap=0, parquet_exact_readback=True,
        hf_datasets_local_load_exact_readback=True,
        metadata_config_check=True, input_sha256=inputs,
        parquet_sha256=sha(output),
        limitations=['No semantic decontamination claim.', 'Remote publication verification is recorded separately.'],
    )
    write_json(folder / 'validation.json', report)
    shutil.copyfile(MARKED / 'adjudications.json', folder / 'adjudications.json')
    write_json(folder / 'manifest.json', dict(
        dataset_version='later-reference-v1', config='later_reference', split='test',
        status='validated_release_snapshot', rows=208, pairs=104,
        schema={f.name: str(f.type) for f in SCHEMA},
        sbn_index_convention='zero-based concept-node positions; separators excluded; gold only',
        input_sha256=inputs, parquet_sha256=sha(output),
        validation_sha256=sha(folder / 'validation.json'),
    ))
    stage.mkdir(parents=True, exist_ok=True)
    staged_hashes = {}
    for name in sorted(allowed):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, target)
        staged_hashes[name] = sha(target)
        assert staged_hashes[name] == sha(HERE / name)
    publication = json.loads((HERE / 'documentation.publication.json').read_text())
    remote = None
    if args.hub_baseline:
        remote = json.loads(args.hub_baseline.read_text())
        if remote['revision'] != publication['revision']:
            raise ValueError('Hub revision differs from the last known documentation baseline; reconcile the card before staging')
        assert all(remote['files_sha256'][name] == h for name, h in baseline.items())
        assert all(remote['files_sha256'][name] == h for name, h in publication['documentation_sha256'].items())
    write_json(stage / 'stage.manifest.json', dict(
        status='staged_locally_not_uploaded', repo_id='Shrikes/repair-sbn',
        config='later_reference', split='test', files_sha256=staged_hashes,
        last_verified_documentation_revision=publication['revision'],
        remote_head_verified=remote is not None,
        verified_parent_revision=remote['revision'] if remote else None,
        hub_baseline=remote,
        protected_existing_files_sha256=baseline,
        publication_rule='Verify current Hub parent and unchanged default splits before an explicit atomic allowlist-only upload; reject a changed parent. Stage manifest is local-only.',
    ))
    print(json.dumps(dict(status=report['status'], stage=str(stage), files=len(staged_hashes), rows=208, pairs=104, schema_fields=len(SCHEMA)), indent=2))


if __name__ == '__main__':
    main()
