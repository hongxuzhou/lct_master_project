"""Accepted fluent split-reference rows and shared provenance fields."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANT = 'fluent_split_reference'
FLUENT_VARIANTS = frozenset({'pmb_fluent', VARIANT})
RSR_VARIANTS = frozenset({'rsr_base', 'rsr_editing_expression'})


def provenance(row):
    """Add provenance without changing any pre-existing field."""
    variant = row['variant']
    if variant == 'pmb_fluent':
        origin = 'pmb'
    elif variant in RSR_VARIANTS:
        origin = 'synthetic_rsr'
    else:
        raise ValueError('Unexpected baseline variant: '+variant)
    family = 'pmb::'+row['doc_id']
    return {**row, 'data_origin': origin, 'construction_type': None,
            'family_id': family, 'partition_unit': family}


def accepted_rows(split, schema):
    metadata = json.loads((HERE/'fluent_group.acceptance.json').read_text())
    if metadata['status'] != 'user_reviewed_and_authorised_for_integration':
        raise ValueError('Fluent supplement has not been accepted')
    source_split = 'dev' if split == 'validation' else split
    spec = metadata['files'][source_split]
    path = ROOT/spec['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Accepted fluent source changed: '+str(path))
    sources = [json.loads(line) for line in path.read_text().splitlines()]
    if len(sources) != spec['rows']:
        raise ValueError('Unexpected fluent supplement count')
    rows = []
    for source in sources:
        if source['proposed_split'] != source_split+'_candidate':
            raise ValueError('Accepted supplement split mismatch')
        identity = 'fluent_group_v1::'+source['id']
        row = dict.fromkeys(schema.names)
        row.update(example_id=identity, item_id=identity, doc_id=identity,
                   variant=VARIANT, text=source['sentence'],
                   sbn_clean=source['sbn'], sbn_repair=source['sbn'],
                   label_source='user_reviewed_synthetic',
                   source='synthetic fluent split-reference; fluent_group_synthesis_v1',
                   data_origin='synthetic_fluent', construction_type=source['construction'],
                   family_id='fluent_group_v1::'+source['family_id'],
                   partition_unit='fluent_group_v1::'+source['partition_unit'])
        rows.append(row)
    return rows
