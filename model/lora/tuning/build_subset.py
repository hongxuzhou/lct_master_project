"""Build a reproducible train-ID manifest from the accepted release."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import condition_rows, digest, load_rows, read_protocol, PROTOCOL_PATH


def allocate(counts, total):
    """Proportional integer quotas; largest remainder, lexical tie break."""
    size = sum(counts.values())
    quotas = {k: total * n // size for k, n in counts.items()}
    ranked = sorted(counts, key=lambda k: (-(total * counts[k] % size), k))
    for key in ranked[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def sample_ids(rows, fraction=0.1, seed=20261103):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    fluent, groups, synthetic = [], defaultdict(list), defaultdict(list)
    for row in rows:
        if row['variant'] == 'pmb_fluent':
            fluent.append(row)
        elif row['variant'] == 'fluent_split_reference' and row.get('partition_unit'):
            synthetic[row['partition_unit']].append(row)
        elif row['variant'] in ('rsr_base', 'rsr_editing_expression') and row['item_id']:
            groups[row['item_id']].append(row)
        else:
            raise ValueError("Unexpected variant or missing repair item ID")

    pools = defaultdict(lambda: defaultdict(list))
    pools['fluent']['all'] = [(r['example_id'], [r]) for r in fluent]
    for unit,members in synthetic.items():
        kind = '+'.join(sorted({r['construction_type'] for r in members}))
        pools['synthetic_fluent'][kind].append((unit,members))
    for item_id, members in groups.items():
        variants = Counter(r['variant'] for r in members)
        paired = 'rsr_editing_expression' in variants
        expected = Counter({'rsr_base': 1})
        if paired:
            expected['rsr_editing_expression'] = 1
        if variants != expected or len({r['repair_type'] for r in members}) != 1:
            raise ValueError(f"Invalid repair group: {item_id}")
        if len({r['doc_id'] for r in members}) != 1:
            raise ValueError(f"Mixed source documents: {item_id}")
        pools['paired' if paired else 'base_only'][members[0]['repair_type']].append((item_id, members))

    chosen, strata = set(), []
    for category in sorted(pools):
        counts = {k: len(v) for k, v in pools[category].items()}
        total = math.floor(sum(counts.values()) * fraction + 0.5)
        for kind, quota in allocate(counts, total).items():
            def key(group):
                return digest([seed, category, kind, group[0]]), group[0]
            for _, members in sorted(pools[category][kind], key=key)[:quota]:
                chosen.update(r['example_id'] for r in members)
            strata.append({'category': category,
                           'repair_type': None if category=='synthetic_fluent' else kind,
                           **({'construction_types':kind.split('+')} if category=='synthetic_fluent' else {}),
                           'source_groups': counts[kind], 'selected_groups': quota})
    selected = [r for r in rows if r['example_id'] in chosen]
    return selected, strata


def composition(rows):
    lengths = sorted(len(r['text'].split()) for r in rows)
    return {
        'rows': len(rows),
        'source_documents': len({r['doc_id'] for r in rows}),
        'variants': dict(sorted(Counter(r['variant'] for r in rows).items())),
        'repair_types_base': dict(sorted(Counter(r['repair_type'] for r in rows
                                                if r['variant'] == 'rsr_base').items())),
        'repair_types_marked': dict(sorted(Counter(r['repair_type'] for r in rows
                                                  if r['variant'] == 'rsr_editing_expression').items())),
        'editing_expressions': dict(sorted(Counter(r['editing_expression'] for r in rows
                                                   if r['variant'] == 'rsr_editing_expression').items())),
        'input_words': {'p50': lengths[len(lengths)//2],
                        'p90': lengths[math.floor((len(lengths)-1)*0.9)], 'max': max(lengths)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL_PATH)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--sampling-seed', type=int, default=20261103)
    args = parser.parse_args()
    protocol = read_protocol(args.protocol)
    rows = load_rows(args.dataset_dir, protocol)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from repair_data_gen.hard_negatives import holdout
    forbidden = holdout.load()
    if forbidden & {r['doc_id'] for r in rows['train']}:
        raise ValueError('Accepted train contains a hard-negative document')
    selected, strata = sample_ids(rows['train'], args.fraction, args.sampling_seed)
    if not selected or not any(r['variant'] == 'pmb_fluent' for r in selected):
        raise ValueError('Subset is too small')
    ids = [r['example_id'] for r in selected]
    manifest = {
        'schema_version': 1, 'split': 'train',
        'dataset_id': protocol['dataset']['id'],
        'dataset_revision': protocol['dataset']['revision'],
        'train_sha256': protocol['dataset']['files']['train']['sha256'],
        'fraction_requested': args.fraction, 'sampling_seed': args.sampling_seed,
        'algorithm': 'sha256 ranking within proportional largest-remainder strata; v1',
        'grouping': 'repair item_id; PMB fluent example_id; synthetic fluent partition_unit',
        'stratification': ['marked availability', 'repair_type'],
        'order': 'original train row order',
        'example_ids_sha256': digest(ids), 'example_ids': ids,
    }
    report = {
        'source': composition(rows['train']), 'subset': composition(selected),
        'fraction_actual': len(selected) / len(rows['train']), 'strata': strata,
        'checks': {'unique_ids': len(set(ids)) == len(ids),
                   'hard_negative_documents': len(forbidden & {r['doc_id'] for r in selected}),
                   'dev_documents': len({r['doc_id'] for r in selected} & {r['doc_id'] for r in rows['validation']}),
                   'complete_repair_groups': True},
        'conditions': {c: {'train_rows': len(condition_rows(selected, c)),
                           'validation_rows': len(condition_rows(rows['validation'], c)),
                           'input_digest': digest([[r['example_id'], r['text']] for r in condition_rows(selected, c)])}
                       for c in ('C1', 'C2', 'C3')},
        'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['No document-group sampling across different repair items or fluent/repair views.',
                        'Input lengths are reported, not stratified; word counts are not tokenizer lengths.'],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [('manifest.json', manifest), ('composition.json', report)]:
        path = args.output_dir / name
        text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        if path.exists() and path.read_text() != text:
            raise ValueError(f'Refusing to replace a different subset: {path}')
    for name, value in [('manifest.json', manifest), ('composition.json', report)]:
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output_dir': str(args.output_dir), 'subset': report['subset'],
                      'checks': report['checks']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
