"""Append marked variants selected from 30% of the frozen base RSR items.

Run from any directory: python3 path/to/insert_editing_expressions.py
The frozen base is never overwritten. No fluent data is assembled here.
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'hard_negatives'))
import holdout  # noqa: E402

BASE = HERE / 'train_rsr.base.tsv'
BASE_SHA = '565ddd28c70d87d4eefa1412683ca3bbb4b62a7ddcfbfd15ae219fd2b3199ca2'
SEED = 20261024
MARKERS = ('I mean', 'no', 'no wait', 'actually')
EXTRA_COLUMNS = ('editing_expression', 'variant_id', 'nl_repair_base', 'repair_onset_base',
                 'editing_start', 'editing_end', 'selection_reason')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quotas(counts, total):
    """Proportional integer quotas, largest remainders; lexical tie-break."""
    size = sum(counts.values())
    if not 0 <= total <= size:
        raise ValueError('quota outside available pool')
    if not size:
        return dict.fromkeys(counts, 0)
    result = {k: total * n // size for k, n in counts.items()}
    order = sorted(counts, key=lambda k: (-(total * counts[k] % size), k))
    for k in order[:total - sum(result.values())]:
        result[k] += 1
    return result


def derive(row, marker=None):
    text, onset = row['nl_repair'], int(row['repair_onset'])
    if row['interregnum'] != 'none' or not 2 <= onset < len(text) or text[onset-2:onset] != ', ':
        raise ValueError(f"invalid base boundary: {row['item_id']}")
    insertion = marker + ', ' if marker else ''
    out = dict(row, editing_expression=marker or '',
               variant_id=row['item_id'] + ('.marked' if marker else '.base'),
               nl_repair_base=text, repair_onset_base=str(onset),
               nl_repair=text[:onset] + insertion + text[onset:],
               repair_onset=str(onset + len(insertion)),
               editing_start=str(onset) if marker else '',
               editing_end=str(onset + len(marker)) if marker else '',
               selection_reason=('ambiguous' if row['ambiguous'] else 'proportional_sample')
               if marker else 'unmarked')
    # Removing only the inserted segment recovers the exact base input.
    assert out['nl_repair'][:onset] + out['nl_repair'][int(out['repair_onset']):] == text
    assert out['nl_repair'][int(out['repair_onset']):] == text[onset:]
    assert out['sbn_clean'] == row['sbn_clean'] and out['sbn_repair'] == row['sbn_repair']
    del out['interregnum']
    return out


def write(name, rows, columns):
    path = HERE / name
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return dict(file=name, items=len(rows), sha256=sha(path))


def main():
    if sha(BASE) != BASE_SHA:
        raise ValueError('base differs from the frozen data card')
    held = holdout.load()
    with BASE.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        columns = [c for c in reader.fieldnames if c != 'interregnum'] + list(EXTRA_COLUMNS)
        rows = list(reader)
    assert len(rows) == len({r['item_id'] for r in rows}) == 3489
    assert not {r['doc_id'] for r in rows} & held
    required = {r['item_id'] for r in rows if r['ambiguous']}
    assert len(required) == 482
    target = (len(rows) * 30 + 50) // 100  # nearest integer, half up
    types = sorted({r['repair_type'] for r in rows})
    remaining = {t: sorted((r for r in rows if r['repair_type'] == t and not r['ambiguous']),
                            key=lambda r: r['item_id']) for t in types}
    counts = {t: len(pool) for t, pool in remaining.items()}
    allocation = quotas(counts, target - len(required))
    rng = random.Random(SEED)
    selected = set(required)
    for t in types:
        selected.update(r['item_id'] for r in rng.sample(remaining[t], allocation[t]))
    assert len(selected) == target
    # Cycle across types, shuffling rows within each type. Marker counts differ
    # by at most one both globally and within each repair type.
    marker_rng = random.Random(SEED + 1)
    markers = {}
    cursor = 0
    for t in types:
        ids = sorted(r['item_id'] for r in rows if r['repair_type'] == t and r['item_id'] in selected)
        marker_rng.shuffle(ids)
        for item_id in ids:
            markers[item_id] = MARKERS[cursor % len(MARKERS)]
            cursor += 1
    base = [derive(r) for r in rows]
    marked = [derive(r, markers[r['item_id']]) for r in rows if r['item_id'] in selected]
    augmented = base + marked
    assert len(marked) == target and required <= {r['item_id'] for r in marked}
    assert all(r['editing_expression'] in MARKERS for r in marked)
    assert len(augmented) == 4536
    assert len({r['variant_id'] for r in augmented}) == len(augmented)
    assert Counter(r['item_id'] for r in augmented) == Counter(
        {r['item_id']: 2 if r['item_id'] in selected else 1 for r in rows})
    outputs = [write('train_rsr.marked30.tsv', marked, columns),
               write('train_rsr.augmented.tsv', augmented, columns)]
    manifest = dict(
        base_file=BASE.name, base_sha256=BASE_SHA,
        script_sha256=sha(Path(__file__)), seed=SEED, marker_seed=SEED + 1,
        holdout_sha256=holdout.SHA256, holdout_overlap=0,
        selection='30% of base rows, rounded half up; all ambiguous rows first; '
                  'remaining quota proportional to non-ambiguous pool sizes by repair type; '
                  'largest remainders with lexical ties; random sample within type',
        marker_assignment='cycle through markers across types; shuffle selected IDs within type',
        markers=list(MARKERS), base_items=len(rows), marked_items=target,
        ambiguous_items=len(required), sampled_items=target-len(required),
        selected_share_of_base=target/len(rows),
        augmented_items=len(augmented), marked_share_of_augmented=target/len(augmented),
        recipe_status='append all selected marked variants; retain every base row, '
                      'including unmarked ambiguous rows; no fluent component yet',
        outputs=outputs,
        per_type={t: dict(base=sum(r['repair_type'] == t for r in rows),
                          ambiguous=sum(r['repair_type'] == t and bool(r['ambiguous']) for r in rows),
                          remaining_pool=counts[t], sampled=allocation[t],
                          marked=sum(r['repair_type'] == t for r in marked),
                          augmented=sum(r['repair_type'] == t for r in augmented),
                          markers=dict(Counter(r['editing_expression'] for r in marked
                                               if r['repair_type'] == t)))
                  for t in types})
    (HERE / 'train_rsr.editing.manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
