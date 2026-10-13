"""Assemble the base train RSR set from the fixed pool (preliminary freeze).

    python3 freeze/apply_fixes.py      # first: writes freeze/pool_fixed.tsv
    python3 freeze/assemble.py         # writes train_rsr.tsv + manifest

Decided by the user, 2026-10-13:
- edge label: the 495 items labelled natural (train_edge and r3_edge
  `natural = yes`; `audit/train_place/selection.tsv` `final = yes`, a-rows
  from `audit/train_place`, b-rows from `audit/train_place_rerun`);
- every other type: simple random sample of 500 with a fixed seed, so each
  type keeps the pool's internal proportions (no balancing within a type);
- forwarding keeps the items whose predicate has a reflexive bound to the
  new subject.
No interregnum, no marked/base assignment: this is the base set.

Every row is checked again here (all automatic checks, `pin`, the repair-part
check, and the text outside the repair against gold); the script stops if any
row fails. The manifest records the seed, sha256, counts, distinct source
documents and, per type, the pool and sample proportions of one property.
"""
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAY = HERE.parent
sys.path.insert(0, str(PLAY))
from audit import auto_pass, read_labels  # noqa: E402
from check_repair_part import problems  # noqa: E402

csv.field_size_limit(10 ** 9)
SEED = 20261013
QUOTA = 500
TYPES = ['content_word', 'constant_name', 'constant_number', 'constant_temporal',
         'tense', 'forwarding', 'edge_label']
OUT = HERE / 'train_rsr.tsv'
MANIFEST = HERE / 'train_rsr_manifest.json'


def read(p):
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def edge_selection(pool):
    """The 495 edge-label rows labelled natural, with their label source."""
    idx = {}
    for r in pool:
        if r['repair_type'] == 'edge_label':
            k = (r['doc_id'], r['nl_repair_model'])
            assert k not in idx, k
            idx[k] = r
    picked = []

    def take(items, ids, source):
        it = {i['audit_id']: i for i in read(PLAY / items)}
        for a, src in ids:
            i = it[a]
            r = idx[(i['doc_id'], i['nl_repair'])]   # KeyError = unmapped label
            picked.append(dict(r, label_source=src, audit_ref=f'{Path(items).parent.name}/{a}'))

    for name in ('r3_edge', 'train_edge'):
        lab = read_labels(PLAY / 'audit' / name)
        take(f'audit/{name}/items.tsv',
             [(a, 'human') for a, l in sorted(lab.items()) if l.get('natural') == 'yes'], source=name)
    sel = [s for s in read(PLAY / 'audit/train_place/selection.tsv') if s['final'] == 'yes']
    take('audit/train_place/items.tsv',
         [(s['audit_id'], s['final_source']) for s in sel if s['audit_id'].startswith('a')], 'place')
    take('audit/train_place_rerun/items.tsv',
         [(s['audit_id'], s['final_source']) for s in sel if s['audit_id'].startswith('b')], 'rerun')
    assert len({id(p) for p in picked}) == len(picked)
    return picked


def final_mark(s):
    return re.search(r'[.?!]*["”]?\s*$', s).group(0).strip()


def check(r):
    """Problems of one row; empty list = passes."""
    out = []
    if not auto_pass(r):
        out.append('auto_pass')
    if 'pin' in r and r['pin'] not in ('', 'ok'):
        out.append('pin')
    out += problems(r)
    g, n, o = r['nl_clean'], r['nl_repair'], int(r['repair_onset'])
    if final_mark(n) != final_mark(g):
        out.append('final mark differs from gold')
    if r['aligned'] == 'yes':
        tail, p = n[o:], len(g) - len(n[o:])
        if n[:p] != g[:p] or tail[1:] != g[p + 1:] or tail[:1].lower() != g[p:p + 1].lower():
            out.append('text outside the repair differs from gold')
    return out


def prop(r):
    """One property per type, to compare pool and sample proportions."""
    t = r['repair_type']
    if t == 'content_word':
        return 'pos=' + r['repair_synset'].split('.')[-2]
    if t == 'constant_temporal':
        m = re.search(r'ClockTime|YearOfCentury|MonthOfYear|DayOfMonth|DayOfWeek', r['sbn_repair'])
        return 'role=' + (m.group(0) if m else '?')
    if t == 'tense':
        return 'reparandum=' + r['reparandum_value']
    if t == 'edge_label':
        return 'label_source=' + r['label_source']
    return 'ambiguous=' + (r['ambiguous'] or 'none')


def main():
    pool = read(HERE / 'pool_fixed.tsv')
    rng = random.Random(SEED)
    rows = []
    for t in TYPES:
        if t == 'edge_label':
            rows += edge_selection(pool)
            continue
        cand = sorted((r for r in pool if r['repair_type'] == t),
                      key=lambda r: (r['source_file'], r['doc_id'], r['nl_repair_model']))
        for r in rng.sample(cand, QUOTA):
            rows.append(dict(r, label_source='auto_pass', audit_ref=''))
    rows.sort(key=lambda r: (TYPES.index(r['repair_type']), r['doc_id'], r['nl_repair']))
    for i, r in enumerate(rows):
        r['item_id'] = f'rsr_train_{i:04d}'

    failed = {r['item_id']: p for r in rows if (p := check(r))}
    dup = [k for k, v in Counter((r['doc_id'], r['nl_repair']) for r in rows).items() if v > 1]
    if failed or dup:
        sys.exit(f'not frozen: {len(failed)} rows fail checks {list(failed.items())[:5]}, '
                 f'{len(dup)} duplicates')

    cols = ['item_id', 'repair_type', 'doc_id', 'nl_clean', 'nl_repair', 'repair_onset',
            'sbn_clean', 'sbn_repair', 'reparandum_surface', 'reparandum_synset',
            'reparandum_value', 'repair_synset', 'site_index', 'devices', 'strategy',
            'wn_relation', 'ambiguous', 'interregnum', 'label_source', 'audit_ref',
            'source_file', 'nl_repair_model', 'repair_onset_model', 'fixes', 'aligned']
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, cols, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    per_type = {}
    for t in TYPES:
        s = [r for r in rows if r['repair_type'] == t]
        p = [r for r in pool if r['repair_type'] == t]
        docs = Counter(r['doc_id'] for r in s)
        ps = Counter(prop(r) for r in s)
        pp = Counter(prop(r) for r in p) if t != 'edge_label' else None
        per_type[t] = dict(
            items=len(s), pool=len(p), distinct_docs=len(docs),
            docs_with_2_items=sum(v == 2 for v in docs.values()),
            source_files=dict(Counter(r['source_file'].split('Glimmer-30B.')[1] for r in s)),
            fixes=dict(Counter(f for r in s for f in r['fixes'].split(',') if f)),
            proportions={k: dict(sample=round(v / len(s), 3),
                                 pool=round(pp[k] / len(p), 3) if pp else None)
                         for k, v in sorted(ps.items(), key=lambda kv: -kv[1])})
    manifest = dict(
        file=OUT.name, sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
        pool_file='pool_fixed.tsv',
        pool_sha256=hashlib.sha256((HERE / 'pool_fixed.tsv').read_bytes()).hexdigest(),
        seed=SEED, quota=QUOTA, items=len(rows),
        distinct_docs=len({r['doc_id'] for r in rows}),
        sampling='simple random per type (fixed seed); edge label = all 495 labelled natural',
        per_type=per_type)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
    print(f'{len(rows)} items, {manifest["distinct_docs"]} documents, all checks pass')
    for t, d in per_type.items():
        print(f'  {t:18} {d["items"]:4} of {d["pool"]:5}  docs {d["distinct_docs"]:4}'
              f'  (2 items: {d["docs_with_2_items"]})')


if __name__ == '__main__':
    main()
