"""Export graph evidence and a manual review sheet; do not assign readings."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

from panel import HERE, digest, read_panel

sys.path.insert(0, str(HERE.parents[1]))
from formal_graph import Graph, InvalidGraph
from formal_local import scope_path
from sbn_env import sbn_src_dir


def graph_evidence(sbn):
    evidence = {'raw_correction_count': len(re.findall(r'\bCORRECTION\b', sbn)),
                'raw_conjunction_count': len(re.findall(r'\bCONJUNCTION\b', sbn)),
                'cc_pairs': None}
    try:
        g = Graph(sbn)
    except InvalidGraph as exc:
        return dict(evidence, pred_status='invalid', reason=str(exc))
    except Exception as exc:
        return dict(evidence, pred_status='reader_error', reason=f'{type(exc).__name__}: {exc}')
    var = g.var
    evidence.update(pred_status='valid', reason='',
                    cc_pairs=[[var(n) for n in pair] for pair in g.pairs()],
                    nodes=[{'id': var(n), 'label': label,
                            'written_box': var(g.box[n]) if n in g.box else None}
                           for n, label in g.labels.items()],
                    box_links=[[var(a), label, var(b)] for a, label, b in g.box_links],
                    relations=[{'source': var(a), 'role': role, 'target': var(b),
                                'written_scope': var(owner),
                                'effective_scope': var(g.condition_scope(owner))}
                               for a, role, b, owner in g.relations])
    try:
        evidence['concept_scope_paths'] = [
            {'id': var(n), 'path': [[list(step), var(box)] for step, box in scope_path(g, g.box[n])]}
            for n in g.concepts]
    except ValueError as exc:
        evidence['scope_review_reason'] = str(exc)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new review directory')
    args = p.parse_args()
    metadata = json.loads((args.run / 'run.json').read_text())
    stimuli = args.run / 'stimuli.json'
    if digest(stimuli) != metadata['stimuli_sha256']:
        raise ValueError('Run stimulus snapshot hash mismatch')
    panel, rows = read_panel(stimuli)
    predictions = args.run / 'predictions.jsonl'
    if metadata['status'] == 'complete' and digest(predictions) != metadata['predictions_sha256']:
        raise ValueError('Prediction hash mismatch')
    by_id = {}
    expected = {r['id']: r for r in rows}
    if predictions.exists():
        for line in predictions.read_text().splitlines():
            r = json.loads(line)
            key = r['id']
            if key not in expected or key in by_id or r['text'] != expected[key]['text']:
                raise ValueError('Extra, duplicate or text-mismatched prediction')
            if r['seed'] != metadata['seed'] or r['condition'] != 'C3' or r['run_id'] != metadata['checkpoint']['run_id']:
                raise ValueError('Prediction run identity mismatch')
            by_id[key] = r
    if metadata['status'] == 'complete' and len(by_id) != len(rows):
        raise ValueError('Completed run has missing predictions')
    args.output.mkdir(parents=True, exist_ok=False)
    manual = ['tea_nodes_and_status', 'coffee_nodes_and_status', 'milk_nodes_and_status',
              'ordering_event_connections', 'marker_as_content', 'unsupported_content',
              'reading', 'unresolved', 'notes']
    fields = ['id', 'seed', 'text', 'paths', 'parent', 'added', 'endpoint_paths',
              'pred_status', 'reason', 'ended_with_eos', 'tokens', 'raw_correction_count',
              'raw_conjunction_count', 'cc_pair_count', 'pred_sbn'] + manual
    counts = Counter()
    with (args.output / 'graphs.jsonl').open('x') as graphs, \
            (args.output / 'manual_review.tsv').open('x', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for row in rows:
            pred = by_id.get(row['id'])
            sbn = pred['pred_sbn'] if pred else ''
            ev = graph_evidence(sbn) if pred else dict(pred_status='missing', reason='no prediction', cc_pairs=None)
            counts[ev['pred_status']] += 1
            graphs.write(json.dumps(dict(id=row['id'], text=row['text'], pred_sbn=sbn, **ev)) + '\n')
            writer.writerow(dict(id=row['id'], seed=metadata['seed'], text=row['text'],
                                 paths='|'.join(row['paths']), parent=row['parent'], added=row['added'],
                                 endpoint_paths='|'.join(k for k, v in panel['paths'].items() if v[-1] == row['id']),
                                 pred_status=ev['pred_status'], reason=ev['reason'],
                                 ended_with_eos=pred['ended_with_eos'] if pred else '',
                                 tokens=pred['tokens'] if pred else '',
                                 raw_correction_count=ev.get('raw_correction_count'),
                                 raw_conjunction_count=ev.get('raw_conjunction_count'),
                                 cc_pair_count=len(ev['cc_pairs']) if ev['cc_pairs'] is not None else '',
                                 pred_sbn=sbn))
    summary = dict(seed=metadata['seed'], source_status=metadata['status'], total=len(rows),
                   counts=dict(counts), predictions_sha256=digest(predictions) if predictions.exists() else None,
                   reader_sha256=digest(HERE.parents[1] / 'formal_graph.py'),
                   sbn_source_hashes={name: digest(sbn_src_dir() / name) for name in
                                      ('sbn_smatch.py', 'sbn_spec.py', 'graph_base.py', 'penman_model.py')},
                   review_script_sha256=digest(Path(__file__)))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
