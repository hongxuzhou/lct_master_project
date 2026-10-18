"""Hand check of the real one-to-one items (SPEC §3), one item per page.

    python hand_check.py serve [--port 8766]
    python hand_check.py stats

Items are swbd/hand_check_1to1.tsv (select_1to1.py --sample), which is not
edited.  `serve` opens a local page that shows each sentence with its
reparandum, interregnum and repair marked, and the two sentences before
and one after it in the conversation.  Each save appends one line to
swbd/hand_check_1to1.labels.jsonl; a later save for the same item replaces
the earlier one when read.  Everything stays on this machine (127.0.0.1).

`stats` prints precision (rsr among rsr + not, unclear left out) with a
Wilson 95% interval, and the notes.
"""

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from audit import wilson  # noqa: E402
from build_sets import NXT_TSV, read_tsv  # noqa: E402

SWBD = HERE / 'swbd'
ITEMS = SWBD / 'hand_check_1to1.tsv'
LABELS = SWBD / 'hand_check_1to1.labels.jsonl'
PAGE = HERE / 'hand_check_page.html'
ANSWERS = ('rsr', 'not', 'unclear')
CONTEXT_BEFORE, CONTEXT_AFTER = 2, 1


def sent_no(key):
    return int(re.search(r'\.s(\d+)$', key).group(1))


def build_items():
    """Join the sample with the offsets, sentence words and context."""
    disf = {d['disf_id']: d for d in read_tsv(SWBD / 'disfluencies.tsv', **NXT_TSV)}
    real = {r['disf_id']: r for r in read_tsv(SWBD / 'real_1to1.tsv', **NXT_TSV)}
    sents = read_tsv(SWBD / 'sentences.tsv', **NXT_TSV)
    by_key = {s['sent_key']: s for s in sents}
    by_conv = {}
    for s in sents:
        by_conv.setdefault(s['conv'], []).append(s)
    for v in by_conv.values():
        v.sort(key=lambda s: sent_no(s['sent_key']))

    def line(s):
        return {'side': s['side'], 'text': s['words']}

    out = []
    for c in read_tsv(ITEMS, **NXT_TSV):
        d, r = disf[c['disf_id']], real[c['disf_id']]
        s = by_key[d['sent_key']]
        conv = by_conv[s['conv']]
        k = conv.index(s)
        out.append({
            'check_id': c['check_id'], 'disf_id': c['disf_id'],
            'side': s['side'], 'words': s['words'].split(),
            'rm': [int(d['rm_start']), int(d['rm_end'])],
            'ip': [int(d['rm_end']), int(d['ip_end'])],
            'rp': [int(d['ip_end']), int(d['rp_end'])],
            'cores': [int(r['rm_at']), int(r['rp_at'])],
            'rm_core': c['rm_core'], 'rp_core': c['rp_core'],
            'before': [line(x) for x in conv[max(0, k - CONTEXT_BEFORE):k]],
            'after': [line(x) for x in conv[k + 1:k + 1 + CONTEXT_AFTER]],
        })
    return out


def read_labels():
    out = {}
    if LABELS.exists():
        for ln in open(LABELS, encoding='utf-8'):
            if ln.strip():
                rec = json.loads(ln)
                out[rec['check_id']] = rec
    return out


def cmd_serve(args):
    items = build_items()
    ids = {i['check_id'] for i in items}
    page = PAGE.read_bytes()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), 'application/json')

        def do_GET(self):  # noqa: N802
            if self.path in ('/', '/index.html'):
                self._send(200, page, 'text/html; charset=utf-8')
            elif self.path == '/api/state':
                self._json({'items': items, 'labels': read_labels()})
            else:
                self._send(404, b'not found', 'text/plain')

        def do_POST(self):  # noqa: N802
            if self.path != '/api/label':
                return self._send(404, b'not found', 'text/plain')
            rec = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if rec.get('check_id') not in ids:
                return self._json({'error': 'unknown check_id'}, 400)
            if rec.get('label') not in ANSWERS + ('',):
                return self._json({'error': 'bad label'}, 400)
            rec = {'check_id': rec['check_id'], 'label': rec.get('label', ''),
                   'note': rec.get('note', '').strip(),
                   'ts': time.strftime('%Y-%m-%dT%H:%M:%S')}
            with open(LABELS, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self._json(rec)

        def log_message(self, *a):  # keep the terminal quiet
            pass

    srv = ThreadingHTTPServer(('127.0.0.1', args.port), H)
    print(f'{len(items)} items, labels -> {LABELS}')
    print(f'open http://127.0.0.1:{args.port}/   (Ctrl-C to stop)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_stats(args):
    items = read_tsv(ITEMS, **NXT_TSV)
    labels = read_labels()
    got = [labels.get(i['check_id'], {}).get('label', '') for i in items]
    y, n, u = (got.count(a) for a in ANSWERS)
    print(f'{y + n + u}/{len(items)} labelled: rsr {y}, not {n}, unclear {u}')
    if y + n:
        lo, hi = wilson(y, y + n)
        print(f'precision {y / (y + n):.0%}  95% CI [{lo:.2f}, {hi:.2f}]'
              f'  (SPEC §3: revise once if below 0.80)')
    notes = [(i, labels[i['check_id']]) for i in items
             if labels.get(i['check_id'], {}).get('note')]
    if notes:
        print('\nnotes:')
        for i, l in notes:
            print(f"  {i['check_id']:>3} {l['label'] or '-':<8}"
                  f"{i['rm_core']} -> {i['rp_core']}: {l['note']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=8766)
    sub.add_parser('stats')
    args = p.parse_args()
    {'serve': cmd_serve, 'stats': cmd_stats}[args.cmd](args)


if __name__ == '__main__':
    main()
