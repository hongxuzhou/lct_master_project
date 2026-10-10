"""Frequency, semantic and phonemic features of reparandum and repair (B1).

    python lexical.py --glove glove.6B.300d.txt --subtlex SUBTLEXus.txt

Reads the head words from swbd/lm_items_real.tsv and lm/lm_items_synth.tsv
(lm_items.py) and writes swbd/lexical_real.tsv and lm/lexical_synth.tsv:

    freq_rm, freq_rp      Zipf frequency in SUBTLEXus, log10(SUBTLWF) + 3
    sem_dist              cosine distance of the GloVe vectors (2014, 6B,
                          300d, lower case)
    phon_dist             Levenshtein distance over CMU dictionary phones
                          (stress removed), the smallest over the words'
                          pronunciations
    phon_dist_norm        phon_dist divided by the longer pronunciation
    missing               which resource lacks which word

The CMU dictionary comes from NLTK (NLTK_DATA must point to it).  Only the
GloVe vectors of the head words are kept in memory.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

from nltk.corpus import cmudict

HERE = Path(__file__).resolve().parent
TSV = dict(delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
SIDES = (('real', HERE / 'swbd'), ('synth', HERE / 'lm'))
COLS = ['item_id', 'rm_head', 'rp_head', 'freq_rm', 'freq_rp', 'sem_dist',
        'phon_dist', 'phon_dist_norm', 'missing']


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, **TSV))


def load_subtlex(path):
    """word (lower case) -> Zipf frequency."""
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        rows = csv.reader(f, delimiter='\t')
        head = [h.strip().lower() for h in next(rows)]
        w, wf = head.index('word'), head.index('subtlwf')
        out = {}
        for r in rows:
            try:
                out.setdefault(r[w].lower(), math.log10(float(r[wf])) + 3)
            except (ValueError, IndexError):
                continue
    return out


def load_glove(path, words):
    vecs = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            word, _, rest = line.partition(' ')
            if word in words:
                vecs[word] = [float(x) for x in rest.split()]
    return vecs


def cos_dist(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu, nv = math.sqrt(sum(a * a for a in u)), math.sqrt(sum(b * b for b in v))
    return 1 - dot / (nu * nv)


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glove', required=True, type=Path)
    ap.add_argument('--subtlex', required=True, type=Path)
    args = ap.parse_args()

    items = {side: read(d / f'lm_items_{side}.tsv') for side, d in SIDES}
    heads = {r[k] for rows in items.values() for r in rows
             for k in ('rm_head', 'rp_head') if r[k]}
    freq = load_subtlex(args.subtlex)
    glove = load_glove(args.glove, heads)
    cmu = cmudict.dict()
    phones = lambda w: [[p.rstrip('012') for p in pron]  # noqa: E731
                        for pron in cmu.get(w, [])]
    print(f'{len(heads)} head words: {sum(h in freq for h in heads)} in '
          f'SUBTLEXus, {len(glove)} in GloVe, {sum(h in cmu for h in heads)} in CMU')

    for side, d in SIDES:
        out = []
        for r in items[side]:
            a, b = r['rm_head'], r['rp_head']
            row = dict(item_id=r['item_id'], rm_head=a, rp_head=b, freq_rm='',
                       freq_rp='', sem_dist='', phon_dist='', phon_dist_norm='')
            missing = []
            for key, w in (('freq_rm', a), ('freq_rp', b)):
                if w in freq:
                    row[key] = f'{freq[w]:.4f}'
                else:
                    missing.append(f'subtlex:{w}')
            if a in glove and b in glove:
                row['sem_dist'] = f'{cos_dist(glove[a], glove[b]):.4f}'
            else:
                missing += [f'glove:{w}' for w in (a, b) if w not in glove]
            pa, pb = phones(a), phones(b)
            if pa and pb:
                dist, x, y = min((levenshtein(x, y), x, y) for x in pa for y in pb)
                row['phon_dist'] = dist
                row['phon_dist_norm'] = f'{dist / max(len(x), len(y)):.4f}'
            else:
                missing += [f'cmu:{w}' for w in (a, b) if not phones(w)]
            row['missing'] = ','.join(missing)
            out.append(row)
        path = d / f'lexical_{side}.tsv'
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, COLS, lineterminator='\n', **TSV)
            w.writeheader()
            w.writerows(out)
        n_miss = sum(bool(r['missing']) for r in out)
        print(f'{side:5} {len(out)} items, {n_miss} with a missing word -> {path}')


if __name__ == '__main__':
    sys.exit(main())
