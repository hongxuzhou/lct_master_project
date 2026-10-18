"""Measures and report for the one-to-one items (SPEC §4-5).

    python measures_1to1.py prepare --glove G.txt --subtlex S.txt
    python score_lm.py --model GPT2_DIR --name gpt2 \
        --requests swbd/lm_req_1to1_real.tsv --out swbd/lm_1to1_real.tsv
    python score_lm.py --model GPT2_DIR --name gpt2 \
        --requests lm/lm_req_1to1_synth.tsv --out lm/lm_1to1_synth.tsv
    python measures_1to1.py report

`prepare` reads swbd/real_1to1.tsv and lm/synth_1to1.tsv (select_1to1.py)
and writes, per side, an items table (PoS, open/closed class, WordNet
relation, frequency, semantic and phonemic distance) and the language-model
requests: two per item, the same context with the reparandum core or the
repair core as target.

The context is the sentence before the core, retrace included.  On the real
side the reparandum and interregnum of every other disfluency in the
sentence, UH words and fragments are left out; on both sides clitics are
joined to their host ("do n't" -> "don't").  A sentence-initial target gets
a capital.  Digits are kept as written.

`report` joins the surprisal scores and writes report_1to1.md (aggregate
figures only, so it may be committed).
"""

import argparse
import bisect
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sets import NXT_TSV, read_tsv, write  # noqa: E402
from hand_check import read_labels  # noqa: E402
from lexical import cos_dist, levenshtein, load_glove, load_subtlex  # noqa: E402
from lm_items import join_clitics, sentence_case  # noqa: E402

HERE = Path(__file__).resolve().parent
SWBD, LM = HERE / 'swbd', HERE / 'lm'
SIDES = {'real': (SWBD / 'real_1to1.tsv', SWBD), 'synth': (LM / 'synth_1to1.tsv', LM)}
REPORT = HERE / 'report_1to1.md'

# Penn Treebank -> Universal PoS (UD conversion table; MD is AUX)
UPOS = {
    'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
    'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB',
    'VBZ': 'VERB', 'MD': 'AUX', 'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
    'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV', 'IN': 'ADP',
    'RP': 'ADP', 'TO': 'PART', 'POS': 'PART', 'CC': 'CCONJ', 'DT': 'DET',
    'PDT': 'DET', 'WDT': 'DET', 'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON',
    'WP$': 'PRON', 'EX': 'PRON', 'CD': 'NUM', 'UH': 'INTJ', 'FW': 'X',
    'LS': 'X', 'SYM': 'SYM', 'BES': 'VERB', 'HVS': 'VERB', 'XX': 'X',
    'GW': 'X', 'ADD': 'X', 'NFP': 'X', 'AFX': 'X'}
OPEN = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'INTJ'}       # UD open classes
WN_POS = {'NOUN': [wn.NOUN], 'VERB': [wn.VERB], 'ADJ': [wn.ADJ, wn.ADJ_SAT]}
RELATIONS = ['same_lemma', 'synonym', 'antonym', 'hypernym', 'hyponym',
             'co-hyponym', 'none', 'n/a']
CONTINUOUS = [('d_freq', 'Δ frequency (Zipf, repair − reparandum)'),
              ('sem_dist', 'semantic distance (GloVe cosine)'),
              ('phon_dist', 'phonemic distance (CMU phones)'),
              ('d_surprisal', 'Δ surprisal (bits, repair − reparandum)')]

ITEM_COLS = ['item_id', 'repair_type', 'rm', 'rp', 'rm_upos', 'rp_upos',
             'rm_class', 'rp_class', 'wn_rel', 'freq_rm', 'freq_rp', 'd_freq',
             'sem_dist', 'phon_dist', 'missing']


# ------------------------------------------------------------- WordNet

def synsets(word, pos):
    out = set()
    for p in WN_POS[pos]:
        lemma = wn.morphy(word, p) or word
        out.update(s for s in wn.synsets(lemma, p))
    return out


def lemmas(word, pos):
    return {wn.morphy(word, p) or word for p in WN_POS[pos]}


def wn_relation(a, ap, b, bp):
    """SPEC §4, first that holds over any sense pair."""
    if ap != bp or ap not in WN_POS:
        return 'n/a'
    sa, sb = synsets(a, ap), synsets(b, bp)
    if not sa or not sb:
        return 'n/a'
    if lemmas(a, ap) & lemmas(b, bp):
        return 'same_lemma'
    if sa & sb:
        return 'synonym'
    ants = {x.synset() for s in sa for l in s.lemmas() for x in l.antonyms()}
    if ants & sb:
        return 'antonym'
    up = lambda s: s.hypernyms() + s.instance_hypernyms()  # noqa: E731
    if any(t in set(s.closure(up)) for s in sa for t in sb):
        return 'hypernym'                       # the repair is more general
    if any(t in set(s.closure(up)) for s in sb for t in sa):
        return 'hyponym'
    if {h for s in sa for h in up(s)} & {h for s in sb for h in up(s)}:
        return 'co-hyponym'
    return 'none'


# ------------------------------------------------------------- prepare

def real_contexts():
    """item_id -> fluent words before the reparandum core."""
    sents = {s['sent_key']: s for s in read_tsv(SWBD / 'sentences.tsv', **NXT_TSV)}
    by_sent = defaultdict(list)
    disf = {}
    for d in read_tsv(SWBD / 'disfluencies.tsv', **NXT_TSV):
        disf[d['disf_id']] = d
        if d['rm_start'] != '':
            by_sent[d['sent_key']].append(d)
    out = {}
    for r in read_tsv(SWBD / 'real_1to1.tsv', **NXT_TSV):
        d, s = disf[r['disf_id']], sents[r['sent_key']]
        words, tags = s['words'].split(), s['pos'].split()
        drop = {n for n, (w, t) in enumerate(zip(words, tags))
                if t == 'UH' or w.startswith('-') or w.endswith('-')}
        for o in by_sent[r['sent_key']]:
            if o['disf_id'] != d['disf_id']:
                drop.update(range(int(o['rm_start']), int(o['ip_end'])))
        out[r['disf_id']] = [words[n] for n in range(int(r['rm_at'])) if n not in drop]
    return out


def requests(item_id, context, rm, rp):
    ctx = join_clitics(context)
    fix = (lambda w: w) if ctx else sentence_case
    if ctx:
        ctx[0] = sentence_case(ctx[0])
    return [dict(req_id=f'{item_id}|{k}', context=' '.join(ctx), target=fix(w))
            for k, w in (('rm', rm), ('rp', rp))]


def cmd_prepare(args):
    rows = {side: read_tsv(path, **NXT_TSV) for side, (path, _) in SIDES.items()}
    words = {r[k].lower() for rs in rows.values() for r in rs
             for k in ('rm_core', 'rp_core')}
    freq = load_subtlex(args.subtlex)
    glove = load_glove(args.glove, words)
    cmu = cmudict.dict()
    contexts = real_contexts()
    for side, (_, d) in SIDES.items():
        items, reqs = [], []
        for r in rows[side]:
            item_id = r['disf_id'] if side == 'real' else r['item_id']
            a, b = r['rm_core'].lower(), r['rp_core'].lower()
            ap, bp = UPOS.get(r['rm_tag'], 'X'), UPOS.get(r['rp_tag'], 'X')
            it = dict(item_id=item_id, repair_type=r.get('repair_type', ''),
                      rm=a, rp=b, rm_upos=ap, rp_upos=bp,
                      rm_class='open' if ap in OPEN else 'closed',
                      rp_class='open' if bp in OPEN else 'closed',
                      wn_rel=wn_relation(a, ap, b, bp))
            missing = [f'subtlex:{w}' for w in (a, b) if w not in freq]
            missing += [f'glove:{w}' for w in (a, b) if w not in glove]
            missing += [f'cmu:{w}' for w in (a, b) if w not in cmu]
            if a in freq and b in freq:
                it.update(freq_rm=f'{freq[a]:.4f}', freq_rp=f'{freq[b]:.4f}',
                          d_freq=f'{freq[b] - freq[a]:.4f}')
            if a in glove and b in glove:
                it['sem_dist'] = f'{cos_dist(glove[a], glove[b]):.4f}'
            if a in cmu and b in cmu:
                strip = lambda p: [x.rstrip('012') for x in p]  # noqa: E731
                it['phon_dist'] = min(levenshtein(strip(x), strip(y))
                                      for x in cmu[a] for y in cmu[b])
            it['missing'] = ','.join(missing)
            items.append(it)
            context = (contexts[item_id] if side == 'real'
                       else r['before'].split())
            reqs += requests(item_id, context, r['rm_core'], r['rp_core'])
        write(d / f'items_1to1_{side}.tsv', items, ITEM_COLS)
        write(d / f'lm_req_1to1_{side}.tsv', reqs, ['req_id', 'context', 'target'])
        print(f'{side:5} {len(items)} items, {len(reqs)} requests, '
              f'{sum(bool(i["missing"]) for i in items)} with a missing word')


# -------------------------------------------------------------- report

def cliffs_delta(xs, ys):
    """P(x > y) - P(x < y) over all pairs, x from xs and y from ys."""
    ys = sorted(ys)
    gt = sum(bisect.bisect_left(ys, x) for x in xs)
    lt = sum(len(ys) - bisect.bisect_right(ys, x) for x in xs)
    return (gt - lt) / (len(xs) * len(ys))


def load(side):
    _, d = SIDES[side]
    items = read_tsv(d / f'items_1to1_{side}.tsv', **NXT_TSV)
    lm = {r['req_id']: float(r['surprisal'])
          for r in read_tsv(d / f'lm_1to1_{side}.tsv', **NXT_TSV)}
    for i in items:
        i['d_surprisal'] = lm[f"{i['item_id']}|rp"] - lm[f"{i['item_id']}|rm"]
    return items


def values(items, key):
    return [float(i[key]) for i in items if i[key] != '']


def summary(xs):
    q = statistics.quantiles(xs, n=4)
    return f'{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]'


def shares(items, key, order=None):
    c = Counter(i[key] for i in items)
    return {k: c[k] / len(items) for k in (order or c)}


def cmd_report(args):
    real, synth = load('real'), load('synth')
    labels = read_labels()
    by_disf = {c['disf_id']: c['check_id']
               for c in read_tsv(SWBD / 'hand_check_1to1.tsv', **NXT_TSV)}
    checked = [i for i in real if by_disf.get(i['item_id']) and
               labels.get(by_disf[i['item_id']], {}).get('label') == 'rsr']
    out = ['# NXT comparison: one-to-one replacements', '',
           'Generated by `measures_1to1.py report`; rules in `SPEC.md`.', '',
           f'Items: real {len(real)}, synthetic {len(synth)}; '
           f'hand-checked real items labelled rsr: {len(checked)}.', '',
           '## Continuous measures', '',
           'Median [IQR]; n = items with both words in the resource. '
           "Cliff's δ > 0: synthetic values tend to be larger.", '',
           "| measure | real | n | synthetic | n | Cliff's δ | real, checked rsr | n |",
           '|---|---|---:|---|---:|---:|---|---:|']
    for key, name in CONTINUOUS:
        r, s, c = values(real, key), values(synth, key), values(checked, key)
        out.append(f'| {name} | {summary(r)} | {len(r)} | {summary(s)} | '
                   f'{len(s)} | {cliffs_delta(s, r):+.2f} | {summary(c)} | {len(c)} |')

    def table(title, key, order=None, both=('real', 'synthetic')):
        sr, ss, sc = (shares(x, key, order) for x in (real, synth, checked))
        keys = order or sorted(set(sr) | set(ss), key=lambda k: -sr.get(k, 0))
        rows = [f'## {title}', '', f'| | {both[0]} | {both[1]} | real, checked rsr |',
                '|---|---:|---:|---:|']
        rows += [f'| {k} | {sr.get(k, 0):.1%} | {ss.get(k, 0):.1%} | '
                 f'{sc.get(k, 0):.1%} |' for k in keys]
        return rows + ['']

    out.append('')
    out += table('Reparandum PoS (UPOS)', 'rm_upos')
    out += table('Repair PoS (UPOS)', 'rp_upos')
    for x in (real, synth, checked):
        for i in x:
            i['same_pos'] = 'same' if i['rm_upos'] == i['rp_upos'] else 'different'
    out += table('Same PoS on both sides of the repair', 'same_pos', ['same', 'different'])
    out += table('Reparandum word class', 'rm_class', ['open', 'closed'])
    out += table('WordNet relation (n/v/a pairs with the same PoS; n/a otherwise)',
                 'wn_rel', RELATIONS)
    out += ['## Synthetic items per repair type', '', '| type | items |', '|---|---:|']
    out += [f'| {t} | {n} |' for t, n in sorted(Counter(i['repair_type'] for i in synth).items())]
    REPORT.write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('prepare')
    s.add_argument('--glove', required=True, type=Path)
    s.add_argument('--subtlex', required=True, type=Path)
    sub.add_parser('report')
    args = p.parse_args()
    {'prepare': cmd_prepare, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
