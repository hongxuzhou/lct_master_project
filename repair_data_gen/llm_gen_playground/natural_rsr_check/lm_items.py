"""Normalise both sides and write the language-model requests (todo §4).

    python lm_items.py [--strict]

--strict stops if num2words is missing (the cluster run uses it).

Reads swbd/real_set.tsv, swbd/disfluencies.tsv, swbd/sentences.tsv and
synth_typed.tsv (all from build_sets.py and swbd_flatten.py), and writes

    swbd/lm_items_real.tsv     one row per typed real repair
    swbd/lm_requests_real.tsv  what score_lm.py scores for them
    lm/lm_items_synth.tsv      the same for the synthetic items
    lm/lm_requests_synth.tsv

A request is a context and a target, both plain words separated by spaces.
The model only ever reads fluent text (todo §4):

    <item>|b1_rm   context: the words before the changed core
                   target:  the reparandum core
    <item>|b1_rp   same context, target: the repair core
    <item>|a2      context: the fluent sentence up to the end of the repair
                   target:  the next word

On the real side the context is made fluent by leaving out the reparandum
and interregnum of every other disfluency in the sentence, UH words and
word fragments.  Items with MUMBLEx in a context are dropped.  Then on both
sides: contractions are joined again, there is no punctuation, digits are
written out (needs num2words; without it they stay, and the count is
printed), runs of capitals become title case, the repair's first letter is
lower case unless it is a name or "I", and the text starts with a capital.

If a core starts with a clitic ("'s"), the word before it is moved from the
context into both targets, so that the scored unit is a whole written word
("that's" against "that was").

Synthetic items are kept when the rules gave them any notation type; `type`
is that type and `given` the type the generator was asked for, so the
analysis can keep only the matches.  The items files also carry the head word of each core (lower case) for
lexical.py, and the length of the context in words.
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sets import CLITIC_FULL, NXT_TSV, head, read_tsv, write  # noqa: E402

try:
    from num2words import num2words
except ImportError:                     # on the laptop; the cluster has it
    num2words = None

HERE = Path(__file__).resolve().parent
SWBD = HERE / 'swbd'
LM = HERE / 'lm'
TYPES = {'content_word', 'constant', 'tense', 'edge_label', 'forwarding'}

CLITICS = set(CLITIC_FULL) | {"'s", "'d"}
LOWER_OK = set(
    'i he she it they we you me him her us them the a an this that these '
    'those his its their our my your there here what who which some no '
    'every each one'.split())

ITEM_COLS = ['item_id', 'side', 'type', 'given', 'note', 'retrace', 'context_len',
             'rm_core', 'rp_core', 'rm_head', 'rp_head', 'next_word', 'flags']
REQ_COLS = ['req_id', 'context', 'target']


# ------------------------------------------------------------ normalising

NUMBER = re.compile(r'(\d[\d,]*(?:\.\d+)?)(st|nd|rd|th)?')
CURRENCY = {'$': 'dollars', '€': 'euros', '£': 'pounds'}
stats = Counter()


def spell_number(m):
    digits, suffix = m.group(1).replace(',', ''), m.group(2)
    if num2words is None:
        stats['digits_left'] += 1
        return m.group(0)
    value = float(digits) if '.' in digits else int(digits)
    if suffix:
        return num2words(value, to='ordinal')
    if isinstance(value, int) and 1100 <= value <= 2099 and len(digits) == 4:
        return num2words(value, to='year')
    return num2words(value)


def spell_out(word):
    """'$500' -> 'five hundred dollars', '24-year-old' -> 'twenty-four-year-old'."""
    if not any(ch.isdigit() for ch in word):
        return [word]
    unit = []
    if word[:1] in CURRENCY:
        word, unit = word[1:], [CURRENCY[word[0]]]
    if word.endswith('%'):
        word, unit = word[:-1], ['percent']
    text = NUMBER.sub(lambda m: f' {spell_number(m)} ', word)
    return re.sub(r'\s*-\s*', '-', text).split() + unit    # 60km, 24-year-old


def fix_word(w):
    w = re.sub(r'(?<=[a-z])3e', 'é', w)          # appliqu3ed, fianc3e
    return w


def join_clitics(words):
    """['do', "n't", 'know'] -> ["don't", 'know']."""
    out = []
    for w in words:
        if out and w.lower() in CLITICS:
            out[-1] += w
        else:
            out.append(w)
    return out


def title_runs(words):
    """Capitals in titles (STAR TREK) to title case; short acronyms stay."""
    caps = [len(w) > 1 and w.isalpha() and w.isupper() for w in words]
    out = []
    for n, w in enumerate(words):
        in_run = caps[n] and ((n and caps[n - 1]) or
                              (n + 1 < len(words) and caps[n + 1]))
        out.append(w.title() if caps[n] and (in_run or len(w) > 4) else w)
    return out


def normalise(words):
    """Treebank words -> words for the language model (no capital fix)."""
    words = title_runs([fix_word(w) for w in words])
    spelled = [x for w in words for x in spell_out(w)]
    return join_clitics(spelled)


def sentence_case(text):
    return text[:1].upper() + text[1:]


def lower_first(word):
    if word[:1].isupper() and word != 'I' and word.lower() in LOWER_OK:
        return word[:1].lower() + word[1:]
    return word


# ------------------------------------------------------------- the items

def item_rows(item_id, side, r, before, rm, rp, after_ctx, rp_at, next_word,
              flags):
    """Build one items row and its requests from Treebank word lists.

    before: fluent words before the core; rm, rp: the cores; after_ctx:
    fluent words up to the end of the repair, whose repair starts at index
    rp_at; next_word: the word after it.
    """
    rm, rp, after_ctx = list(rm), list(rp), list(after_ctx)
    if rp:
        rp[0] = lower_first(rp[0])
    if rp_at < len(after_ctx):
        after_ctx[rp_at] = lower_first(after_ctx[rp_at])
    if next_word.lower() in CLITICS:
        next_word = ''
        flags.append('clitic_next')
    if (rm and rm[0].lower() in CLITICS) or (rp and rp[0].lower() in CLITICS):
        if before:
            rm, rp, before = before[-1:] + rm, before[-1:] + rp, before[:-1]
            flags.append('clitic_shift')
    ctx = normalise(before)
    rm_n, rp_n = normalise(rm), normalise(rp)
    a2_ctx = normalise(after_ctx)
    nxt = normalise([next_word]) if next_word else []

    def text(context, target):
        joined = ' '.join(context + target)
        cap = sentence_case(joined)
        n = len(' '.join(context))
        return cap[:n].strip(), cap[n:].strip()

    item = dict(item_id=item_id, side=side, type=r['outcome'],
                given=r.get('given', ''), note=r['note'],
                retrace=r['retrace'], context_len=len(ctx),
                rm_core=' '.join(rm_n), rp_core=' '.join(rp_n),
                rm_head=heads(r, 'rm'), rp_head=heads(r, 'rp'),
                next_word=' '.join(nxt), flags=','.join(flags))
    reqs = []
    if 'mumble' not in flags:
        for tag, tgt in (('b1_rm', rm_n), ('b1_rp', rp_n)):
            c, t = text(ctx, tgt)
            reqs.append(dict(req_id=f'{item_id}|{tag}', context=c, target=t))
        if nxt:
            c, t = text(a2_ctx, nxt)
            reqs.append(dict(req_id=f'{item_id}|a2', context=c, target=t))
    return item, reqs


def heads(r, side):
    words = r[f'{side}_core'].split()
    tags = r[f'{side}_core_pos'].split()
    keep = [(w, t) for w, t in zip(words, tags) if t != 'DT'] or list(
        zip(words, tags))
    if not keep:
        return ''
    ws, ts = map(list, zip(*keep))
    return ws[head(ws, ts)]


def real_items():
    sents = {s['sent_key']: s for s in read_tsv(SWBD / 'sentences.tsv', **NXT_TSV)}
    disf = read_tsv(SWBD / 'disfluencies.tsv', **NXT_TSV)
    by_id = {d['disf_id']: d for d in disf}
    by_sent = defaultdict(list)
    for d in disf:
        if d['rm_start'] != '':
            by_sent[d['sent_key']].append(d)
    for r in read_tsv(SWBD / 'real_set.tsv', **NXT_TSV):
        if r['outcome'] not in TYPES:
            continue
        d, s = by_id[r['disf_id']], sents[r['sent_key']]
        words, tags = s['words'].split(), s['pos'].split()
        rm_start, ip_end, rp_end = int(d['rm_start']), int(d['ip_end']), int(d['rp_end'])
        drop = set()
        for o in by_sent[r['sent_key']]:
            if o['disf_id'] != d['disf_id']:
                drop.update(range(int(o['rm_start']), int(o['ip_end'])))
        drop.update(n for n, (w, t) in enumerate(zip(words, tags))
                    if t == 'UH' or w.endswith('-') or w.startswith('-'))
        fluent = lambda a, b: [words[n] for n in range(a, b) if n not in drop]  # noqa: E731
        i = int(r['retrace'])
        rm = words[rm_start + i:rm_start + i + len(r['rm_core'].split())]
        rp = words[ip_end + i:ip_end + i + len(r['rp_core'].split())]
        lead = fluent(0, rm_start)
        before = lead + words[rm_start:rm_start + i]
        after = lead + words[ip_end:rp_end]
        nxt = next((words[n] for n in range(rp_end, len(words))
                    if n not in drop), '')
        flags = ['mumble'] if 'MUMBLEx' in before + after + [nxt] else []
        yield item_rows(r['disf_id'], 'real', r, before, rm, rp, after,
                        len(lead), nxt, flags)


def synth_items():
    for r in read_tsv(HERE / 'synth_typed.tsv', **NXT_TSV):
        if r['outcome'] not in TYPES:
            continue
        toks = r['tokens'].split()
        n_left, start, n_rp, i = (int(r[k]) for k in ('n_left', 'start',
                                                      'n_rp', 'retrace'))
        left, right = toks[:n_left], toks[n_left:]
        rm = left[start + i:start + i + len(r['rm_core'].split())]
        rp = right[i:i + len(r['rp_core'].split())]
        before = left[:start + i]
        after = left[:start] + right[:n_rp]
        nxt = right[n_rp] if n_rp < len(right) else ''
        item_id = f"{r['doc_id']}|{r['repair_type']}"
        yield item_rows(item_id, 'synth', r, before, rm, rp, after, start,
                        nxt, [])


def main():
    if '--strict' in sys.argv[1:] and num2words is None:
        sys.exit('num2words is not installed (--strict)')
    LM.mkdir(exist_ok=True)
    for side, rows, out in (('real', real_items(), SWBD),
                            ('synth', synth_items(), LM)):
        items, reqs = [], []
        for item, rq in rows:
            items.append(item)
            reqs.extend(rq)
        write(out / f'lm_items_{side}.tsv', items, ITEM_COLS)
        write(out / f'lm_requests_{side}.tsv', reqs, REQ_COLS)
        flags = Counter(f for it in items for f in it['flags'].split(',') if f)
        print(f'{side:5} items {len(items):5}  requests {len(reqs):5}  '
              f'types {dict(Counter(it["type"] for it in items))}  flags {dict(flags)}')
    if stats['digits_left']:
        print(f'WARNING: num2words missing, {stats["digits_left"]} numbers left '
              'as digits (fine on the laptop, not for the real run)')


if __name__ == '__main__':
    sys.exit(main())
