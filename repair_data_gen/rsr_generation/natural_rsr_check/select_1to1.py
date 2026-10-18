"""Select one-word-for-one-word replacement repairs on both sides (SPEC §1-3).

    python select_1to1.py            # selection and funnels
    python select_1to1.py --sample   # draw the hand-check sample (once)

Reads swbd/disfluencies.tsv (swbd_flatten.py) and ../dataset/train_rsr.base.tsv,
and writes

    swbd/real_1to1.tsv          real items that pass every step
    swbd/real_1to1.funnel.tsv   rows left after each step
    lm/synth_1to1.tsv           synthetic items that pass
    lm/synth_1to1.funnel.tsv    the same, per repair type

Reparandum and repair are compared word by word in lower case, with "an"
as "a".  The retrace is their longest common prefix, the tail their longest
common suffix; the rest of each is its core.  A row drops at the first step
it fails (STEPS).  On the synthetic side the reparandum is the text removed
from nl_repair to give nl_clean, and the repair is as many words of nl_clean
from that point (continues() catches repairs longer than that window).

--sample draws 100 passing real items (seed 20261017) into
swbd/hand_check_1to1.tsv and refuses to overwrite it.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sets import NXT_TSV, ptb_tokens, read_tsv, tag, write  # noqa: E402

HERE = Path(__file__).resolve().parent
SWBD = HERE / 'swbd'
LM = HERE / 'lm'
SYNTH_IN = HERE.parent / 'dataset' / 'train_rsr.base.tsv'
SEED = 20261017
N_SAMPLE = 100

STEPS = ['issues', 'nested', 'no_repair', 'repetition', 'empty_core',
         'bad_core_word', 'long_core']
CLITICS = {"'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}

REAL_COLS = ['disf_id', 'sent_key', 'rm_core', 'rp_core', 'rm_tag', 'rp_tag',
             'rm_at', 'rp_at', 'retrace', 'interregnum', 'display']
SYNTH_COLS = ['item_id', 'repair_type', 'rm_core', 'rp_core', 'rm_tag',
              'rp_tag', 'retrace', 'before', 'nl_repair']
CHECK_COLS = ['check_id', 'disf_id', 'rm_core', 'rp_core', 'display',
              'label', 'note']


def key(w):
    w = w.lower()
    return 'a' if w == 'an' else w


def split(a, b):
    """Retrace length and the two cores of reparandum a and repair b."""
    ka, kb = [key(w) for w in a], [key(w) for w in b]
    i = 0
    while i < min(len(a), len(b)) and ka[i] == kb[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and ka[-1 - j] == kb[-1 - j]:
        j += 1
    return i, slice(i, len(a) - j), slice(i, len(b) - j)


def bad(word, t):
    return (word.startswith('-') or word.endswith('-') or t == 'UH'
            or 'mumblex' in word.lower() or word.lower() in CLITICS)


def judge(a, at, b, bt, multiword_name=lambda i: False):
    """First failed step from no_repair on, or the passing split."""
    if not b:
        return 'no_repair', None
    if [key(w) for w in a] == [key(w) for w in b]:
        return 'repetition', None
    i, ca, cb = split(a, b)
    if not a[ca] or not b[cb]:
        return 'empty_core', None
    if any(bad(w, t) for w, t in zip(a[ca] + b[cb], at[ca] + bt[cb])):
        return 'bad_core_word', None
    if len(a[ca]) > 1 or len(b[cb]) > 1 or multiword_name(i):
        return 'long_core', None
    return None, i


def funnel(counts, total):
    rows, left = [], total
    for s in STEPS:
        left -= counts[s]
        rows.append(dict(step=s, dropped=counts[s], left=left))
    return rows


def real():
    out, drops = [], Counter()
    disf = read_tsv(SWBD / 'disfluencies.tsv', **NXT_TSV)
    for r in disf:
        if r['issues']:
            drops['issues'] += 1
            continue
        if r['depth'] != '0' or r['n_children'] != '0':
            drops['nested'] += 1
            continue
        a, b = r['reparandum'].split(), r['repair'].split()
        step, i = judge(a, r['rm_pos'].split(), b, r['rp_pos'].split())
        if step:
            drops[step] += 1
            continue
        out.append(dict(
            disf_id=r['disf_id'], sent_key=r['sent_key'], rm_core=a[i],
            rp_core=b[i], rm_tag=r['rm_pos'].split()[i],
            rp_tag=r['rp_pos'].split()[i], rm_at=int(r['rm_start']) + i,
            rp_at=int(r['ip_end']) + i, retrace=i,
            interregnum=r['interregnum'], display=r['display']))
    return out, funnel(drops, len(disf))


def synth_split(r):
    """Prefix tokens, reparandum span, and the words from the repair point."""
    rep, clean, onset = r['nl_repair'], r['nl_clean'], int(r['repair_onset'])
    cut = onset - (len(rep) - len(clean))
    # a sentence-initial repair is lower case in nl_repair ("The snow, the rain")
    if rep[:cut] == clean[:cut] and rep[onset:].lower() == clean[cut:].lower():
        return (ptb_tokens(clean[:cut]),
                ptb_tokens(rep[cut:onset].rstrip(' ,')), ptb_tokens(clean[cut:]))
    # the repair spells out a contraction ("I bought, will buy" / "I'll buy"):
    # the prefix is the words nl_repair and nl_clean share
    left, full = ptb_tokens(rep[:onset]), ptb_tokens(clean)
    n = 0
    while n < min(len(left), len(full)) and key(left[n]) == key(full[n]):
        n += 1
    return left[:n], left[n:], ptb_tokens(rep[onset:])


def continues(r, right, tags, i, n):
    """Whether the repair core at right[i], the last of the n window words,
    is the start of a longer expression: a name, a modal with its verb, or
    a multi-word concept.  A shared tail inside the window ("throw up, threw
    up") already marks where the repair ends."""
    if i + 1 < n or i + 1 >= len(right):
        return False
    w, nxt = right[i], right[i + 1].lower()
    if w[:1].isupper() and right[i + 1][:1].isupper() and right[i + 1] != 'I':
        return True
    if tags[i] == 'MD' and tags[i + 1].startswith('VB'):
        return True
    lemma = r['repair_synset'].split('.')[0].split('_')
    return (r['repair_type'] in ('content_word', 'tense') and len(lemma) > 1
            and nxt.startswith(lemma[1]))


def synth():
    out, drops, total = [], Counter(), Counter()
    for r in read_tsv(SYNTH_IN):
        t = r['repair_type']
        total[t] += 1
        pre, a, right = synth_split(r)
        b = right[:len(a)]
        at = tag(pre + a + right[len(b):])[len(pre):len(pre) + len(a)]
        rt = tag(pre + right)[len(pre):]
        bt = rt[:len(b)]
        step, i = judge(a, at, b, bt, lambda i: continues(r, right, rt, i, len(b)))
        if step:
            drops[t, step] += 1
            continue
        out.append(dict(item_id=r['item_id'], repair_type=t, rm_core=a[i],
                        rp_core=b[i], rm_tag=at[i], rp_tag=bt[i], retrace=i,
                        before=' '.join(pre + a[:i]), nl_repair=r['nl_repair']))
    rows = []
    for t in sorted(total) + ['all']:
        sub = (Counter({s: n for (u, s), n in drops.items() if u == t})
               if t != 'all' else Counter({s: sum(n for (u, x), n in drops.items()
                                                  if x == s) for s in STEPS}))
        n = total[t] if t != 'all' else sum(total.values())
        rows += [dict(repair_type=t, items=n, **f) for f in funnel(sub, n)]
    return out, rows


def sample():
    path = SWBD / 'hand_check_1to1.tsv'
    if path.exists():
        sys.exit(f'{path} exists; the sample is drawn once (SPEC §3)')
    items = read_tsv(SWBD / 'real_1to1.tsv', **NXT_TSV)
    picked = random.Random(SEED).sample(items, N_SAMPLE)
    write(path, [dict(r, check_id=n, label='', note='')
                 for n, r in enumerate(picked, 1)], CHECK_COLS)
    print(f'{N_SAMPLE} items -> {path}')


def main():
    if '--sample' in sys.argv[1:]:
        return sample()
    items, steps = real()
    write(SWBD / 'real_1to1.tsv', items, REAL_COLS)
    write(SWBD / 'real_1to1.funnel.tsv', steps, ['step', 'dropped', 'left'])
    print('real:', len(items), 'items')
    for s in steps:
        print(f"  {s['step']:<14}{s['dropped']:>7}{s['left']:>7}")
    items, steps = synth()
    write(LM / 'synth_1to1.tsv', items, SYNTH_COLS)
    write(LM / 'synth_1to1.funnel.tsv', steps,
          ['repair_type', 'items', 'step', 'dropped', 'left'])
    print('synthetic:', len(items), 'items')
    for s in steps:
        if s['step'] == STEPS[-1]:
            print(f"  {s['repair_type']:<18}{s['left']:>5} / {s['items']}")


if __name__ == '__main__':
    main()
