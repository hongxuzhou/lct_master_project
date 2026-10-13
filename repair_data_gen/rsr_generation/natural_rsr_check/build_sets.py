"""Sort real and synthetic repairs into the notation's repair types (todo §5).

    python build_sets.py [--force]

Reads swbd/disfluencies.tsv and swbd/sentences.tsv (from swbd_flatten.py) and
the synthetic run, and writes

    swbd/real_set.tsv     every NXT disfluency with its outcome and features
    swbd/hand_check.tsv   the items to check by hand (§5 step 4); never
                          overwritten unless --force, since it holds your marks
    synth_typed.tsv       every synthetic item, typed by the same rules

synth_typed.tsv also keeps the item's Treebank tokens in original case
(`tokens`: the words before the onset, then the words after it), the number
before the onset (`n_left`), where the reparandum starts (`start`) and the
repair's length (`n_rp`), so that lm_items.py can rebuild the spans.

The same `classify` runs on both sides.  Reparandum and repair are compared
as lower-cased Treebank tokens, with "a" and "an" taken as one word: the
words they share at the start (the retrace) and at the end are removed, and
the rest (the core) decides the type from its part of speech, WordNet lemma
and a few word lists.

The synthetic side has no reparandum/repair spans in the NXT sense, so they
are rebuilt from `nl_repair`, `repair_onset` and `reparandum_surface`:
  - the retrace k is the longest run of words that ends just before the
    reparandum surface and is said again at the start of the repair ("The
    gnat, the fly" -> k = 1);
  - the reparandum runs from those k words to the onset;
  - the repair is the first n words after the onset, with n chosen near the
    reparandum's length so that the two spans end on the same word (by
    lemma: "is brainwashed, has been brainwashed").  When no n does, n is the
    reparandum's length, stretched past articles, auxiliaries and multi-word
    names ("Canada, the United States").  This is an approximation.
Synthetic tags come from spaCy's en_core_web_sm (Penn Treebank tag set), run
on the two fluent readings (with the reparandum, and with the repair), since a
tagger goes wrong on the disfluent string.  The real tags are the Treebank's
own.  Nationality and language words ("Swedish", "German") are a known blind
spot: PMB writes some as names and some as concepts, so the rules call them
content words and mark them `proper` for the hand check.
"""

import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

from nltk.corpus import wordnet as wn

HERE = Path(__file__).resolve().parent
SWBD = HERE / 'swbd'
SYNTH_IN = HERE.parent / 'results' / 'meta-models_Muse-Glimmer-30B.train_full.checked.tsv'
SEED = 20261006

NXT_TSV = dict(quoting=csv.QUOTE_NONE, escapechar='\\')

MONTHS_DAYS = {
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
    'sunday', 'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december'}
NUMBER_WORDS = set(
    'zero one two three four five six seven eight nine ten eleven twelve '
    'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty '
    'thirty forty fifty sixty seventy eighty ninety hundred thousand million '
    'billion dozen half quarter first second third fourth fifth sixth seventh '
    'eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth '
    'sixteenth seventeenth eighteenth nineteenth twentieth thirtieth '
    'hundredth thousandth point noon midnight a.m p.m'.split()) | {"o'clock"}
DIGITS = re.compile(r'^[$\u20ac\u00a3]?\d[\d,.:]*(st|nd|rd|th|s|bn|m|k)?$')
PREPOSITIONS = {
    'to', 'from', 'in', 'on', 'at', 'into', 'onto', 'with', 'for', 'by',
    'about', 'of', 'over', 'under', 'through', 'after', 'before', 'around'}
NEGATION = {"n't", 'not', 'never', 'no'}
AUX_LEMMA = {
    'is': 'be', 'am': 'be', 'are': 'be', 'was': 'be', 'were': 'be',
    'been': 'be', 'being': 'be', 'be': 'be', "'s": 'be', "'m": 'be',
    "'re": 'be', 'has': 'have', 'have': 'have', 'had': 'have',
    'having': 'have', "'ve": 'have', 'does': 'do', 'did': 'do', 'do': 'do',
    'will': 'will', "'ll": 'will', 'wo': 'will', 'would': 'would',
    "'d": 'would', 'shall': 'shall', 'can': 'can', 'ca': 'can',
    'could': 'could', 'may': 'may', 'might': 'might', 'must': 'must',
    'should': 'should'}
AUX = set(AUX_LEMMA.values())
EXTEND = set(AUX_LEMMA) | {'a', 'the', 'to', 'my', 'your', 'his', 'her',
                           'its', 'our', 'their', 'this', 'that'}
FUNCTION_TAGS = {'PRP', 'PRP$', 'DT', 'PDT', 'CC', 'MD', 'WDT', 'WP', 'WP$',
                 'WRB', 'IN', 'EX', 'TO'}
CONTENT_TAGS = {'NN', 'NNS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN',
                'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'RP'}
NOMINAL_TAGS = {'PRP', 'NN', 'NNS', 'NNP', 'NNPS'}
SUBJECT_TAGS = NOMINAL_TAGS | {'DT', 'PRP$', 'JJ', 'CD'}
LIGHT_VERBS = {'have', 'make', 'get', 'do', 'be', 'go', 'take', 'give', 'come',
               'put'}
WN_POS = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}

# outcome -> what the hand check asks
QUESTIONS = {
    'content_word': 'One concept replaced by a different concept of the same '
                    'part of speech (notation §4.1)?',
    'constant': 'Same concept, different name, number or time (§4.2)?',
    'tense': 'Same verb, different past/present/future (§4.3)? A copula '
             'counts only with a noun predicate (be.v.01), not an adjective. '
             "Note 'aspect': the core changes form but not clearly the tense.",
    'edge_label': 'Same event, different preposition, PP attached to the '
                  'verb (§4.4)?',
    'forwarding': 'Different subject, verb said again (§4.5)?',
    'unplaced:longer_core': 'Does it belong to any notation type? Write the '
                            'type in comment.',
}


CLITIC_FULL = {"'re": 'are', "'m": 'am', "'ve": 'have', "'ll": 'will',
               "n't": 'not', 'ca': 'can', 'wo': 'will'}


def norm(w, t=''):
    """Comparison key: "an" is "a", and a contraction is its full form
    ("he 's" = "he is"), so that a repair that only expands or contracts
    a word counts as a repetition."""
    w = w.lower()
    if w == 'an':
        return 'a'
    if w == "'s":
        return {'BES': 'is', 'HVS': 'has', 'VBZ': 'is'}.get(t, w)
    return CLITIC_FULL.get(w, w)


def verbal(w, t):
    return t[:2] in ('VB', 'MD') or t in ('BES', 'HVS')   # Switchboard 's


def vlemmas(w):
    """Every lemma a verb form may have: "fell" -> {fell, fall}."""
    if w in AUX_LEMMA:
        return {AUX_LEMMA[w]}
    if w in CLITIC_FULL:
        return {AUX_LEMMA.get(CLITIC_FULL[w], CLITIC_FULL[w])}
    return set(wn._morphy(w, 'v')) | {w}


def lemmas(w, t):
    pos = WN_POS.get(t[:1])
    return (set(wn._morphy(w, pos)) | {w}) if pos else {w}


def coarse(t):
    """Part-of-speech class for matching; participles go with adjectives."""
    return 'JJ' if t == 'VBN' else t[:2]


def tense(words, tags):
    if {'will', "'ll", 'wo'} & set(words):
        return 'future'
    if 'VBD' in tags or {'was', 'were', 'had', 'did'} & set(words):
        return 'past'
    if ({'VBZ', 'VBP', 'BES', 'HVS'} & set(tags)
            or {'is', 'am', 'are', "'s", "'m", "'re", 'has', 'does'} & set(words)):
        return 'present'
    return None


def constantish(w, t):
    w = w[:-3] + 'y' if w.endswith('ies') else w           # sixties
    return (bool(DIGITS.match(w)) or w in MONTHS_DAYS
            or t in ('CD', 'NNP', 'NNPS')
            or all(p in NUMBER_WORDS for p in w.split('-')))


def is_constant(ws, ts):
    return bool(ws) and all(constantish(w, t) for w, t in zip(ws, ts))


def truncated(x, y, tx, ty):
    """One word is a cut-off start of the other, and not a word itself."""
    if not (x.startswith(y) or y.startswith(x)):
        return False
    short, tag = (x, tx) if len(x) <= len(y) else (y, ty)
    return tag not in FUNCTION_TAGS and not wn.synsets(short)


def head(ws, ts):
    """Index of the head of a short content core."""
    for pred in (lambda t: t.startswith('NN'), lambda t: t.startswith('VB'),
                 lambda t: t.startswith('JJ'), lambda t: t.startswith('RB')):
        idx = [i for i, t in enumerate(ts) if pred(t)]
        if idx:
            return idx[-1] if ts[idx[0]].startswith('NN') else idx[0]
    return 0


def classify(a, ap, b, bp, acap, bcap):
    """Type one repair: reparandum a and repair b, their tags and capitals."""
    out = dict(outcome='', note='', retrace='', rm_core='', rp_core='',
               rm_core_pos='', rp_core_pos='')
    a = [w.lower() for w in a]
    b = [w.lower() for w in b]
    na = [norm(w, t) for w, t in zip(a, ap)]
    nb = [norm(w, t) for w, t in zip(b, bp)]
    if not b:
        return dict(out, outcome='drop:empty_repair')
    if any(w.endswith('-') for w in a + b):
        return dict(out, outcome='drop:fragment')
    if na == nb or nb[:len(na)] == na:
        return dict(out, outcome='drop:repetition')
    i = 0
    while i < min(len(a), len(b)) and na[i] == nb[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and na[-1 - j] == nb[-1 - j]:
        j += 1
    ca, cb = a[i:len(a) - j], b[i:len(b) - j]
    pa, pb = ap[i:len(ap) - j], bp[i:len(bp) - j]
    xa, xb = acap[i:len(a) - j], bcap[i:len(b) - j]
    out.update(retrace=i, rm_core=' '.join(ca), rp_core=' '.join(cb),
               rm_core_pos=' '.join(pa), rp_core_pos=' '.join(pb))
    res = lambda o, note='': dict(out, outcome=o, note=note)  # noqa: E731

    # a tagger may read "Tom goofs" as a plural noun; with one word on each
    # side, a noun that shares a verb lemma with the verb opposite is a verb
    if (len(ca) == len(cb) == 1 and 'MD' not in (pa[0], pb[0])
            and verbal(ca[0], pa[0]) != verbal(cb[0], pb[0])
            and vlemmas(ca[0]) & vlemmas(cb[0])):
        if verbal(ca[0], pa[0]):
            pb = ['VB']
            bp = bp[:i] + pb + bp[len(bp) - j:]
        else:
            pa = ['VB']
            ap = ap[:i] + pa + ap[len(ap) - j:]

    # tense: only verb forms change, and the last verb of both spans (the
    # main verb) is the same; one core may be empty if the other holds only
    # auxiliaries ("will hurt, hurt")
    if (ca or cb) and all(verbal(w, t) or t == 'TO'
                          for w, t in zip(ca + cb, pa + pb)):
        va = [vlemmas(w) for w, t in zip(a, ap) if verbal(w, t)]
        vb = [vlemmas(w) for w, t in zip(b, bp) if verbal(w, t)]
        if va and vb and va[-1] & vb[-1]:
            ta, tb = tense(a, ap), tense(b, bp)
            notes = [] if (ta and tb and ta != tb) else ['aspect']
            notes += ['be'] if 'be' in va[-1] & vb[-1] else []
            return res('tense', ','.join(notes))
    if not ca or not cb:
        return res('drop:insertion_or_deletion')
    if len(ca) > 3 or len(cb) > 3:
        return res('drop:restart')
    if len(ca) == len(cb) == 1 and truncated(ca[0], cb[0], pa[0], pb[0]):
        return res('drop:truncated_word')
    if NEGATION & set(ca + cb):
        return res('outside:negation')

    # forwarding: a new subject, then the same verb said again, either as
    # the shared end of both spans ("Jim took off, Tom took off") or as their
    # last word in another form ("John has, I have")
    subject = lambda ts: bool(ts) and set(ts) <= SUBJECT_TAGS \
        and ts[-1] in NOMINAL_TAGS                              # noqa: E731
    if j and verbal(b[len(b) - j], bp[len(bp) - j]) \
            and subject(pa) and subject(pb):
        subj = (pa, pb)
    elif (len(a) - i > 1 and len(b) - i > 1 and a[i:-1] != b[i:-1]
            and verbal(a[-1], ap[-1]) and verbal(b[-1], bp[-1])
            and vlemmas(a[-1]) & vlemmas(b[-1])
            and subject(ap[i:-1]) and subject(bp[i:-1])):
        subj = (ap[i:-1], bp[i:-1])
    else:
        subj = None
    if subj:
        return res('forwarding',
                   'pronoun' if 'PRP' in (subj[0][-1], subj[1][-1]) else '')

    if (len(ca) == len(cb) == 1 and pa[0] == pb[0] == 'IN'
            and ca[0] in PREPOSITIONS and cb[0] in PREPOSITIONS):
        return res('edge_label')

    # determiners do not decide content or constant ("the plastic stool,
    # the chair")
    keep = lambda *xs: [x for x in zip(*xs) if x[1] != 'DT']  # noqa: E731
    ka, kb = keep(ca, pa, xa), keep(cb, pb, xb)
    det = ['det'] if (len(ka), len(kb)) != (len(ca), len(cb)) else []
    if ka and kb:
        wa, ta, ya = map(list, zip(*ka))
        wb, tb, yb = map(list, zip(*kb))
        if is_constant(wa, ta) and is_constant(wb, tb):
            return res('constant', ','.join(det))
        if set(ta) <= CONTENT_TAGS and set(tb) <= CONTENT_TAGS:
            ha, hb = head(wa, ta), head(wb, tb)
            la, lb = lemmas(wa[ha], ta[ha]), lemmas(wb[hb], tb[hb])
            if coarse(ta[ha]) == coarse(tb[hb]) and not la & lb:
                notes = ['light_verb'] if (ta[ha][:2] == 'VB'
                                           and la & LIGHT_VERBS) else []
                notes += ['multiword'] if max(len(wa), len(wb)) > 1 else []
                notes += ['proper'] if (ya[ha] or yb[hb]) else []
                return res('content_word', ','.join(notes + det))
    if len(ca) == len(cb) == 1:
        if pa[0] in FUNCTION_TAGS or pb[0] in FUNCTION_TAGS:
            return res('outside:function_word')
        return res('unplaced:1to1')
    return res('unplaced:longer_core')


def capitals(words, first_index):
    """Capitalised and not sentence-initial ("I" never counts)."""
    return [w[:1].isupper() and w != 'I' and first_index + n > 0
            for n, w in enumerate(words)]


# ---------------------------------------------------------------- real side

def read_tsv(path, **kw):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t', **kw))


def real_rows():
    sents = {s['sent_key']: s for s in read_tsv(SWBD / 'sentences.tsv', **NXT_TSV)}
    for r in read_tsv(SWBD / 'disfluencies.tsv', **NXT_TSV):
        row = dict(disf_id=r['disf_id'], sent_key=r['sent_key'])
        if r['issues']:
            c = dict(outcome='drop:issues')
        elif r['depth'] != '0':
            c = dict(outcome='drop:nested_child')
        elif r['n_children'] != '0':
            c = dict(outcome='drop:has_nested')
        else:
            a, b = r['reparandum'].split(), r['repair'].split()
            c = classify(a, r['rm_pos'].split(), b, r['rp_pos'].split(),
                         capitals(a, int(r['rm_start'])),
                         capitals(b, int(r['ip_end'])))
        s = sents.get(r['sent_key'])
        q4 = (s is not None and not r['interregnum'] and s['n_disf'] == '1'
              and 'UH' not in s['pos'].split())
        row.update(c, rm_len=len(r['reparandum'].split()),
                   rp_len=len(r['repair'].split()),
                   interregnum=r['interregnum'], q4_ok=int(q4),
                   display=r['display'])
        yield row


# ----------------------------------------------------------- synthetic side

CONTRACTION = re.compile(r"(?i)^(.+?)(n't|'s|'re|'ve|'ll|'d|'m)$")
EDGE_PUNCT = '.,?!;:"()[]'


def ptb_tokens(text):
    """Treebank-style tokens: contractions split, punctuation dropped."""
    out = []
    for w in text.replace('’', "'").split():
        w = w.strip(EDGE_PUNCT)
        if not w or not any(ch.isalnum() for ch in w):
            continue
        m = CONTRACTION.match(w)
        if m:
            out += m.groups()                       # can't -> ca n't
        else:
            out.append(w)
    return out


GIVEN = {'content_word': 'content_word', 'constant_name': 'constant',
         'constant_number': 'constant', 'constant_temporal': 'constant',
         'tense': 'tense', 'forwarding': 'forwarding'}


def loose(w):
    """Word key for matching span ends: auxiliaries and inflections merged."""
    w = norm(w)
    return AUX_LEMMA.get(w) or wn.morphy(w, 'v') or wn.morphy(w, 'n') or w


def repair_length(rm, right, k):
    """How many words after the onset make up the repair (see docstring)."""
    def tail(n):
        j = 0
        while (j < min(len(rm), n) - k
               and loose(rm[-1 - j]) == loose(right[n - 1 - j])):
            j += 1
        return j
    cands = range(max(1, len(rm) - 1), min(len(right), len(rm) + 2) + 1)
    best = max(cands, key=lambda n: (tail(n), -abs(n - len(rm))), default=0)
    if best and tail(best):
        return best
    n = min(len(rm), len(right))
    while n < len(right) and (norm(right[n - 1]) in EXTEND or (
            right[n - 1][:1].isupper() and right[n][:1].isupper())):
        n += 1
    return n


NLP = None


def tag(words):
    global NLP
    if NLP is None:                 # spaCy only here, so other scripts can
        import spacy                # import this module without it
        from spacy.tokens import Doc
        NLP = spacy.load('en_core_web_sm', disable=['parser', 'ner',
                                                     'lemmatizer'])
        tag.Doc = Doc
    return [t.tag_ for t in NLP(tag.Doc(NLP.vocab, words=words))]


def synth_rows():
    csv.field_size_limit(10 ** 9)
    for r in read_tsv(SYNTH_IN):
        if r['status'] != 'ok':
            continue
        row = dict(doc_id=r['doc_id'], repair_type=r['repair_type'],
                   given=GIVEN.get(r['repair_type'], r['repair_type']),
                   nl_repair=r['nl_repair'])
        onset = int(r['repair_onset'])
        left = ptb_tokens(r['nl_repair'][:onset])
        right = ptb_tokens(r['nl_repair'][onset:])
        ln, rn = [norm(w) for w in left], [norm(w) for w in right]
        S = [norm(w) for w in ptb_tokens(r['reparandum_surface'])]
        hits = [s for s in range(len(ln) - len(S) + 1) if ln[s:s + len(S)] == S]
        if not S or not hits or not right:
            yield dict(row, outcome='drop:surface_not_found', match=0)
            continue
        s = hits[-1]
        k = max(k for k in range(min(s, len(rn)) + 1) if ln[s - k:s] == rn[:k])
        start = s - k
        rm = left[start:]
        n = repair_length(rm, right, k)
        rp = right[:n]
        wrong = tag(left + right[n:])
        fluent = tag(left[:start] + right)
        c = classify(rm, wrong[start:len(left)], rp, fluent[start:start + n],
                     capitals(rm, start), capitals(rp, start))
        yield dict(row, **c, match=int(c['outcome'] == row['given']),
                   rm=' '.join(rm).lower(), rp=' '.join(rp).lower(),
                   tokens=' '.join(left + right), n_left=len(left),
                   start=start, n_rp=n)


# --------------------------------------------------------------- write out

def write(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore', delimiter='\t',
                           lineterminator='\n', **NXT_TSV)
        w.writeheader()
        w.writerows(rows)


REAL_COLS = ['disf_id', 'sent_key', 'outcome', 'note', 'retrace', 'rm_core',
             'rp_core', 'rm_core_pos', 'rp_core_pos', 'rm_len', 'rp_len',
             'interregnum', 'q4_ok', 'display']
SYNTH_COLS = ['doc_id', 'repair_type', 'given', 'outcome', 'match', 'note',
              'retrace', 'rm_core', 'rp_core', 'rm_core_pos', 'rp_core_pos',
              'rm', 'rp', 'nl_repair', 'tokens', 'n_left', 'start', 'n_rp']
CHECK_COLS = ['check_id', 'stratum', 'disf_id', 'note', 'rm_core', 'rp_core',
              'display', 'question', 'ok', 'comment']


def hand_check(real):
    rng = random.Random(SEED)
    by = lambda o: [r for r in real if r['outcome'] == o]   # noqa: E731
    strata = [(o, by(o)) for o in ('tense', 'constant', 'edge_label',
                                   'forwarding')]
    strata.append(('content_word', rng.sample(by('content_word'), 100)))
    strata.append(('unplaced:longer_core',
                   rng.sample(by('unplaced:longer_core'), 50)))
    rows = []
    for stratum, items in strata:
        for r in items:
            rows.append(dict(r, check_id=f'h{len(rows) + 1:03d}',
                             stratum=stratum, question=QUESTIONS[stratum],
                             ok='', comment=''))
    return rows


def main(force=False):
    real = list(real_rows())
    write(SWBD / 'real_set.tsv', real, REAL_COLS)
    synth = list(synth_rows())
    write(HERE / 'synth_typed.tsv', synth, SYNTH_COLS)

    check_path = SWBD / 'hand_check.tsv'
    if check_path.exists() and not force:
        print(f'{check_path.name} exists, left as it is (--force to redo)')
    else:
        rows = hand_check(real)
        write(check_path, rows, CHECK_COLS)
        print(f'{check_path.name}: {len(rows)} items')

    print('\nreal, all disfluencies')
    for o, n in sorted(Counter(r['outcome'] for r in real).items()):
        q4 = sum(r['q4_ok'] for r in real if r['outcome'] == o)
        print(f'  {n:6}  {o:30} q4_ok {q4}')
    print('\nsynthetic, given type -> outcome')
    for g in sorted({r['given'] for r in synth}):
        rows = [r for r in synth if r['given'] == g]
        c = Counter(r['outcome'] for r in rows)
        print(f'  {g:14} n={len(rows):5}  match {c[g] / len(rows):.0%}  '
              + ', '.join(f'{o} {n}' for o, n in c.most_common() if o != g))


if __name__ == '__main__':
    main(force='--force' in sys.argv[1:])
