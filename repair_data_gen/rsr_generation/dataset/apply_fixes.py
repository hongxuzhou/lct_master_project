"""Surface fixes for the train RSR pass pool (preliminary freeze, 2026-10-13).

    python3 dataset/apply_fixes.py            # from rsr_generation/

Reads every row that passes all checks (as `audit.auto_pass`, plus `pin = ok`
for the runs that have it; edge label: all passing rows, selection comes
later) and writes `dataset/pool_fixed.tsv` with the corrected `nl_repair` and
`repair_onset`. The model's own text stays in `nl_repair_model` and
`repair_onset_model`; `fixes` lists what changed. Nothing in `results/` is
edited. A log of every changed row goes to `dataset/fix_log.tsv`.

What is fixed (decided by the user, 2026-10-13):
- Text outside the repair follows PMB gold character for character: quote
  marks (dialogue documents lost their quotes), final punctuation (none where
  gold has none, gold's mark otherwise), apostrophes (’ U+2019 against '
  U+0027, as gold writes them). The repaired sentence is rebuilt as
  gold[:p] + reparandum part + ", " + gold[p:], where p is where the repair
  part starts in gold.
- Where the repair part itself legitimately differs from gold (tense repairs
  write out contractions: "It was, is disrespect"), the sentence cannot be
  rebuilt from gold; only the final punctuation and the apostrophe style are
  set to gold's.
- Case of the first letter of the repair part: the model's choice, overridden
  by `dataset/onset_case_decisions.tsv` (sentence-initial reparanda, reviewed
  by a sub-agent) and ONSET_LOWER (reparandum at the start of a later
  sentence or quoted utterance).
- Hand fixes in MANUAL.
"""
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAY = HERE.parent
sys.path.insert(0, str(PLAY))
from audit import auto_pass  # noqa: E402

csv.field_size_limit(10 ** 9)
R = 'results/meta-models_Muse-Glimmer-30B.'
# (source file, extra condition) in pool order
SOURCES = [
    ('train_full.checked.tsv', 'non_edge'),
    ('r2.checked.tsv', 'non_edge'),
    ('train_forwarding_extra.checked.tsv', 'non_edge'),
    ('train_temporal_extra.checked.pin.tsv', 'pin'),
    ('train_edge.checked.tsv', None),
    ('r3_edge.checked.tsv', None),
    ('train_place.checked.tsv', None),
    ('place_r5.checked.pin.tsv', 'pin'),
    ('place_r5_rerun.checked.pin.tsv', 'pin'),
]
# (doc_id, model nl_repair) -> reparandum part to use instead
MANUAL = {
    ('p96/d3215', 'Boy scratched, He scratched his head.'): 'The boy scratched',
}
# Reparandum at the start of a later sentence or quoted utterance, which the
# sentence-initial review did not cover: lower-case the repair part's first word
ONSET_LOWER = {
    ('p02/d3410', 'Stop yelling. My leg hurts, My head hurts.'),
    ('p16/d3310', 'Whose pants are these?" "Pants are, They are Ken\'s.'),
    ('p83/d2097', 'Thing is, It is her favorite Italian word.'),
}
APOS = str.maketrans({'’': "'", '‘': "'"})


def read(p):
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def pool():
    for name, cond in SOURCES:
        for r in read(PLAY / (R + name)):
            if not auto_pass(r):
                continue
            if cond == 'non_edge' and r['repair_type'] == 'edge_label':
                continue
            if cond == 'pin' and r.get('pin') != 'ok':
                continue
            r['source_file'] = R + name
            yield r


def key(s):
    """Compare text ignoring quote marks, apostrophe style, case and the
    final punctuation."""
    s = s.replace('"', '').translate(APOS).lower().rstrip()
    return re.sub(r'[.?!]+$', '', s).rstrip()


def eq(a, b):
    return a.translate(APOS).lower() == b.translate(APOS).lower()


def gold_start(g, suf):
    """Where the repair part starts in gold. If the model's repair part
    opens with a quote mark, the quote goes with it (smallest p);
    otherwise it stays in the prefix (largest p)."""
    ps = [p for p in range(len(g)) if g[p] not in ' ' and key(g[p:]) == key(suf)
          and (p == 0 or g[p - 1] in ' "')]
    if not ps:
        return None
    return min(ps) if suf.startswith('"') else max(ps)


def strip_prefix(model_left, gold_prefix):
    """Consume gold_prefix from the start of model_left, skipping quote marks
    on either side; return the rest (the reparandum part) or None."""
    i = j = 0
    while i < len(gold_prefix):
        if j < len(model_left) and eq(gold_prefix[i], model_left[j]):
            i += 1
            j += 1
        elif gold_prefix[i] == '"':
            i += 1
        elif j < len(model_left) and model_left[j] == '"':
            j += 1
        else:
            return None
    return model_left[j:]


def final_mark(s):
    m = re.search(r'[.?!]*["”]?\s*$', s)
    return m.group(0)


def first_letter(s):
    m = re.search(r'[A-Za-z]', s)
    return m.start() if m else None


def gold_apostrophes(text, gold):
    if '’' in gold and "'" not in gold:
        return text.replace("'", '’')
    if "'" in gold and '’' not in gold:
        return text.replace('’', "'")
    return text


def main():
    decisions = {(d['doc_id'], int(d['repair_onset']), d['onset_word']): d['decision']
                 for d in read(HERE / 'onset_case_decisions.tsv')}
    rows, log, unaligned = [], [], []
    for r in pool():
        g, n, o = r['nl_clean'], r['nl_repair'], int(r['repair_onset'])
        word = (re.match(r"[\w']+", n[o:].lstrip('"')) or [''])[0]
        case = decisions.get((r['doc_id'], o, word))
        if (r['doc_id'], n) in ONSET_LOWER:
            case = 'lower'
        fixes = []
        suf = n[o:]
        p = gold_start(g, suf)
        part = strip_prefix(n[:o - 2], g[:p]) if p is not None else None
        if part is not None:
            if (r['doc_id'], n) in MANUAL:
                part = MANUAL[(r['doc_id'], n)]
                fixes.append('manual')
            part = gold_apostrophes(part, g)
            tail = g[p:]
            # first letter of the repair part: the model's case
            k, km = first_letter(tail), first_letter(suf)
            if k is not None and km is not None:
                tail = tail[:k] + suf[km] + tail[k + 1:]
            new = g[:p] + part + ', ' + tail
            new_o = p + len(part) + 2
        else:
            unaligned.append(r)
            new, new_o = n, o
            # final punctuation as gold
            fm, gm = final_mark(n), final_mark(g)
            if fm.strip() != gm.strip():
                new = new[:len(new) - len(fm)] + gm.rstrip()
            new = gold_apostrophes(new, g)
        if case is not None:
            k = first_letter(new[new_o:])
            c = new[new_o + k]
            new = new[:new_o + k] + (c.lower() if case == 'lower' else c) + new[new_o + k + 1:]
            if case == 'lower' and c != c.lower():
                fixes.append('onset_lower')
        # name what changed
        if new != n:
            if new.count('"') != n.count('"'):
                fixes.append('quotes')
            if final_mark(new).strip() != final_mark(n).strip():
                fixes.append('final_punct')
            if new.translate(APOS) != new or n.translate(APOS) != n:
                if new.count('’') != n.count('’') or new.count("'") != n.count("'"):
                    fixes.append('apostrophe')
            if not fixes:
                fixes.append('other')
        fixes = sorted(set(fixes))
        r.update(nl_repair_model=n, repair_onset_model=str(o),
                 nl_repair=new, repair_onset=str(new_o),
                 fixes=','.join(fixes), aligned='yes' if part is not None else 'no')
        rows.append(r)
        if new != n:
            log.append(dict(source_file=r['source_file'], doc_id=r['doc_id'],
                            repair_type=r['repair_type'], fixes=r['fixes'],
                            aligned=r['aligned'], nl_clean=g, nl_repair_model=n,
                            nl_repair=new))
    cols = ['source_file'] + [c for c in rows[0] if c not in
                              ('source_file', 'nl_repair_model', 'repair_onset_model',
                               'fixes', 'aligned')] + \
        ['nl_repair_model', 'repair_onset_model', 'fixes', 'aligned']
    with open(HERE / 'pool_fixed.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, cols, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with open(HERE / 'fix_log.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(log[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(log)
    print(f'pool {len(rows)}, changed {len(log)}, not aligned to gold {len(unaligned)}')


if __name__ == '__main__':
    main()
