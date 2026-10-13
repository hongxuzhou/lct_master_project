"""Flatten the NXT-format Switchboard disfluency layer into two TSV tables.

    python swbd_flatten.py <nxt_root>/xml <out_dir>

Only the layers needed for disfluency are read: terminals (words), syntax
(sentence units and INTJ/PRN nodes), disfluency, turns, and the dialogue
metadata.  All pointers are resolved to word positions inside a sentence, and
nested disfluencies become flat rows that keep a pointer to their parent.

sentences.tsv    one row per Treebank sentence unit (a <parse>)
disfluencies.tsv one row per <disfluency>, nested ones included

Every disfluency is four word offsets into its sentence's `words`:

    words[rm_start:rm_end]   reparandum
    words[rm_end:ip_end]     interregnum (may be empty)
    words[ip_end:rp_end]     repair      (may be empty)

rm_end is the interruption point.  NXT has no interregnum element: the
editing words are inside its <repair>, and nothing ever stands between the
reparandum and the repair.  The interregnum is recovered as the run of whole
INTJ or PRN constituents (or bare UH-tagged words) at the start of the NXT
repair, so "[we] you know we hire" gives interregnum "you know", repair "we".

Silences, traces and punctuation are dropped; word positions count words
only.  Treebank tokenisation is kept ("it 's", "do n't").

Rows with a problem carry it in `issues` and are kept:
  dangling      a pointer to a terminal missing from the release (the
                syntax and dialAct layers point to the same missing ids)
  cross_sent    the disfluency spans two sentence units; offsets are blank
  rm_empty      no reparandum words
  not_adjacent  reparandum, interregnum and repair do not tile one span
"""

import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

NITE = '{http://nite.sourceforge.net/}'
NID = NITE + 'id'
CHILD = NITE + 'child'
REF = re.compile(r'#id\(([^)]*)\)(?:\.\.id\(([^)]*)\))?$')
EDIT_CATS = ('INTJ', 'PRN')

SENT_COLS = ['sent_key', 'conv', 'side', 'speaker', 'turn', 'sent',
             'n_words', 'n_disf', 'words', 'pos']
DISF_COLS = ['disf_id', 'sent_key', 'parent_id', 'slot', 'depth',
             'n_children', 'rm_start', 'rm_end', 'ip_end', 'rp_end',
             'reparandum', 'interregnum', 'repair', 'ip_cats',
             'rm_pos', 'rp_pos', 'issues', 'display']


def ref_ids(href):
    m = REF.search(href)
    return m.group(1), m.group(2)


def load_speakers(xml_dir):
    """(conv, side) -> speaker id, from dialogues.xml."""
    out = {}
    root = ET.parse(xml_dir / 'corpus-resources' / 'dialogues.xml').getroot()
    for dial in root:
        conv = 'sw' + dial.get('swbdid')
        for p in dial.findall(NITE + 'pointer'):
            if p.get('role') in ('A', 'B'):
                out[conv, p.get('role')] = ref_ids(p.get('href'))[0]
    return out


class Side:
    """One speaker side of one conversation, e.g. sw2005.A."""

    def __init__(self, xml_dir, conv, side):
        self.name = f'{conv}.{side}'
        layer = lambda l: xml_dir / l / f'{self.name}.{l}.xml'

        # words, grouped by sentence unit; terminal id "s12_4" is in parse s12
        self.sents = {}                       # parse id -> [word element]
        self.where = {}                       # word id -> (parse id, offset)
        self.terminals = set()                # every terminal id, words or not
        for t in ET.parse(layer('terminals')).getroot():
            self.terminals.add(t.get(NID))
            if t.tag != 'word':
                continue
            sid = t.get(NID).split('_')[0]
            ws = self.sents.setdefault(sid, [])
            self.where[t.get(NID)] = (sid, len(ws))
            ws.append(t)

        # INTJ/PRN nodes, keyed by their first word: [(sorted offsets, cat)]
        self.edit_nodes = {}
        self.parse_order = []
        for parse in ET.parse(layer('syntax')).getroot():
            self.parse_order.append(parse.get(NID))
            for nt in parse.iter('nt'):
                if nt.get('cat') not in EDIT_CATS:
                    continue
                offs = sorted(self.where[w][1] for w in self._leaves(nt)
                              if w in self.where)
                if offs:
                    first = self.sents[parse.get(NID)][offs[0]].get(NID)
                    self.edit_nodes.setdefault(first, []).append(
                        (offs, nt.get('cat')))

        # turns point to ranges of parses: "#id(s1)..id(s4)"
        pos = {p: i for i, p in enumerate(self.parse_order)}
        self.turn = {}
        for turn in ET.parse(layer('turns')).getroot():
            for c in turn.iter(CHILD):
                a, b = ref_ids(c.get('href'))
                for p in self.parse_order[pos[a]:pos[b or a] + 1]:
                    self.turn[p] = turn.get(NID)

        self.disf_root = ET.parse(layer('disfluency')).getroot()

    @staticmethod
    def _leaves(el):
        """Terminal ids under el in document order, through nested nodes."""
        return [ref_ids(c.get('href'))[0] for c in el.iter(CHILD)]

    def split_interregnum(self, sid, rp):
        """Split NXT repair offsets into (interregnum, repair)."""
        words = self.sents[sid]
        i = 0
        cats = []
        while i < len(rp):
            nodes = [n for n in self.edit_nodes.get(words[rp[i]].get(NID), [])
                     if n[0] == rp[i:i + len(n[0])]]
            if nodes:
                offs, cat = max(nodes, key=lambda n: len(n[0]))
                i += len(offs)
                cats.append(cat)
            elif words[rp[i]].get('pos') == 'UH':
                i += 1
                cats.append('UH')
            else:
                break
        return rp[:i], rp[i:], cats

    def rows(self):
        for d in self.disf_root.findall('disfluency'):
            yield from self._visit(d, parent=None, slot='', depth=0)

    def _visit(self, d, parent, slot, depth):
        issues = []
        parts = {}
        for name in ('reparandum', 'repair'):
            ids = self._leaves(d.find(name))
            if any(w not in self.terminals for w in ids) \
                    and 'dangling' not in issues:
                issues.append('dangling')
            parts[name] = [self.where[w] for w in ids if w in self.where]
        rm_loc, rp_loc = parts['reparandum'], parts['repair']
        sids = {s for s, _ in rm_loc + rp_loc}
        children = [(c, s) for s in ('reparandum', 'repair')
                    for c in d.find(s).findall('disfluency')]

        row = dict.fromkeys(DISF_COLS, '')
        row.update(disf_id=f'{self.name}.{d.get(NID)}', parent_id=parent or '',
                   slot=slot, depth=depth, n_children=len(children))
        if not rm_loc:
            issues.append('rm_empty')
        if len(sids) > 1:
            issues.append('cross_sent')
            row['sent_key'] = f'{self.name}.{(rm_loc or rp_loc)[0][0]}'
            for name, loc in (('reparandum', rm_loc), ('repair', rp_loc)):
                row[name] = ' '.join(self.sents[s][o].get('orth')
                                     for s, o in loc)
        elif sids:
            sid = sids.pop()
            words = self.sents[sid]
            rm = sorted(o for _, o in rm_loc)
            ip, rp, cats = self.split_interregnum(sid, sorted(o for _, o in rp_loc))
            span = rm + ip + rp
            if span != list(range(span[0], span[0] + len(span))):
                issues.append('not_adjacent')
            start = rm[0] if rm else (ip + rp)[0]
            rm_end = rm[-1] + 1 if rm else start
            ip_end = ip[-1] + 1 if ip else rm_end
            rp_end = rp[-1] + 1 if rp else ip_end
            text = lambda offs, a='orth': ' '.join(words[o].get(a) for o in offs)
            row.update(sent_key=f'{self.name}.{sid}', rm_start=start,
                       rm_end=rm_end, ip_end=ip_end, rp_end=rp_end,
                       reparandum=text(rm), interregnum=text(ip),
                       repair=text(rp), ip_cats=' '.join(cats),
                       rm_pos=text(rm, 'pos'), rp_pos=text(rp, 'pos'))
            if 'not_adjacent' not in issues:
                w = [x.get('orth') for x in words]
                ipx = ['{' + ' '.join(w[rm_end:ip_end]) + '}'] if ip else []
                row['display'] = ' '.join(
                    w[:start] + ['['] + w[start:rm_end] + ['+'] + ipx
                    + w[ip_end:rp_end] + [']'] + w[rp_end:])
        row['issues'] = ','.join(issues)
        yield row
        for c, s in children:
            yield from self._visit(c, row['disf_id'], s, depth + 1)


def main(xml_dir, out_dir):
    xml_dir, out_dir = Path(xml_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    speakers = load_speakers(xml_dir)
    sides = sorted(p.name.split('.')[:2]
                   for p in (xml_dir / 'disfluency').glob('*.disfluency.xml'))

    stats = Counter()
    kw = dict(delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\',
              lineterminator='\n')
    with open(out_dir / 'sentences.tsv', 'w', newline='') as fs, \
         open(out_dir / 'disfluencies.tsv', 'w', newline='') as fd:
        ws = csv.DictWriter(fs, SENT_COLS, **kw)
        wd = csv.DictWriter(fd, DISF_COLS, **kw)
        ws.writeheader()
        wd.writeheader()
        for conv, side in sides:
            s = Side(xml_dir, conv, side)
            rows = list(s.rows())
            n_disf = Counter(r['sent_key'] for r in rows)
            for sid in s.parse_order:
                words = s.sents.get(sid, [])
                key = f'{s.name}.{sid}'
                ws.writerow(dict(
                    sent_key=key, conv=conv, side=side,
                    speaker=speakers.get((conv, side), ''),
                    turn=s.turn.get(sid, ''), sent=sid, n_words=len(words),
                    n_disf=n_disf[key],
                    words=' '.join(w.get('orth') for w in words),
                    pos=' '.join(w.get('pos') for w in words)))
            wd.writerows(rows)
            stats['sides'] += 1
            stats['sentences'] += len(s.parse_order)
            stats['disfluencies'] += len(rows)
            stats['nested'] += sum(r['depth'] > 0 for r in rows)
            stats['with_interregnum'] += sum(bool(r['interregnum']) for r in rows)
            stats['empty_repair'] += sum(
                r['rp_end'] != '' and r['rp_end'] == r['ip_end'] for r in rows)
            for r in rows:
                for i in filter(None, r['issues'].split(',')):
                    stats['issue:' + i] += 1
    for k, v in stats.items():
        print(f'{k:20} {v}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
