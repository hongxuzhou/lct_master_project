"""Build isolated, authored fluent candidates. Never changes release splits."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'repair_data_gen/later_reference_challenge_v1'),
                str(ROOT / 'repair_data_gen/hard_negatives')]
from pipeline import Draft, check, _exact, parse_sbn_line, process_row
from holdout import load
from sbn_lin import read_split
from sbn_smatch import SBNGraph
from sbn_spec import SBN_EDGE_TYPE

SCENE_FIELDS = {'concept','predicate','role','past','name','colour','found',
                'object_event','profession','existential','location','actor_name','actor_role'}


def introduce(d, key, spec):
    roles = {}
    if spec.get('name'):
        roles['Name'] = json.dumps(spec['name'])
    if spec.get('profession'):
        roles['Role'] = '@'+key+'_profession'
    node(d, key, spec['concept'], **roles)
    if spec.get('profession'):
        node(d, key+'_profession', spec['profession'])
    if spec.get('colour'):
        node(d, key+'_colour', spec['colour'], ColourOf='@'+key)


def node(d, key, concept, **roles):
    d.node(key, concept, list(roles.items()))


def tense(d, key, past=False):
    node(d, key, 'time.n.08', **{'TPR' if past else 'EQU': 'now'})


def scene(d, key, concept, predicate, role='AttributeOf', past=False,
          name=None, colour=None, found=False, object_event=False, profession=None,
          existential=False, location=None, actor_name=None, actor_role='Agent'):
    if existential:
        node(d, key+'_existential_anchor', 'entity.n.01')
        node(d, key+'_predicate', 'be.v.03', EQU='@'+key+'_existential_anchor',
             Time='@'+key+'_time', Theme='@'+key, Location='@'+key+'_location')
        tense(d, key+'_time', past)
        node(d, key, concept)
        node(d, key+'_location', location)
        return
    if found or object_event:
        node(d, key+'_speaker', 'person.n.01',
             **({'Name': json.dumps(actor_name)} if actor_name else {'EQU': 'speaker'}))
        tense(d, key+'_time', True if found else past)
        node(d, key+'_event', 'find.v.01' if found else predicate,
             **{actor_role: '@'+key+'_speaker', 'Time': '@'+key+'_time',
                'Theme' if found else role: '@'+key})
        node(d, key, concept)
        if colour:
            node(d, key+'_colour', colour, ColourOf='@'+key)
    else:
        roles = {'Name': json.dumps(name)} if name else {}
        if profession:
            roles['Role'] = '@'+key+'_profession'
        node(d, key, concept, **roles)
        if profession:
            node(d, key+'_profession', profession)
        if colour:
            node(d, key+'_colour', colour, ColourOf='@'+key)
        tense(d, key+'_time', past)
        node(d, key+'_predicate', predicate, **{role: '@'+key, 'Time': '@'+key+'_time'})


def group(d, spec):
    roles = [('Sub', '@a'), ('Sub', '@b')]
    if spec.get('quantity') is not None:
        roles.append(('Quantity', str(spec['quantity'])))
    d.node('group', spec.get('group_concept', 'entity.n.01'), roles)


def build(spec):
    d = Draft()
    semantic = lambda s: {k: v for k, v in s.items() if k in SCENE_FIELDS}
    if spec.get('construction') == 'same_clause':
        introduce(d, 'a', spec['a'])
        tense(d, 'a_time', True)
        node(d, 'a_predicate', spec['relation']['concept'], **{spec['relation']['subject_role']: '@a', spec['relation']['object_role']: '@b', 'Time': '@a_time'})
        introduce(d, 'b', spec['b'])
    else:
        scene(d, 'a', **semantic(spec['a']))
        d.sep(spec.get('introduction_relation', 'CONTINUATION'), d.box)
        scene(d, 'b', **semantic(spec['b']))
    d.sep('CONTINUATION', d.box)
    tail = spec['tail']
    if tail['form'] in {'imperative', 'transitive', 'locative'}:
        actor_roles = ({'Name': json.dumps(tail['actor_name'])} if tail.get('actor_name') else
                       {'EQU': 'hearer' if tail['form']=='imperative' else 'speaker'})
        if tail.get('actor_concept'):
            actor_roles = {}
        node(d, 'tail_actor', tail.get('actor_concept', 'person.n.01'), **actor_roles)
        roles = {tail.get('actor_role', 'Agent'): '@tail_actor'}
        if tail['form'] == 'locative':
            roles['Location'] = '@tail_location'
        else:
            roles[tail.get('role', 'Theme')] = '@group'
        if tail['form'] != 'imperative':
            tense(d, 'tail_time', tail.get('past', True))
            roles['Time'] = '@tail_time'
        if tail.get('destination'):
            roles['Destination'] = '@tail_destination'
        node(d, 'tail_predicate', tail['predicate'], **roles)
        if tail['form'] == 'locative':
            node(d, 'tail_location', 'location.n.01', **{tail['spatial_relation']: '@group'})
        group(d, spec)
        if tail.get('destination'):
            node(d, 'tail_destination', tail['destination'])
    else:
        group(d, spec)
        tense(d, 'tail_time', tail.get('past', False))
        node(d, 'tail_predicate', tail['predicate'],
             **{tail.get('role', 'AttributeOf'): '@group', 'Time': '@tail_time'})
    if tail.get('explanation'):
        d.sep('EXPLANATION', d.box)
        node(d, 'group_again', 'entity.n.01', ANA='@group')
        tense(d, 'explanation_time', False)
        node(d, 'explanation_predicate', tail['explanation'],
             AttributeOf='@group_again', Time='@explanation_time')
    return d


def validate(spec, d):
    sbn = d.serialize()
    checks = check(sbn)
    errors = [k for k in ['parse', 'dag', 'index', 'magnitude', 'edges'] if checks[k] != 'ok']
    doc = parse_sbn_line(sbn)
    positions = {n['id']: i for i, n in enumerate(d.nodes())}
    expected = {(positions['group'], positions['a']), (positions['group'], positions['b'])}
    if checks['parse'] == 'ok':
        graph = SBNGraph().from_string(sbn, is_single_line=True)
        actual = {(u[1], v[1]) for u, v, edge in graph.edges(data=True)
                  if edge.get('type') == SBN_EDGE_TYPE.ROLE and edge.get('token') == 'Sub'}
        if actual != expected:
            errors.append('parsed_member_edges')
    if not all(positions[k] < positions['group'] for k in ['a', 'b']):
        errors.append('members_not_backward')
    tail = spec['tail']
    if tail['form'] == 'locative':
        loc = doc.concepts[positions['tail_location']]
        expected_link = (tail['spatial_relation'], f"{positions['group']-loc.pos:+d}")
        if expected_link not in loc.roles:
            errors.append('spatial_group_link')
    else:
        pred = doc.concepts[positions['tail_predicate']]
        role = tail.get('role', 'Theme' if tail['form'] in {'transitive','imperative'} else 'AttributeOf')
        if (role, f"{positions['group']-pred.pos:+d}") not in pred.roles:
            errors.append('continuation_group_role')
    if any(r in {'Sub', 'SubOf'} for c in doc.concepts[:positions['group']] for r, _ in c.roles):
        errors.append('preexisting_group')
    if any(s.name == 'CORRECTION' for s in doc.separators):
        errors.append('repair')
    # Ordinary CONJUNCTION is not categorically a repair. Scope-changing cases
    # require their own reviewed construction; these batches contain none.
    if any(s.name in {'NEGATION', 'POSSIBILITY', 'NECESSITY', 'CONDITION', 'CONSEQUENCE', 'PRECONDITION'}
           for s in doc.separators):
        errors.append('scope_requires_separate_review')
    target = doc.concepts[positions['group']]
    quantities = [v for r, v in target.roles if r == 'Quantity']
    if quantities != ([str(spec['quantity'])] if spec.get('quantity') is not None else []):
        errors.append('quantity')
    if bool(re.search(r'\b(both|two)\b', spec['continuation'], re.I)) != (spec.get('quantity') == 2):
        errors.append('nl_quantity')
    if not spec['sentence'].endswith(spec['continuation']):
        errors.append('continuation_surface')
    for c in doc.concepts:
        lemma, pos, number = c.synset.rsplit('.', 2)
        if _exact(lemma, pos, number) is None:
            errors.append('wordnet:'+c.synset)
    derived = process_row(sbn)
    if derived['gold_status'] != 'ok' or derived['partition'] != 'core':
        errors.append('evaluation_adapter')
    return {'failures': errors, 'checks': {k: v for k, v in checks.items() if k != 'boxes'},
            'repair_pair_check': 'not applicable to fluent', 'partition': derived['partition']}


def render(d):
    positions = {n['id']: i for i, n in enumerate(d.nodes())}
    lines = []
    for item in d.items:
        if 'separator' in item:
            lines.append(f"{item['separator']} <{item['box']-item['parent']}")
        else:
            parts = [item['concept']]
            for role, value in item['roles']:
                if str(value).startswith('@'):
                    value = f"{positions[value[1:]]-positions[item['id']]:+d}"
                parts.extend([role, str(value)])
            lines.append(' '.join(parts))
    return '\n'.join(lines)


def norm(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))



def main():
    import argparse
    from collections import defaultdict
    from difflib import SequenceMatcher
    import pyarrow.parquet as pq
    parser=argparse.ArgumentParser();parser.add_argument('--batch');args=parser.parse_args()
    sources=sorted((HERE/'source').glob('batch*.json'))
    if args.batch:sources=[HERE/'source'/f'batch{args.batch}.json']
    specs=[s for p in sources for s in json.loads(p.read_text())]
    assert len({s['id'] for s in specs})==len(specs)
    held=load();gold=read_split(ROOT/'data/pmb-5.1.0/split/en/train/gold.sbn')
    gold=[d for d in gold if d.doc_id not in held]
    pattern_index=defaultdict(list)
    for doc in gold:
        for c in doc.concepts:
            if len(pattern_index[c.synset])<8:
                pattern_index[c.synset].append({'doc_id':doc.doc_id,'sentence':doc.sentence,'sbn':doc.raw})
    rows=[];reports=[];lexicon={};readable=[]
    for s in specs:
        d=build(s);report=validate(s,d)
        rows.append({**s,'sbn':d.serialize(),'stable_nodes':d.items,'expected_members':['a','b'],
                     'expected_group':'group','split':'unassigned','language_condition':'fluent',
                     'annotation_status':'assistant_checked_candidate; human_review_pending'})
        reports.append({'id':s['id'],**report})
        readable.extend([f"## {s['id']}\n",s['sentence']+'\n','```sbn\n'+render(d)+'\n```\n'])
        for n in d.nodes():
            l,p,k=n['concept'].rsplit('.',2);ss=_exact(l,p,k)
            lexicon[n['concept']]={'wordnet_id':ss.name() if ss else None,'definition':ss.definition() if ss else None,'examples':ss.examples() if ss else [],'pmb_patterns':pattern_index[n['concept']][:2]}
    full_release={};release_hashes={}
    for split in ['train','validation']:
        p=ROOT/'repair_data_gen/training_dataset/data'/f'{split}.parquet';release_hashes[split]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in pq.read_table(p,columns=['text','example_id']).to_pylist():full_release.setdefault(norm(r['text']),[]).append([split,r['example_id']])
    challenges=[]
    for name in ['objects','names','subjects']:
        p=ROOT/'repair_data_gen/later_reference_challenge_v1'/f'{name}.json'
        challenges.extend(json.loads(p.read_text()))
    pilots=[]
    for name in ['batch01','batch02','batch03']:
        pilots.extend(json.loads((ROOT/'repair_data_gen/fluent_group_supplement_v1'/f'{name}.source.json').read_text()))
    def canonpair(s):
        out=[]
        for k in ['a','b']:
            l,p,n=s[k]['concept'].rsplit('.',2);ss=_exact(l,p,n);out.append(ss.name() if ss else s[k]['concept'])
        return tuple(sorted(out))
    flags=[]
    for r in rows:
        if norm(r['sentence']) in full_release:flags.append({'id':r['id'],'type':'release_exact','matches':full_release[norm(r['sentence'])]})
        for pool,label in [(challenges,'challenge'),(pilots,'pilot')]:
            for s in pool:
                if canonpair(r)==canonpair(s) and canonpair(r)!=('person.n.01','person.n.01'):
                    flags.append({'id':r['id'],'type':label+'_member_pair','other_id':s['id']})
                texts=[s.get('sentence',''),s.get('fluent',''),s.get('rsr','')]
                if any(norm(r['sentence'])==norm(t) for t in texts):flags.append({'id':r['id'],'type':label+'_exact','other_id':s['id']})
    duplicate=[];near=[]
    for i,r in enumerate(rows):
        for s in rows[:i]:
            if norm(r['sentence'])==norm(s['sentence']):duplicate.append([r['id'],s['id']])
            elif SequenceMatcher(None,norm(r['sentence']),norm(s['sentence'])).ratio()>=.83:near.append([r['id'],s['id']])
    negative={}
    base=next(s for s in specs if s.get('quantity')==2)
    for label,change in [('wrong_member',lambda g:g['roles'].__setitem__(0,['Sub','@a_time'])),('wrong_quantity',lambda g:g['roles'].append(['Quantity','3']))]:
        d=build(base);change(next(n for n in d.nodes() if n['id']=='group'));negative[label]=bool(validate(base,d)['failures'])
    negative['index_10']=check('entity.n.01 Theme +10 '+' '.join(['entity.n.01']*10))['magnitude']!='ok'
    for field,suffix in [('existential','_existential_anchor'),('profession','_profession'),('colour','_colour')]:
        s=next((s for s in specs if s['a'].get(field)),None)
        if s:
            d=build(s);next(n for n in d.nodes() if n['id']=='group')['roles'][0][1]='@a'+suffix
            negative[field+'_mislink']=bool(validate(s,d)['failures'])
    out=HERE/'derived'/(f'batch{args.batch}' if args.batch else 'all');out.mkdir(parents=True,exist_ok=True)
    summary={'rows':len(rows),'failed_rows':sum(bool(r['failures']) for r in reports),'failures':[r for r in reports if r['failures']],
             'overlap_flags':flags,'duplicates':duplicate,'near_pairs':near,'negative_checks':negative,'release_sha256':release_hashes,
             'construction_counts':dict(Counter(s['construction'] for s in specs)),
             'source_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'samples.md').write_text('# Fluent group candidates\n\n'+'\n'.join(readable))
    for name,obj in [('validation',reports),('summary',summary),('lexicon',lexicon)]:
        (out/f'{name}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'failed_rows':summary['failed_rows'],'failed':[{k:v for k,v in r.items() if k in {'id','failures'}} for r in reports if r['failures']],'overlap_flags':flags,'duplicates':duplicate,'near_pairs':near},ensure_ascii=False))
    if summary['failed_rows'] or duplicate or any(not v for v in negative.values()):raise SystemExit(1)

if __name__=='__main__':main()
