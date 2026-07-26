from probe_index_domain import read_records, parse, load_roles
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
roles=load_roles(f'{ROOT}/data/pmb-5.1.0/src/sbn/sbn_spec.py')
c=Counter(); ex=[]
for b in read_records(f'{ROOT}/data/pmb-5.1.0/split/en/train/gold.sbn'):
    if len(b)<3: continue
    sbn=' '.join(b[2:])
    try: concepts,edges,boxes,snap=parse(sbn,roles)
    except Exception: continue
    parent={bx[0]:bx[1] for bx in boxes}
    def anc(x):
        s=[]; seen=set()
        while x is not None and x not in seen and x in parent:
            seen.add(x); s.append(x); x=parent[x]
        return s
    for e in edges:
        if e['kind']!='concept_ptr' or e['n']<=0: continue
        t=e['src']+e['n']
        if not (0<=t<len(concepts)): c['pos_unresolvable']+=1; continue
        sb, tb = e['box'], concepts[t][2]
        c['pos_total']+=1
        if sb==tb: c['same_box']+=1
        elif sb in anc(tb): c['target_in_DESCENDANT_box']+=1
        elif tb in anc(sb):
            c['target_in_ANCESTOR_box (rule-b violation)']+=1
            if len(ex)<5: ex.append((b[0],b[1].strip(),e,concepts[t][1],sb,tb,sbn))
        else:
            c['target_in_SIBLING_box (rule-b violation)']+=1
            if len(ex)<5: ex.append((b[0],b[1].strip(),e,concepts[t][1],sb,tb,sbn))
for k,v in c.most_common(): print(f'  {k:46s} {v:6d}')
print()
for d,s,e,tok,sb,tb,sbn in ex:
    print(f'{d} | {s}')
    print(f'   {e["role"]} {e["raw"]} : box B{sb} -> B{tb}, target {tok}')
    print(f'   {sbn[:160]}\n')
