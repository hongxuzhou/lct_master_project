import pandas as pd, numpy as np
from scipy import stats

P = 'pilot_eval/v2_pilot_eval_run/preds_long_scored.parquet'
df = pd.read_parquet(P)
df['f1'] = df['f1'].astype(float)

# 1. drop items whose GOLD is broken (3 ids) -- they can never be scored
bad = set(df.loc[df.status=='gold_error','id'])
print(f'gold_error ids dropped: {len(bad)} -> {sorted(bad)}')
df = df[~df.id.isin(bad)].copy()
N = df.id.nunique(); print('items kept:', N)

GROUP = {'gold':'clean',
         'repair_head':'rsr_nocue','repair_mid':'rsr_nocue','repair_tail':'rsr_nocue',
         'repair_head_interrug':'rsr_cue','repair_mid_interrug':'rsr_cue','repair_tail_interrug':'rsr_cue'}
df['group'] = df.condition.map(GROUP)
df['ok']    = (df.status=='success').astype(float)
df['f1_pen']= df.f1.fillna(0.0)          # failure counts as 0

order = ['clean','rsr_nocue','rsr_cue']

print('\n================ A. THREE-GROUP SUMMARY (row-level / micro) ================')
rows=[]
for g in order:
    s = df[df.group==g]
    rows.append(dict(group=g, n_rows=len(s),
        wellformed_pct = 100*s.ok.mean(),
        ill_formed     = (s.status=='ill_formed').sum(),
        parse_error    = (s.status=='parse_error').sum(),
        f1_success     = s.f1.mean(),
        f1_penalized   = s.f1_pen.mean()))
print(pd.DataFrame(rows).round(4).to_string(index=False))

print('\n================ B. IS POOLING DEFENSIBLE? per-condition inside each group ================')
per = df[df.group!='clean'].groupby(['group','condition']).agg(
        wellformed_pct=('ok', lambda s:100*s.mean()),
        f1_success=('f1','mean'), f1_pen=('f1_pen','mean'), n=('f1_pen','size'))
print(per.round(4).to_string())
for g in ['rsr_nocue','rsr_cue']:
    sub = df[df.group==g]
    # Friedman across the 3 positions, paired by item, on penalized F1
    w = sub.pivot(index='id', columns='condition', values='f1_pen')
    st,p = stats.friedmanchisquare(*[w[c].values for c in w.columns])
    # chi2 on well-formedness
    ct = pd.crosstab(sub.condition, sub.ok)
    c2,pc,_,_ = stats.chi2_contingency(ct)
    print(f'{g}: Friedman on F1_pen across positions chi2={st:.2f} p={p:.2e} | '
          f'well-formedness chi2={c2:.2f} p={pc:.2e}')

print('\n================ C. PAIRED ITEM-LEVEL (n=833, mean over the 3 positions) ================')
item = df.pivot_table(index='id', columns='group', values=['f1_pen','f1','ok'], aggfunc='mean')
for metric,label in [('f1_pen','penalized F1'),('f1','F1 | success'),('ok','well-formed rate')]:
    m = item[metric]
    print(f'\n-- {label} --')
    print(m[order].mean().round(4).to_string())
    for g in ['rsr_nocue','rsr_cue']:
        d = (m[g]-m['clean']).dropna()
        w,p = stats.wilcoxon(d)
        dz = d.mean()/d.std()
        print(f'  {g} vs clean : delta={d.mean():+.4f}  Wilcoxon p={p:.3e}  dz={dz:.2f}  n={len(d)}')
    d = (m['rsr_cue']-m['rsr_nocue']).dropna()
    w,p = stats.wilcoxon(d); dz=d.mean()/d.std()
    print(f'  cue vs nocue : delta={d.mean():+.4f}  Wilcoxon p={p:.3e}  dz={dz:.2f}  n={len(d)}')
