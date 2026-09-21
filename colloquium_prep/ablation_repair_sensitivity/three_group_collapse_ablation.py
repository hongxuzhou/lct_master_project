import pandas as pd, numpy as np
from scipy import stats
a = pd.read_parquet('ablation_repair_sensitivity/full_ablation_out_scored.parquet')
bad = {'p04/d2024','p05/d0919','p69/d1719'}
a = a[~a.id.isin(bad)].copy()
GROUP = {'gold':'clean',
         'repair_head':'rsr_nocue','repair_mid':'rsr_nocue','repair_tail':'rsr_nocue',
         'repair_head_interrug':'rsr_cue','repair_mid_interrug':'rsr_cue','repair_tail_interrug':'rsr_cue'}
a['group']=a.condition.map(GROUP)
print('items:',a.id.nunique(),' rows:',len(a))
print('\n=== ABLATION: base Gemma-2-9B-it, zero-shot repair DETECTION (plain NL, no SBN) ===')
rows=[]
for g in ['clean','rsr_nocue','rsr_cue']:
    s=a[a.group==g]
    rows.append(dict(group=g,n=len(s),
        gt_positive=s.gt_positive.mean(),
        detected_pct=100*s.detected.mean(),
        malformed_pct=100*s.malformed.mean(),
        reparandum_f1=s.reparandum_f1.mean(),
        repair_f1=s.repair_f1.mean(),
        interreg_f1=s.interregnum_f1.mean()))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print('\n--- per condition (pooling check) ---')
print(a.groupby(['group','condition']).agg(det=('detected',lambda s:100*s.mean()),
     rep_f1=('reparandum_f1','mean'), rpr_f1=('repair_f1','mean'), n=('detected','size')).round(4).to_string())
print('\n--- paired item-level detection rate ---')
it=a.pivot_table(index='id',columns='group',values='detected',aggfunc='mean')
print(it[['clean','rsr_nocue','rsr_cue']].mean().round(4).to_string())
for g in ['rsr_nocue','rsr_cue']:
    d=(it[g]-it['clean']).dropna(); w,p=stats.wilcoxon(d)
    print(f'  {g} vs clean: +{d.mean():.4f} p={p:.2e}')
d=(it['rsr_cue']-it['rsr_nocue']).dropna(); w,p=stats.wilcoxon(d)
print(f'  cue vs nocue: {d.mean():+.4f} p={p:.2e}')
# detection conditional on scoring: F1 among detected only
print('\n--- span F1 among DETECTED rows only ---')
det=a[a.detected & (a.group!="clean")]
print(det.groupby('group')[['reparandum_f1','repair_f1','interregnum_f1']].agg(['mean','count']).round(4).to_string())
