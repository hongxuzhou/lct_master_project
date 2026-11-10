"""Analyse the frozen cloud snapshot without contacting W&B."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
LORA = OUT.parents[2]
snapshot = json.loads((OUT / 'runs.json').read_text())
histories = json.loads((OUT / 'histories.json').read_text())
spec_path = LORA / 'tuning/configs/lr-rank-v2.json'
spec = json.loads(spec_path.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected_hashes = {
    'grid_sha256': digest(spec_path),
    'protocol_sha256': digest(LORA / spec['protocol']),
    'subset_manifest_sha256': digest(LORA / spec['subset']),
    'sweep_entry_sha256': digest(LORA / 'tuning/run_sweep.py'),
    'plan_code_sha256': digest(LORA / 'tuning/plans.py'),
}
groups = defaultdict(list)
flat = []
for r in snapshot['runs']:
    c, s = r['config'], r['summary']
    if c.get('tuning_plan') != spec['name']:
        continue
    issues = []
    if r['state'] != 'finished' or s.get('status') != 'development_passed':
        issues.append('not_finished_and_passed')
    loss, reload = s.get('best_eval_loss'), s.get('reload_eval_loss')
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (loss, reload)) or not math.isclose(loss, reload, rel_tol=1e-5, abs_tol=1e-5):
        issues.append('reload_check')
    for k, v in expected_hashes.items():
        if c.get(k) != v:
            issues.append(k)
    code = c.get('code_sha256', {})
    for name in ('train.py', 'data.py'):
        if code.get(name) != digest(LORA / name):
            issues.append(name)
    if code.get(Path(spec['protocol']).name) != expected_hashes['protocol_sha256']:
        issues.append('protocol_code_hash')
    if c.get('subset', {}).get('manifest_sha256') != expected_hashes['subset_manifest_sha256']:
        issues.append('subset_hash')
    candidate = c.get('candidate')
    if candidate not in spec['candidates'] or [c.get('learning_rate'), c.get('lora_rank')] != spec['candidates'].get(candidate):
        issues.append('candidate_assignment')
    if c.get('seed') != 17 or c.get('lora_alpha') != 2*c.get('lora_rank', 0):
        issues.append('seed_or_alpha')
    condition = c.get('condition')
    steps = 183 if condition == 'C1' else 267
    if s.get('steps') != steps:
        issues.append('step_budget')
    ev = [h for h in histories.get(r['id'], []) if isinstance(h.get('eval_loss'), (int, float))]
    if ev and loss is not None and not math.isclose(min(h['eval_loss'] for h in ev), loss, rel_tol=1e-6, abs_tol=1e-6):
        issues.append('history_minimum')
    row = dict(id=r['id'], model_size=c.get('model_size'), condition=condition,
               candidate=candidate, learning_rate=c.get('learning_rate'), rank=c.get('lora_rank'),
               state=r['state'], status=s.get('status'), best_eval_loss=loss,
               reload_eval_loss=reload, best_step=s.get('best_step'), steps=s.get('steps'),
               final_eval_loss=ev[-1]['eval_loss'] if ev else None,
               seconds=s.get('seconds'), train_seconds=s.get('train_seconds'),
               peak_allocated_gib=s.get('peak_allocated_gib'), gpu=c.get('gpu'),
               cluster=c.get('cluster'), sweep=r['sweep'], url=r['url'], issues=issues)
    flat.append(row)
    groups[f"{row['model_size']}/{condition}"].append(row)

for size in ('2B', '9B', '27B'):
    for condition in ('C1', 'C2', 'C3'):
        groups.setdefault(f'{size}/{condition}', [])
results = {}
for key, rows in sorted(groups.items()):
    good = sorted([r for r in rows if not r['issues']], key=lambda x:x['best_eval_loss'])
    counts = Counter(r['candidate'] for r in rows)
    complete = len(rows) == 9 and set(counts) == set(spec['candidates']) and all(v == 1 for v in counts.values()) and len(good) == 9
    results[key] = dict(complete=complete, states=dict(Counter(r['state'] for r in rows)),
                        missing=sorted(set(spec['candidates'])-set(counts)),
                        selected_run=good[0]['id'] if complete else None,
                        ranked_successful=good, all_runs=rows)
(OUT / 'analysis.json').write_text(json.dumps({'fetched_at':snapshot['fetched_at'],
    'expected_hashes':expected_hashes, 'groups':results}, indent=2))
with (OUT / 'trials.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
for key, g in results.items():
    print(key, 'complete=',g['complete'], 'states=',g['states'], 'missing=',g['missing'])
    for r in g['all_runs']:
        if r['issues']:
            print(' ISSUES',r['id'],r['issues'])
    for r in g['ranked_successful']:
        print(' ',r['candidate'],r['id'],round(r['best_eval_loss'],8), 'best_step',r['best_step'],'final',r['final_eval_loss'],'sec',round(r['seconds']), 'GiB',round(r['peak_allocated_gib'],2))

# Every colour scale is local to one size/condition; losses are not pooled.
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/repairsbn-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
for i, size in enumerate(('2B', '9B')):
    for j, condition in enumerate(('C1', 'C2', 'C3')):
        ax = axes[i,j]
        g = results[f'{size}/{condition}']
        matrix = np.full((3,4), np.nan)
        for r in g['ranked_successful']:
            matrix[(16,32,64).index(r['rank']), (0.0001,0.0002,0.0003,0.0004).index(r['learning_rate'])] = r['best_eval_loss']
        ax.imshow(matrix, cmap='YlGnBu_r', aspect='auto')
        best = g['ranked_successful'][0]
        for y in range(3):
            for x in range(4):
                v = matrix[y,x]
                label = f'{v:.5f}' if np.isfinite(v) else 'not tested'
                isbest = np.isfinite(v) and v == best['best_eval_loss']
                ax.text(x,y,label,ha='center',va='center',fontsize=10,
                        color='white' if np.isfinite(v) and v < np.nanmedian(matrix) else '#222222',
                        weight='bold' if isbest else 'normal',
                        bbox=dict(facecolor='none',edgecolor='red',linewidth=2,pad=5) if isbest else None)
        ax.set(xticks=range(4),xticklabels=('1e-4','2e-4','3e-4','4e-4'),
               yticks=range(3),yticklabels=('16','32','64'),xlabel='Learning rate',ylabel='LoRA rank',title=f'{size} / {condition}')
fig.suptitle('Qwen3.5 LoRA v2: best development loss\nRed box = selected candidate; colour scales are separate for each panel',fontsize=14)
fig.savefig(OUT/'candidate_losses.png',dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
colors = plt.cm.tab10(np.linspace(0,1,10))
for i,size in enumerate(('2B','9B')):
    for j,condition in enumerate(('C1','C2','C3')):
        ax=axes[i,j]
        g=results[f'{size}/{condition}']
        for k,candidate in enumerate(spec['candidates']):
            matches=[r for r in g['all_runs'] if r['candidate']==candidate]
            if not matches:
                continue
            r=matches[0]
            ev=[h for h in histories.get(r['id'],[]) if isinstance(h.get('eval_loss'),(int,float))]
            if not ev:
                continue
            selected=r['id']==g['selected_run']
            ax.plot([h['_step'] for h in ev],[h['eval_loss'] for h in ev],
                    color=colors[k],label=candidate,linewidth=2.8 if selected else 1,
                    alpha=1 if selected else .65)
        ax.set(title=f'{size} / {condition}',xlabel='Optimizer step',ylabel='Development loss')
        ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
fig.suptitle('Full development-loss histories (bold = selected candidate)',fontsize=14)
fig.savefig(OUT/'development_curves.png',dpi=180)
plt.close(fig)
