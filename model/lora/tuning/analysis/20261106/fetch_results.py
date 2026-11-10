"""Read-only W&B snapshot; uses the existing repairsbn login."""
import datetime
import json
from collections import Counter
from pathlib import Path
import wandb

OUT = Path(__file__).resolve().parent
api = wandb.Api(overrides={'base_url': 'https://api.forge.coreweave.com'}, timeout=60)
rows = []
for run in api.runs('hxzhou-research/qwen35-rsr-lora', per_page=100):
    rows.append(dict(id=run.id, name=run.name, state=run.state, url=run.url,
                     created_at=run.created_at, tags=run.tags,
                     sweep=run.sweep.id if run.sweep else None,
                     config=dict(run.config), summary=dict(run.summary)))
(OUT / 'runs.json').write_text(json.dumps({
    'fetched_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'base_url': 'https://api.forge.coreweave.com', 'runs': rows}, indent=2))
print('Runs:', len(rows), flush=True)
print(Counter((r['config'].get('tuning_plan'), r['config'].get('model_size'),
               r['state']) for r in rows), flush=True)
histories = {}
for r in rows:
    if r['config'].get('tuning_plan') != 'lr-rank-v2':
        continue
    run = api.run('hxzhou-research/qwen35-rsr-lora/' + r['id'])
    histories[r['id']] = list(run.scan_history(page_size=1000))
    print(r['id'], r['state'], len(histories[r['id']]), flush=True)
    (OUT / 'histories.json').write_text(json.dumps(histories, indent=2))
