"""Versioned candidate sets shared by registration, execution and selection."""
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LORA = HERE.parent
PLAN_NAMES = ('lr-rank-v1', 'lr-rank-v2')


def load_plan(name='lr-rank-v1'):
    if name not in PLAN_NAMES:
        raise ValueError(f'Unknown tuning plan: {name}')
    path = HERE / 'configs' / f'{name}.json'
    spec = json.loads(path.read_text())
    if name == 'lr-rank-v1':
        candidates = {f'lr{lr:g}-r{rank}': (lr, rank)
                      for lr, rank in itertools.product(
                          spec['parameters']['learning_rate']['values'],
                          spec['parameters']['lora_rank']['values'])}
        protocol = LORA / 'prompts/sbn_v2/protocol.json'
        subset = HERE / 'subsets/train10_s20261103/manifest.json'
    else:
        candidates = {k: tuple(v) for k, v in spec['candidates'].items()}
        protocol, subset = LORA / spec['protocol'], LORA / spec['subset']
    if len(set(candidates.values())) != len(candidates):
        raise ValueError('Duplicate tuning configurations')
    return dict(name=name, spec=spec, candidates=candidates, protocol=protocol,
                subset=subset, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def resolve_assignment(config, plan):
    result = dict(config)
    if plan['name'] == 'lr-rank-v2':
        candidate = result.get('candidate')
        if candidate not in plan['candidates']:
            raise ValueError('Unknown tuning candidate')
        lr, rank = plan['candidates'][candidate]
        if any(key in result and result[key] != value
               for key, value in [('learning_rate', lr), ('lora_rank', rank)]):
            raise ValueError('Candidate conflicts with assigned LR/rank')
        result.update(learning_rate=lr, lora_rank=rank)
    if ((result.get('learning_rate'), result.get('lora_rank')) not in plan['candidates'].values()
            or result.get('seed') != 17):
        raise ValueError(f"Assignment differs from {plan['name']}")
    return result
