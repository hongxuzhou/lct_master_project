"""Register three condition-specific sweeps, only with explicit --create."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plans import load_plan, PLAN_NAMES

HERE = Path(__file__).resolve().parent


def configs(size, plan_name='lr-rank-v1'):
    plan = load_plan(plan_name)
    template = plan['spec']
    if plan_name == 'lr-rank-v2':
        template = dict(name=plan_name, program='tuning/run_sweep.py', method='grid',
                        metric={'name': 'best_eval_loss', 'goal': 'minimize'},
                        run_cap=len(plan['candidates']),
                        command=['${env}', '${interpreter}', '${program}', '--plan', plan_name],
                        parameters={'candidate': {'values': list(plan['candidates'])},
                                    'seed': {'value': 17}})
    subset_hash = hashlib.sha256(plan['subset'].read_bytes()).hexdigest()
    result = {}
    for condition in ('C1', 'C2', 'C3'):
        spec = copy.deepcopy(template)
        spec['name'] += f'-{size}-{condition}'
        for key, value in {'model_size': size, 'condition': condition,
                           'subset_manifest_sha256': subset_hash}.items():
            spec['parameters'][key] = {'value': value}
        if plan_name == 'lr-rank-v2':
            for key, value in {'tuning_plan': plan_name, 'grid_sha256': plan['sha256'],
                               'protocol_sha256': hashlib.sha256(plan['protocol'].read_bytes()).hexdigest()}.items():
                spec['parameters'][key] = {'value': value}
        result[condition] = spec
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-size', choices=('2B', '9B', '27B'), default='2B')
    parser.add_argument('--plan', choices=PLAN_NAMES, default='lr-rank-v1')
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--registry', type=Path)
    args = parser.parse_args()
    specs = configs(args.model_size, args.plan)
    if not args.create:
        print(json.dumps(specs, indent=2))
        return
    if not args.registry or args.registry.exists():
        parser.error('--create requires a new --registry path; retain partial registries instead of recreating sweeps')
    if importlib.metadata.version('wandb') != '0.31.0':
        raise RuntimeError('Requires wandb==0.31.0')
    plan = load_plan(args.plan)
    settings = json.loads(plan['protocol'].read_text())['wandb']
    os.environ['WANDB_BASE_URL'] = settings['base_url']
    import wandb
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    registry = {'model_size': args.model_size, 'tuning_plan': args.plan,
                'grid_sha256': plan['sha256'], 'base_url': settings['base_url'], 'sweeps': {}}
    if args.plan == 'lr-rank-v2':
        registry.update(subset_manifest_sha256=hashlib.sha256(plan['subset'].read_bytes()).hexdigest(),
                        protocol_sha256=hashlib.sha256(plan['protocol'].read_bytes()).hexdigest())
    # Keep each successful registration if a later request fails.
    args.registry.write_text(json.dumps(registry, indent=2)+'\n')
    for condition, spec in specs.items():
        sweep_id = wandb.sweep(spec, entity=settings['entity'], project=settings['project'])
        registry['sweeps'][condition] = f"{settings['entity']}/{settings['project']}/{sweep_id}"
        args.registry.write_text(json.dumps(registry, indent=2)+'\n')
    print(json.dumps(registry, indent=2))


if __name__ == '__main__':
    main()
