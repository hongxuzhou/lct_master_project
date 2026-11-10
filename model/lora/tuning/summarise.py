"""Select only complete, compatible runs from a versioned candidate set."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plans import load_plan, PLAN_NAMES


def collect(root, plan_name='lr-rank-v1', model_size=None, condition=None):
    plan = load_plan(plan_name)
    groups = defaultdict(list)
    incomplete = []
    for path in sorted(Path(root).glob(f'{model_size or "*"}/{condition or "*"}/*/config.json')):
        directory = path.parent
        config = json.loads(path.read_text())
        summary_path = directory / 'summary.json'
        if config.get('purpose') != 'tuning':
            continue
        if not summary_path.exists() or (directory / 'failure.json').exists():
            incomplete.append(str(directory))
            continue
        summary = json.loads(summary_path.read_text())
        if (summary['status'] != 'development_passed' or not math.isfinite(summary['best_eval_loss'])
                or not math.isclose(summary['best_eval_loss'], summary['reload_eval_loss'], rel_tol=1e-5, abs_tol=1e-5)):
            incomplete.append(str(directory))
            continue
        groups[(config['model_size'], config['condition'])].append({
            'run_id': config['wandb_run_id'], 'directory': str(directory),
            'learning_rate': config['learning_rate'], 'lora_rank': config['lora_rank'],
            'seed': config['seed'], 'subset_sha256': config['subset']['manifest_sha256'],
            'protocol_sha256': config['code_sha256']['protocol.json'],
            'train_sha256': config['code_sha256']['train.py'],
            'data_sha256': config['code_sha256']['data.py'],
            'entry_sha256': config['sweep_assignment']['sweep_entry_sha256'],
            'tuning_plan': config['sweep_assignment'].get('tuning_plan', 'lr-rank-v1'),
            'grid_sha256': config['sweep_assignment'].get('grid_sha256'),
            'plan_code_sha256': config['sweep_assignment'].get('plan_code_sha256'),
            **summary})
    expected = set(plan['candidates'].values())
    result = {}
    for (size, condition), trials in groups.items():
        points = [(t['learning_rate'], t['lora_rank']) for t in trials]
        compatible = len({(t['seed'], t['subset_sha256'], t['protocol_sha256'], t['train_sha256'],
                            t['data_sha256'], t['entry_sha256'], t['grid_sha256'],
                            t['plan_code_sha256']) for t in trials}) == 1
        matches_plan = all(t['tuning_plan'] == plan_name for t in trials)
        if plan_name == 'lr-rank-v2':
            matches_plan = matches_plan and all(
                t['seed'] == 17 and t['grid_sha256'] == plan['sha256']
                and t['subset_sha256'] == hashlib.sha256(plan['subset'].read_bytes()).hexdigest()
                and t['protocol_sha256'] == hashlib.sha256(plan['protocol'].read_bytes()).hexdigest()
                and t['plan_code_sha256'] is not None for t in trials)
        complete = len(points) == len(expected) and set(points) == expected and compatible and matches_plan
        ordered = sorted(trials, key=lambda t: (t['best_eval_loss'], t['run_id']))
        result[f'{size}/{condition}'] = {'complete_grid': complete,
            'tuning_plan': plan_name, 'matches_plan': matches_plan,
            'missing': [list(p) for p in sorted(expected-set(points))],
            'unexpected': [list(p) for p in sorted(set(points)-expected)],
            'duplicate_configurations': len(points) != len(set(points)),
            'compatible_provenance': compatible,
            'selected_run': ordered[0]['run_id'] if complete else None, 'trials': ordered}
    return {'groups': result, 'incomplete_or_failed': incomplete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', choices=PLAN_NAMES, default='lr-rank-v1')
    parser.add_argument('--model-size', choices=('2B', '9B', '27B'))
    parser.add_argument('--condition', choices=('C1', 'C2', 'C3'))
    args = parser.parse_args()
    result = collect(args.run_root, args.plan, args.model_size, args.condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {n:v for n,v in g.items() if n != 'trials'} for k,g in result['groups'].items()}, indent=2))


if __name__ == '__main__':
    main()
