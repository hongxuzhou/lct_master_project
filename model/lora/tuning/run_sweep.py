"""One fresh training process per CLI-agent assignment; no nested W&B run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LORA = HERE.parent
sys.path.insert(0, str(LORA))
sys.path.insert(0, str(HERE))
from plans import load_plan, resolve_assignment, PLAN_NAMES
from data import condition_rows, load_rows, read_protocol, select_train_subset
from train import parse_args, run_training, training_plan

PROTOCOL = LORA / 'prompts/sbn_v2/protocol.json'
SUBSET = HERE / 'subsets/train10_s20261103/manifest.json'
GRID = HERE / 'configs/lr-rank-v1.json'


def training_args(config, output, plan_name='lr-rank-v1', cluster='habrok'):
    plan = load_plan(plan_name)
    config = resolve_assignment(config, plan)
    if config['model_size'] not in ('2B', '9B', '27B') or config['condition'] not in ('C1', 'C2', 'C3'):
        raise ValueError('Invalid size or condition')
    return parse_args([
        '--protocol', str(plan['protocol']), '--purpose', 'tuning', '--max-steps', '0',
        '--model-size', config['model_size'], '--condition', config['condition'],
        '--seed', str(config['seed']), '--learning-rate', str(config['learning_rate']),
        '--lora-rank', str(config['lora_rank']), '--lora-alpha', str(2 * config['lora_rank']),
        '--train-subset', str(plan['subset']), '--eval-epoch-fraction', '0.3',
        '--micro-batch-size', '1', '--generation-examples', '0',
        '--cluster', cluster, '--wandb-mode', 'online',
        '--dataset-dir', os.environ['DATASET_DIR'], '--model-cache', os.environ['MODEL_CACHE'],
        '--output-dir', str(output)])


def dry_run(size, condition, plan_name='lr-rank-v1', cluster='habrok'):
    plan = load_plan(plan_name)
    protocol = read_protocol(plan['protocol'])
    rows, subset = select_train_subset(load_rows(os.environ['DATASET_DIR'], protocol), plan['subset'], protocol)
    result = []
    for candidate, (lr, rank) in plan['candidates'].items():
        config = dict(candidate=candidate, model_size=size, condition=condition, learning_rate=lr, lora_rank=rank, seed=17)
        args = training_args(config, Path('/tmp/unused-tuning-dry-run'), plan_name, cluster)
        result.append({**config, 'lora_alpha': rank*2,
                       **training_plan(len(condition_rows(rows['train'], condition)), protocol, args)})
    print(json.dumps({'tuning_plan': plan_name, 'cluster': cluster, 'subset': subset, 'validation_rows': len(condition_rows(rows['validation'], condition)),
                      'trials': result}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model-size', choices=('2B', '9B', '27B'), default='2B')
    parser.add_argument('--condition', choices=('C1', 'C2', 'C3'), default='C1')
    parser.add_argument('--plan', choices=PLAN_NAMES, default='lr-rank-v1')
    parser.add_argument('--cluster', choices=('habrok', 'lst', 'local'),
                        default=os.environ.get('TUNING_CLUSTER', 'habrok'))
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.model_size, args.condition, args.plan, args.cluster)
        return
    if importlib.metadata.version('wandb') != '0.31.0':
        raise RuntimeError('This tuning entry point requires wandb==0.31.0')
    if not os.environ.get('WANDB_SWEEP_ID'):
        raise RuntimeError('Start this entry point through a W&B CLI sweep agent')
    plan = load_plan(args.plan)
    host = read_protocol(plan['protocol'])['wandb']['base_url']
    if os.environ.get('WANDB_BASE_URL') != host:
        raise RuntimeError('Agent and training must use the configured Forge API host')
    for key, value in {'WANDB_DISABLE_CODE': 'true', 'WANDB_DISABLE_GIT': 'true',
                       'WANDB_CONSOLE': 'off', 'WANDB_LOG_MODEL': 'false', 'WANDB_WATCH': 'false'}.items():
        os.environ[key] = value
    import wandb
    wandb_dir = Path(os.environ['TUNING_RUN_ROOT']) / 'wandb'
    wandb_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(job_type='tuning', mode='online', dir=str(wandb_dir), settings=wandb.Settings(
            base_url=host, disable_git=True, disable_code=True, console='off',
            init_timeout=30, finish_timeout=30)) as run:
        config = resolve_assignment(dict(run.config), plan)
        for env_key, config_key in [('TUNING_MODEL_SIZE', 'model_size'), ('TUNING_CONDITION', 'condition')]:
            if os.environ.get(env_key) and config[config_key] != os.environ[env_key]:
                raise ValueError(f'Sweep assignment differs from allocated {config_key}')
        expected = hashlib.sha256(plan['subset'].read_bytes()).hexdigest()
        if config.get('subset_manifest_sha256') != expected:
            raise ValueError('Sweep was registered with a different subset')
        if args.plan == 'lr-rank-v2':
            expected_metadata = {'tuning_plan': args.plan, 'grid_sha256': plan['sha256'],
                                 'protocol_sha256': hashlib.sha256(plan['protocol'].read_bytes()).hexdigest()}
            if any(config.get(k) != v for k, v in expected_metadata.items()):
                raise ValueError('Sweep plan or protocol differs from its registration')
            run.config.update({'learning_rate': config['learning_rate'], 'lora_rank': config['lora_rank'],
                               'plan_code_sha256': hashlib.sha256((HERE / 'plans.py').read_bytes()).hexdigest()})
        run.name = f"{config['model_size']}-{config['condition']}-lr{config['learning_rate']:g}-r{config['lora_rank']}-{run.id}"
        run.tags = ['tuning', args.plan, args.cluster, config['model_size'], config['condition']]
        output = Path(os.environ['TUNING_RUN_ROOT']) / config['model_size'] / config['condition'] / run.id
        run.config.update({'sweep_entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'grid_sha256': plan['sha256']})
        run_training(training_args(config, output, args.plan, args.cluster), wandb_run=run)


if __name__ == '__main__':
    main()
