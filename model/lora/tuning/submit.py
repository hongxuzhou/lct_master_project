"""Submit all conditions for a complete v2 model sweep, or preview commands."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import sweep_path

LORA = Path(__file__).resolve().parents[1]


def command(cluster, size, registry, condition, count, fit=False):
    if (cluster, size) not in {('habrok', '2B'), ('lst', '9B'), ('lst', '27B')}:
        raise ValueError('Use Habrok 2B, LST A100 9B or LST H100 27B')
    if not 1 <= count <= 9 or condition not in ('C1', 'C2', 'C3'):
        raise ValueError('Invalid count or condition')
    mode = 'fit' if fit else 'tuning'
    path = str(Path(registry).resolve()) if registry else '-'
    if not fit:
        sweep_path(path, size, condition)
    if cluster == 'habrok':
        return ['sbatch', '--export=ALL', str(LORA / 'tuning/habrok/agent_v2.sbatch'),
                path, condition, str(count), mode]
    # HTCondor arguments are unquoted macros. Reject whitespace and quote syntax
    # rather than silently splitting paths or introducing new submit directives.
    if any(any(ch.isspace() or ch in '\"\'' for ch in value) for value in (str(LORA), path)):
        raise ValueError('LST code and registry paths must not contain whitespace or quotes')
    resources = dict(CPUS=4, HOST_RAM='64G', CAPABILITY='8.0', MIN_GPU_MB=39000, MAX_GPU_MB=41000)
    if size == '27B':
        resources = dict(CPUS=8, HOST_RAM='128G', CAPABILITY='9.0', MIN_GPU_MB=79000, MAX_GPU_MB=85000)
    macros = dict(LORA_DIR=str(LORA), MODEL_SIZE=size, REGISTRY=path, CONDITION=condition,
                  COUNT=count, MODE=mode, WALL_SECONDS=259200 if size == '27B' else 86400, **resources)
    cmd = ['condor_submit', str(LORA / 'tuning/lst/agent_v2.sub')]
    for key, value in macros.items():
        cmd.extend(['-append', f'{key} = {value}'])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster', required=True, choices=('habrok', 'lst'))
    parser.add_argument('--model-size', required=True, choices=('2B', '9B', '27B'))
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--condition', default='all', choices=('all', 'C1', 'C2', 'C3'))
    parser.add_argument('--count', type=int, help='Default: all nine candidates; fit check: one')
    parser.add_argument('--fit-check', action='store_true')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--submit', action='store_true')
    action.add_argument('--check', action='store_true', help='Scheduler dry run; no GPU job submitted')
    args = parser.parse_args()
    if not args.fit_check and not args.registry:
        args.registry = LORA / f'tuning/registrations/lr-rank-v2-{args.model_size}.json'
    if args.count is None:
        args.count = 1 if args.fit_check else 9
    if args.fit_check and args.count != 1:
        parser.error('A fit check is one calibration, not a multi-trial agent')
    try:
        conditions = (['C3'] if args.fit_check else ['C1', 'C2', 'C3']) if args.condition == 'all' else [args.condition]
        commands = [command(args.cluster, args.model_size, args.registry, c, args.count, args.fit_check)
                    for c in conditions]
    except (ValueError, KeyError, FileNotFoundError) as e:
        parser.error(str(e))
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if not (args.check or args.submit):
        return
    for key in ('PYTHON', 'MODEL_CACHE', 'DATASET_DIR', 'TUNING_RUN_ROOT'):
        if not os.environ.get(key):
            parser.error(f'Missing {key}; use the cluster submit_v2.sh wrapper')
    for split in ('train', 'validation'):
        if not (Path(os.environ['DATASET_DIR']) / 'data' / f'{split}.parquet').is_file():
            parser.error(f'Missing prepared {split} split')
    env = {**os.environ, 'LORA_DIR': str(LORA)}
    cwd = LORA / 'tuning/habrok' if args.cluster == 'habrok' else Path('/nethome/honzhou')
    logs = cwd / 'logs' if args.cluster == 'habrok' else Path('/scratch/hongxuzhou/logs')
    logs.mkdir(parents=True, exist_ok=True)
    for cmd in commands:
        if args.check:
            if args.cluster == 'habrok':
                cmd.insert(1, '--test-only')
            else:
                with tempfile.TemporaryDirectory(prefix='tuning-v2-condor-') as directory:
                    subprocess.run(cmd + ['-dry-run', str(Path(directory) / 'job.ad')], env=env, cwd=cwd, check=True)
                continue
        subprocess.run(cmd, env=env, cwd=cwd, check=True)


if __name__ == '__main__':
    main()
