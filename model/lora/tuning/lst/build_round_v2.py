"""Write a CPU-preparation -> six GPU agents DAG. Standard library only."""
import argparse
from pathlib import Path
import re
import sys

LORA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LORA / 'tuning'))
from submit import command


def build_round(output):
    output = Path(output).resolve()
    if any(ch.isspace() or ch in '\"\'' for ch in str(output)):
        raise ValueError('DAG output path must not contain whitespace or quotes')
    output.mkdir(parents=True, exist_ok=True)
    template = (LORA / 'tuning/lst/agent_v2.sub').read_text()
    template = template.replace('$(LORA_DIR)/tuning/agent_v2.sh lst',
                                '$(LORA_DIR)/tuning/lst/run_uploaded_v2.sh')
    template = template.replace('getenv = PYTHON,MODEL_CACHE,DATASET_DIR,TUNING_RUN_ROOT,WANDB_API_KEY,HF_TOKEN_PATH',
                                'getenv = WANDB_API_KEY,HF_TOKEN_PATH')
    prep = f'''universe = vanilla
initialdir = /nethome/honzhou
executable = /bin/bash
arguments = {LORA}/tuning/lst/prepare_uploaded_v2.sh
output = /scratch/hongxuzhou/logs/qwen35-tune-v2-prepare.$(ClusterId).out
error = /scratch/hongxuzhou/logs/qwen35-tune-v2-prepare.$(ClusterId).err
log = /scratch/hongxuzhou/logs/qwen35-tune-v2-prepare.$(ClusterId).log
request_cpus = 4
request_memory = 16G
request_gpus = 0
requirements = (TARGET.UidDomain == "coli.uni-saarland.de")
+MaxWallTime = 7200
notification = Never
should_transfer_files = NO
getenv = WANDB_API_KEY,HF_TOKEN_PATH
queue 1
'''
    (output / 'prepare.sub').write_text(prep)
    lines = [f'JOB PREPARE {output / "prepare.sub"}']
    children = []
    for size in ('9B', '27B'):
        for condition in ('C1', 'C2', 'C3'):
            # Reuse the normal submission resource routing without requiring
            # registries that the CPU parent has not created yet.
            cmd = command('lst', size, None, condition, 9, fit=True)
            values = dict(item.split(' = ', 1) for item in cmd[3::2])
            values.update(REGISTRY=str(LORA / f'tuning/registrations/lr-rank-v2-{size}.json'), MODE='tuning')
            text = re.sub(r'\$\((\w+)\)', lambda m: values.get(m[1], m[0]), template)
            node = f'TUNE_{size}_{condition}'
            path = output / f'{size}-{condition}.sub'
            path.write_text(text)
            lines.append(f'JOB {node} {path}')
            children.append(node)
    lines.append('PARENT PREPARE CHILD ' + ' '.join(children))
    dag = output / 'round.dag'
    dag.write_text('\n'.join(lines) + '\n')
    return dag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=LORA / 'tuning/lst/round-v2')
    args = parser.parse_args()
    print(build_round(args.output))
