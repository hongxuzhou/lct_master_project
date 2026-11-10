"""Read a v2 registry without importing ML packages or contacting W&B."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plans import load_plan


def sweep_path(path, size, condition):
    registry = json.loads(Path(path).read_text())
    plan = load_plan('lr-rank-v2')
    if (registry.get('tuning_plan') != plan['name'] or registry.get('model_size') != size
            or registry.get('grid_sha256') != plan['sha256']):
        raise ValueError('Registry plan, model size or candidate hash differs')
    protocol = json.loads(plan['protocol'].read_text())
    settings = protocol['wandb']
    if registry.get('base_url') != settings['base_url']:
        raise ValueError('Registry W&B host differs')
    for key, source in [('subset_manifest_sha256', plan['subset']), ('protocol_sha256', plan['protocol'])]:
        if registry.get(key) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f'Registry {key} differs')
    sweep = registry['sweeps'][condition]
    prefix = f"{settings['entity']}/{settings['project']}/"
    if not sweep.startswith(prefix) or not sweep[len(prefix):].isalnum():
        raise ValueError('Invalid sweep path')
    return sweep


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', type=Path)
    parser.add_argument('size', choices=('2B', '9B', '27B'))
    parser.add_argument('condition', choices=('C1', 'C2', 'C3'))
    args = parser.parse_args()
    print(sweep_path(args.registry, args.size, args.condition))
