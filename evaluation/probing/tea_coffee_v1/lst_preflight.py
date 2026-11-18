"""Check the small deployment bundle and runtime inside a compute allocation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from panel import digest, read_panel
from adapters import selected_entry


def check_cache(cache, entry):
    snapshot = cache / ('models--' + entry['base_model'].replace('/', '--')) / 'snapshots' / entry['base_revision']
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json'}
    index = snapshot / 'model.safetensors.index.json'
    if index.is_file():
        required.add(index.name)
        required.update(json.loads(index.read_text())['weight_map'].values())
    else:
        required.add('model.safetensors')
    for name in required:
        path = snapshot / name
        if not path.is_file() or not os.access(path, os.R_OK) or not path.stat().st_size:
            raise ValueError(f'Missing, empty or unreadable base-model file: {path}')
    return str(snapshot)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    bundle = json.loads((args.root / 'bundle.json').read_text())
    for name, sha in bundle['files'].items():
        if digest(args.root / name) != sha:
            raise ValueError(f'Deployment bundle hash mismatch: {name}')
    manifest = args.root / 'frozen-9B-C3.json'
    for seed in (24, 42, 4224):
        entry = selected_entry(manifest, seed)
        snapshot = check_cache(args.model_cache, entry)
    panel, rows = read_panel(args.root / 'evaluation/probing/tea_coffee_v1/stimuli.json')

    import torch
    import transformers
    import peft
    import huggingface_hub

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError('Expected one visible BF16 CUDA GPU')
    gpu = torch.cuda.get_device_properties(0)
    if 'A100' not in gpu.name or gpu.total_memory < 39 * 10**9:
        raise ValueError(f'Expected A100 with at least 39 GB, got {gpu.name}')
    token = Path(os.environ['HF_TOKEN_PATH'])
    report = dict(status='passed', python=sys.executable, gpu=gpu.name,
                  gpu_bytes=gpu.total_memory, base_snapshot=snapshot,
                  family_id=panel['family_id'], inputs=len(rows),
                  bundle_sha256=digest(args.root / 'bundle.json'),
                  hf_token_file_readable=token.is_file() and os.access(token, os.R_OK),
                  hub_access='checked by adapter preparation only when a download is needed',
                  packages={name: importlib.metadata.version(name) for name in
                            ('torch', 'transformers', 'peft', 'huggingface-hub')})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
