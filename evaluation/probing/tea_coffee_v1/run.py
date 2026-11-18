"""Generate independent prefix parses with a frozen 9B C3 adapter."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from panel import HERE, digest, read_panel
from adapters import prepare_adapter, selected_entry

ROOT = HERE.parents[2]
LORA = ROOT / 'model' / 'lora'


def checkpoint_spec(manifest, seed, checkpoint=None, config_path=None):
    entry = selected_entry(manifest, seed)
    path = checkpoint or Path(entry['checkpoint_path'])
    config_path = config_path or path.parent / 'config.json'
    if digest(config_path) != entry['config_sha256']:
        raise ValueError('Training config hash mismatch')
    for name, expected in entry['files'].items():
        if digest(path / name) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    config = json.loads(config_path.read_text())
    if (config['run_id'], config['model_size'], config['condition'], str(config['seed'])) != (
            entry['run_id'], '9B', 'C3', str(seed)):
        raise ValueError('Training config identity mismatch')
    spec = config['protocol']['models']['9B']
    if (spec['id'], spec['revision']) != (entry['base_model'], entry['base_revision']):
        raise ValueError('Base model identity mismatch')
    return entry, path, config_path, config


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stimuli', type=Path, default=HERE / 'stimuli.json')
    p.add_argument('--stimuli-only', action='store_true', help='validate the 16 inputs without checkpoint files')
    p.add_argument('--manifest', type=Path, help='existing frozen checkpoint manifest')
    p.add_argument('--seed', type=int, choices=(24, 42, 4224))
    p.add_argument('--checkpoint', type=Path, help='relocated selected adapter directory')
    p.add_argument('--config', type=Path, help='unchanged training config.json if stored separately')
    p.add_argument('--adapter-store', type=Path, help='reuse verified adapters here; download missing files')
    p.add_argument('--adapter-repo', default='Shrikes/repair-sbn-formal-adapters')
    p.add_argument('--adapter-revision', help='full Hub commit SHA; otherwise pin once in the adapter store')
    p.add_argument('--model-cache', type=Path)
    p.add_argument('--output', type=Path, help='new directory for this seed and attempt')
    p.add_argument('--check', action='store_true', help='verify inputs and checkpoint hashes without a model')
    args = p.parse_args()
    panel, rows = read_panel(args.stimuli)
    print(f"{panel['family_id']}: {len(rows)} unique inputs, {len(panel['paths'])} paths", flush=True)
    if args.stimuli_only:
        for row in rows:
            print(f"{row['id']}\t{row['text']}")
        return
    if args.manifest is None or args.seed is None:
        p.error('--manifest and --seed are required unless --stimuli-only is used')
    if args.adapter_store and (args.checkpoint or args.config):
        p.error('--adapter-store cannot be combined with --checkpoint or --config')
    if args.adapter_revision and not args.adapter_store:
        p.error('--adapter-revision requires --adapter-store')
    if not args.check:
        if args.output is None:
            p.error('--output is required for generation')
        if args.output.exists():
            raise FileExistsError(args.output)
    adapter_source = None
    if args.adapter_store:
        args.checkpoint, args.config, adapter_source = prepare_adapter(
            args.manifest, args.seed, args.adapter_store, args.adapter_repo,
            args.adapter_revision, allow_download=not args.check)
        print(f"Adapter source: {adapter_source['repo']}@{adapter_source['revision']}", flush=True)
    entry, checkpoint, config_path, config = checkpoint_spec(
        args.manifest, args.seed, args.checkpoint, args.config)
    print(f"Verified {entry['run_id']} at {checkpoint}", flush=True)
    if args.check:
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(LORA))
    from data import prompt_ids

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 CUDA GPU required')
    protocol = config['protocol']
    cache = str(args.model_cache or os.environ.get('MODEL_CACHE') or config['model_cache'])
    generation = dict(do_sample=False, use_cache=True,
                      max_new_tokens=protocol['candidate']['generation_max_new_tokens'])
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.stimuli, args.output / 'stimuli.json')
    shutil.copyfile(config_path, args.output / 'training-config.json')
    meta = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                family_id=panel['family_id'], seed=args.seed, condition='C3', model_size='9B',
                checkpoint=entry, resolved_checkpoint_path=str(checkpoint.resolve()),
                adapter_source=adapter_source, adapter_prepare_code_sha256=digest(HERE / 'adapters.py'),
                frozen_manifest_sha256=digest(args.manifest), stimuli_sha256=digest(args.stimuli),
                script_sha256=digest(Path(__file__)), prompt_code_sha256=digest(LORA / 'data.py'),
                instruction=protocol['instruction'], enable_thinking=False,
                generation=generation, model_cache=cache, attention='sdpa', dtype='bfloat16',
                gpu=torch.cuda.get_device_name(0), python=sys.version,
                packages={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft')})
    write_json(args.output / 'run.json', meta)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base = AutoModelForCausalLM.from_pretrained(
            entry['base_model'], revision=entry['base_revision'], cache_dir=cache,
            local_files_only=True, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            low_cpu_mem_usage=True, attn_implementation='sdpa')
        model = PeftModel.from_pretrained(base, checkpoint, local_files_only=True).eval()
        with (args.output / 'predictions.jsonl').open('x') as stream:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for row in rows:
                    rendered, ids = prompt_ids(tokenizer, row['text'], protocol['instruction'])
                    inputs = torch.tensor([ids], device='cuda:0')
                    tokens = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            **generation)[0, len(ids):].tolist()
                    result = dict(row, family_id=panel['family_id'], seed=args.seed, condition='C3',
                                  run_id=entry['run_id'], prompt=rendered, prompt_token_ids=ids,
                                  generated_token_ids=tokens,
                                  raw_output=tokenizer.decode(tokens, skip_special_tokens=False),
                                  pred_sbn=tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                                  tokens=len(tokens),
                                  ended_with_eos=bool(tokens and tokens[-1] == tokenizer.eos_token_id))
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(f"{row['id']}: {len(tokens)} generated tokens", flush=True)
        meta.update(status='complete', predictions=len(rows),
                    predictions_sha256=digest(args.output / 'predictions.jsonl'))
    except BaseException as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meta['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'run.json', meta)


if __name__ == '__main__':
    main()
