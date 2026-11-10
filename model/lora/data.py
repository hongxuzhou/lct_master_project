"""Condition views and the audited neutral chat format. No data is rewritten."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
FLUENT_VARIANTS = frozenset({'pmb_fluent', 'fluent_split_reference'})
RSR_VARIANTS = frozenset({'rsr_base', 'rsr_editing_expression'})


def read_protocol(path=PROTOCOL_PATH):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_rows(dataset_dir, protocol):
    import pyarrow.parquet as pq

    result = {}
    for split, spec in protocol["dataset"]["files"].items():
        path = Path(dataset_dir) / spec["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"{split} file checksum differs from the accepted release")
        rows = pq.read_table(path).to_pylist()
        if Counter(r["variant"] for r in rows) != protocol["dataset"]["variants"][split]:
            raise ValueError(f"Unexpected {split} counts")
        if len({r["example_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {split} example IDs")
        result[split] = rows
    if {r["doc_id"] for r in result["train"]} & {r["doc_id"] for r in result["validation"]}:
        raise ValueError("Train/dev source-document overlap")
    return result


def condition_rows(rows, condition):
    if condition not in ("C1", "C2", "C3"):
        raise ValueError(condition)
    field = "sbn_repair" if condition == "C3" else "sbn_clean"
    return [{**r, "target": r[field]} for r in rows
            if condition != "C1" or r["variant"] in FLUENT_VARIANTS]


def select_train_subset(rows, manifest_path, protocol):
    """Validate an ID manifest against the accepted train; preserve source order."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    expected = {'schema_version': 1, 'split': 'train',
                'dataset_id': protocol['dataset']['id'],
                'dataset_revision': protocol['dataset']['revision'],
                'train_sha256': protocol['dataset']['files']['train']['sha256']}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Subset provenance differs from the accepted train')
    ids = manifest['example_ids']
    if not ids or len(set(ids)) != len(ids) or digest(ids) != manifest['example_ids_sha256']:
        raise ValueError('Invalid subset IDs or digest')
    wanted = set(ids)
    selected = [r for r in rows['train'] if r['example_id'] in wanted]
    if [r['example_id'] for r in selected] != ids:
        raise ValueError('Subset IDs are missing or differ from source order')
    selected_items = {r['item_id'] for r in selected if r['variant'] in RSR_VARIANTS}
    if any(r['example_id'] not in wanted for r in rows['train']
           if r['variant'] in RSR_VARIANTS and r['item_id'] in selected_items):
        raise ValueError('Subset splits a base/marked repair group')
    selected_units = {r.get('partition_unit') for r in selected if r['variant']=='fluent_split_reference'}
    if any(r['example_id'] not in wanted for r in rows['train']
           if r['variant']=='fluent_split_reference' and r.get('partition_unit') in selected_units):
        raise ValueError('Subset splits a fluent reference partition unit')
    if not any(r['variant'] in FLUENT_VARIANTS for r in selected) or not selected_items:
        raise ValueError('Subset must contain fluent and repair rows')
    return {**rows, 'train': selected}, {
        'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'example_ids_sha256': digest(ids), 'sampling_seed': manifest['sampling_seed'],
        'rows': len(selected)}


def prompt_ids(tokenizer, text, instruction):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction + "\n" + text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return rendered, tokenizer(rendered, add_special_tokens=False)["input_ids"]


def tokenize_row(row, tokenizer, instruction, max_length):
    prompt, prefix = prompt_ids(tokenizer, row["text"], instruction)
    target = row["target"]
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction + "\n" + row["text"]},
         {"role": "assistant", "content": target}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    suffix = ids[len(prefix):]
    if not full.startswith(prompt) or ids[:len(prefix)] != prefix:
        raise ValueError(f"Ambiguous assistant boundary: {row['example_id']}")
    if not suffix or not target.strip() or target not in full[len(prompt):]:
        raise ValueError(f"Missing assistant target: {row['example_id']}")
    if tokenizer.eos_token_id not in suffix:
        raise ValueError(f"Unsupervised EOS: {row['example_id']}")
    if len(ids) > max_length:
        raise ValueError(f"Refusing target truncation: {row['example_id']} ({len(ids)} tokens)")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + suffix,
            "prompt_length": len(prefix), "example_id": row["example_id"]}


def audit(rows, tokenizer, protocol, condition):
    encoded, report = {}, {}
    for split, source in rows.items():
        view = condition_rows(source, condition)
        examples = [tokenize_row(r, tokenizer, protocol["instruction"], protocol["candidate"]["max_length"]) for r in view]
        lengths = sorted(len(r["input_ids"]) for r in examples)
        encoded[split] = examples
        report[split] = {
            "rows": len(view), "source_documents": len({r["doc_id"] for r in view}),
            "input_digest": digest([[r["example_id"], r["text"]] for r in view]),
            "target_digest": digest([r["target"] for r in view]),
            "prompt_tokens": sum(r["prompt_length"] for r in examples),
            "supervised_tokens": sum(len(r["labels"]) - r["prompt_length"] for r in examples),
            "sequence_p50": lengths[len(lengths)//2], "sequence_max": max(lengths),
            "truncated": 0,
        }
    return encoded, report
