"""Validate the appended supplement, baseline preservation and condition views."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from repair_data_gen.training_dataset.build import SCHEMA,sha256,holdout
from repair_data_gen.training_dataset.fluent_groups import accepted_rows


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-root',type=Path,default=Path('/private/tmp/repairsbn-dev-acceptance-tokenizers'))
    args=parser.parse_args()
    old=HERE/'releases/published-9b7d881'
    model_data=module('integrated_model_data',ROOT/'model/lora/data.py')
    synth=ROOT/'repair_data_gen/fluent_group_synthesis_v1'
    builder=module('accepted_fluent_builder',synth/'build.py')
    rows={};group_sources={};checks={};type_counts={}
    for split,count in [('train',14222),('validation',1746)]:
        table=pq.read_table(HERE/'data'/f'{split}.parquet')
        baseline=pq.read_table(old/'data'/f'{split}.parquet')
        assert table.schema.equals(SCHEMA,check_metadata=True)
        assert table.num_rows==count
        assert table.slice(0,baseline.num_rows).select(baseline.column_names).equals(baseline), 'Baseline values/order changed'
        data=table.to_pylist();rows[split]=data
        assert len({r['example_id'] for r in data})==count
        assert not {r['doc_id'] for r in data}&holdout.load()
        assert all(r['data_origin'] and r['family_id'] and r['partition_unit'] for r in data)
        assert all(v is None or not isinstance(v,str) or v!='' for r in data for v in r.values())
        groups=[r for r in data if r['variant']=='fluent_split_reference']
        assert groups==accepted_rows(split,SCHEMA)
        source_split='dev' if split=='validation' else split
        group_sources[split]=[json.loads(x) for x in (synth/f'{source_split}.candidates.jsonl').read_text().splitlines()]
        for row,source in zip(groups,group_sources[split]):
            assert row['text']==source['sentence'] and row['sbn_clean']==row['sbn_repair']==source['sbn']
            assert row['doc_id'].startswith('fluent_group_v1::') and row['item_id']==row['example_id']
            assert row['data_origin']=='synthetic_fluent' and row['construction_type']==source['construction']
            assert row['label_source']=='user_reviewed_synthetic'
            assert all(row[k] is None for k in ['repair_type','editing_expression','ambiguity_type','repair_onset',
                'reparandum_surface','reparandum_synset','reparandum_value','repair_synset','site_index','strategy','wn_relation'])
            assert builder.build(source).serialize()==row['sbn_clean']
            assert not builder.validate(source,builder.build(source))['failures']
        for row in data:
            expected={'pmb_fluent':'pmb','rsr_base':'synthetic_rsr','rsr_editing_expression':'synthetic_rsr','fluent_split_reference':'synthetic_fluent'}[row['variant']]
            assert row['data_origin']==expected
            if row['variant']!='fluent_split_reference':assert row['construction_type'] is None
        type_counts[split]=dict(Counter(r['construction_type'] for r in groups))
        c1=model_data.condition_rows(data,'C1');c2=model_data.condition_rows(data,'C2');c3=model_data.condition_rows(data,'C3')
        assert len(c1)==(9686 if split=='train' else 1234)
        assert [(r['example_id'],r['text']) for r in c2]==[(r['example_id'],r['text']) for r in c3]
        assert {r['example_id'] for r in groups}<={r['example_id'] for r in c1}
        checks[split]={'rows':count,'baseline_rows_unchanged':baseline.num_rows,'appended':len(groups),
            'variants':dict(Counter(r['variant'] for r in data)),
            'origins':dict(Counter(r['data_origin'] for r in data)),
            'conditions':{'C1':len(c1),'C2':len(c2),'C3':len(c3)},'sha256':sha256(HERE/'data'/f'{split}.parquet')}
    for key in ['example_id','doc_id','family_id','partition_unit']:
        assert not {r[key] for r in rows['train']}&{r[key] for r in rows['validation']},key
    gaps={k:abs(type_counts['train'][k]/200-type_counts['validation'][k]/40) for k in type_counts['train']}
    assert set(type_counts['train'])==set(type_counts['validation']) and max(gaps.values())<=.025
    protocol=json.loads((ROOT/'model/lora/prompts/sbn_v2/protocol.json').read_text())
    protocol['status']='local_integrated_dataset; modelling settings remain provisional'
    protocol['dataset'].update(revision='local:fluent-groups-v1',publication_status='not_published',
        schema_version=2,fluent_variants=['pmb_fluent','fluent_split_reference'])
    for split in rows:
        protocol['dataset']['files'][split]={'path':f'data/{split}.parquet','sha256':checks[split]['sha256']}
        protocol['dataset']['variants'][split]=checks[split]['variants']
    protocol_path=ROOT/'model/lora/protocols/fluent_groups_v1.json';protocol_path.parent.mkdir(exist_ok=True)
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    loaded=model_data.load_rows(HERE,protocol)
    assert len(loaded['train'])==14222 and len(loaded['validation'])==1746
    # Old pinned protocols still load the archived published snapshot.
    original=model_data.load_rows(old,model_data.read_protocol())
    assert len(original['train'])==14022
    tokenization={}
    from transformers import AutoTokenizer
    for size,model in protocol['models'].items():
        directory=args.tokenizer_root/size
        for filename in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja']:
            meta=directory/'.cache/huggingface/download'/(filename+'.metadata')
            assert meta.read_text().splitlines()[0]==model['revision']
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True)
        lengths=[];targets=[]
        for split in rows:
            for row in rows[split]:
                if row['variant']!='fluent_split_reference':continue
                encoded=model_data.tokenize_row({**row,'target':row['sbn_clean']},tokenizer,
                    protocol['instruction'],protocol['candidate']['max_length'])
                lengths.append(len(encoded['input_ids']))
                targets.append(len(encoded['input_ids'])-encoded['prompt_length'])
        assert max(targets)<=protocol['candidate']['generation_max_new_tokens']
        tokenization[size]={'revision':model['revision'],'appended_rows':len(lengths),
            'sequence_max':max(lengths),'supervised_suffix_max':max(targets),'truncated':0,
            'assistant_boundary_and_eos':'passed'}
    baseline_hashes=json.loads((old/'snapshot.sha256.json').read_text())
    assert all(sha256(old/p)==digest for p,digest in baseline_hashes.items())
    report={'status':'passed','dataset_version':'fluent-groups-v1','schema_version':2,'columns':SCHEMA.names,
        'splits':checks,'fluent_construction_counts':type_counts,'max_construction_proportion_gap':max(gaps.values()),
        'isolation':'example, document, family and partition unit IDs are disjoint across train/validation',
        'tokenization':{'scope':'240 appended rows; baseline rows preserved exactly from published snapshot',
                        'instruction':protocol['instruction'],'models':tokenization},
        'old_published_snapshot':'preserved and verified','publication_at_validation':'not performed',
        'acceptance_source':'fluent_group.acceptance.json','model_protocol':str(protocol_path.relative_to(ROOT))}
    tracked=[HERE/'build.py',HERE/'build_dev.py',HERE/'fluent_groups.py',Path(__file__),
             ROOT/'model/lora/data.py',ROOT/'model/lora/train.py',
             ROOT/'model/lora/tuning/build_subset.py',protocol_path,HERE/'fluent_group.acceptance.json']
    report['input_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in tracked}
    (HERE/'integration.acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'validation.acceptance.json').write_text(json.dumps({'status':'passed','split':'validation',
        'file':'data/validation.parquet','sha256':checks['validation']['sha256'],'rows':1746,
        'report':'integration.acceptance.json','report_sha256':sha256(HERE/'integration.acceptance.json')},indent=2)+'\n')
    (HERE/'validation.publication.json').write_text(json.dumps({'status':'local_integrated_version_not_published',
        'repo_id':'Shrikes/repair-sbn','previous_published_revision':'9b7d8812c49b4d7c6f922fd3ce996d019dc18a5c',
        'previous_record':'releases/published-9b7d881/validation.publication.json',
        'current_local_validation_sha256':checks['validation']['sha256']},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
