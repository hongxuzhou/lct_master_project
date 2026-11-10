"""Publish the staged, accepted dataset atomically and verify exact readback."""
import argparse
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
from .build import HERE,ROOT,sha256

REPO='Shrikes/repair-sbn'
PARENT='9b7d8812c49b4d7c6f922fd3ce996d019dc18a5c'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir',type=Path,required=True)
    args=parser.parse_args();stage=args.stage_dir.resolve()
    staged=json.loads((stage/'stage.manifest.json').read_text())
    for name,digest in staged['files_sha256'].items():
        if sha256(stage/name)!=digest:raise ValueError('Staged input changed: '+name)
    report=json.loads((stage/'integration.acceptance.json').read_text())
    if report['status']!='passed':raise ValueError('Acceptance not passed')
    for path,digest in report['input_sha256'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Accepted input changed: '+path)
    api=HfApi();pending=HERE/'publication.pending.json'
    if pending.exists():
        state=json.loads(pending.read_text())
        if state['files_sha256']!=staged['files_sha256']:raise ValueError('Pending publication differs from stage')
        revision=state['revision']
    else:
        info=api.repo_info(REPO,repo_type='dataset')
        if info.sha!=PARENT:raise ValueError('Hub parent changed; refusing to overwrite another update: '+info.sha)
        # Verify exact old split bytes, not just a remembered revision string.
        snapshot=HERE/'releases/published-9b7d881'
        for split in ['train','validation']:
            remote=Path(hf_hub_download(REPO,f'data/{split}.parquet',repo_type='dataset',revision=PARENT))
            if sha256(remote)!=sha256(snapshot/'data'/f'{split}.parquet'):
                raise ValueError('Hub baseline differs from the preserved split: '+split)
        commit=api.create_commit(repo_id=REPO,repo_type='dataset',parent_commit=PARENT,
            operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(stage/name))
                        for name in staged['files_sha256']],
            commit_message='Add reviewed fluent split-reference supervision to train and validation',
            commit_description='Append 200 train and 40 validation examples; preserve all prior field values and order. Add explicit provenance, construction and family fields, shared condition views, and acceptance records.')
        revision=commit.oid
        state={'status':'commit_created_readback_pending','repo_id':REPO,'revision':revision,
               'parent_revision':PARENT,'files_sha256':staged['files_sha256']}
        pending.write_text(json.dumps(state,indent=2)+'\n')
    import pyarrow.parquet as pq
    downloaded={}
    for name,digest in staged['files_sha256'].items():
        path=Path(hf_hub_download(REPO,name,repo_type='dataset',revision=revision))
        if sha256(path)!=digest:raise ValueError('Remote readback differs: '+name)
        downloaded[name]=path
    for split,expected in [('train',14222),('validation',1746)]:
        remote=pq.read_table(downloaded[f'data/{split}.parquet']);local=pq.read_table(HERE/'data'/f'{split}.parquet')
        if remote.num_rows!=expected or not remote.equals(local,check_metadata=True):
            raise ValueError('Remote schema or rows differ: '+split)
    record={**state,'status':'published_and_verified','commit_url':f'https://huggingface.co/datasets/{REPO}/commit/{revision}',
            'dataset_version':'fluent-groups-v1','schema_version':2,'rows':{'train':14222,'validation':1746},
            'verification':'all staged file bytes and complete Parquet schemas/rows match local',
            'source_acceptance':'fluent_group.acceptance.json','publisher_sha256':sha256(Path(__file__))}
    (HERE/'publication.json').write_text(json.dumps(record,indent=2)+'\n')
    (HERE/'validation.publication.json').write_text(json.dumps(record,indent=2)+'\n')
    protocol=json.loads((ROOT/'model/lora/protocols/fluent_groups_v1.json').read_text())
    protocol['dataset'].update(revision=revision,publication_status='published_and_verified')
    protocol['status']='verified_published_dataset; model settings remain provisional'
    (ROOT/'model/lora/protocols/fluent_groups_v1.published.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'revision':revision,'commit_url':record['commit_url'],'rows':record['rows']},indent=2))


if __name__=='__main__':main()
