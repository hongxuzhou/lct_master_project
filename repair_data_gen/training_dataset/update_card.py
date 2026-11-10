"""Publish reader-facing README/schema changes without altering dataset files."""
import json
from pathlib import Path
from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
from .build import HERE,sha256


def main():
    data_record=json.loads((HERE/'publication.json').read_text())
    if data_record['status']!='published_and_verified':raise ValueError('Data release is not verified')
    repo=data_record['repo_id'];docs=['README.md','SCHEMA.md']
    hashes={name:sha256(HERE/name) for name in docs}
    expected={**data_record['files_sha256'],**hashes}
    for name,digest in data_record['files_sha256'].items():
        if name not in docs and sha256(HERE/name)!=digest:raise ValueError('Non-document input changed: '+name)
    record_path=HERE/'documentation.publication.json';pending_path=HERE/'documentation.publication.pending.json'
    previous=json.loads(record_path.read_text()) if record_path.exists() else None
    pending=json.loads(pending_path.read_text()) if pending_path.exists() else None
    api=HfApi()
    if pending and pending['status']=='commit_created_readback_pending':
        if pending['documentation_sha256']!=hashes:raise ValueError('Pending documentation differs from local files')
        revision=pending['revision'];parent=pending['parent_revision']
    elif previous and previous['documentation_sha256']==hashes:
        revision=previous['revision'];parent=previous['parent_revision']
    else:
        parent=previous['revision'] if previous else data_record['revision']
        head=api.repo_info(repo,repo_type='dataset').sha
        if head!=parent:raise ValueError('Hub head changed; refusing to overwrite another update: '+head)
        commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(HERE/name)) for name in docs],
            commit_message='Rewrite dataset card and schema for dataset users',
            commit_description='Describe dataset scope, per-split counts and proportions, actual examples and field meanings. Remove internal experiment and update-log terminology. Data files are unchanged.')
        revision=commit.oid
        pending={'status':'commit_created_readback_pending','revision':revision,'parent_revision':parent,'documentation_sha256':hashes}
        pending_path.write_text(json.dumps(pending,indent=2)+'\n')
    for name,digest in expected.items():
        remote=Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision))
        if sha256(remote)!=digest:raise ValueError('Readback mismatch: '+name)
    record={'status':'published_and_verified','repo_id':repo,'revision':revision,'parent_revision':parent,
        'data_release_revision':data_record['revision'],'documentation_sha256':hashes,
        'commit_url':f'https://huggingface.co/datasets/{repo}/commit/{revision}',
        'verified_files':len(expected),'data_files_unchanged':True,
        'publisher_sha256':sha256(Path(__file__))}
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    pending_path.write_text(json.dumps({**record,'status':'readback_completed'},indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
