"""Stage the validated train/dev release with an explicit file allowlist."""
import argparse
import json
from pathlib import Path
import shutil
from .build import HERE,ROOT,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((HERE/'integration.acceptance.json').read_text())
    if report['status']!='passed':raise ValueError('Integrated acceptance has not passed')
    for path,digest in report['input_sha256'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Accepted input changed: '+path)
    for split,meta in report['splits'].items():
        if sha256(HERE/'data'/f'{split}.parquet')!=meta['sha256']:
            raise ValueError('Accepted Parquet changed: '+split)
    stage=args.stage_dir.resolve();stage.mkdir(parents=True,exist_ok=True)
    files=['README.md','SCHEMA.md','data/train.parquet','data/validation.parquet',
           'manifest.json','validation.manifest.json','integration.acceptance.json',
           'validation.acceptance.json','fluent_group.acceptance.json']
    existing={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
    if existing-set(files)-{'stage.manifest.json'}:raise ValueError('Unexpected files in stage')
    hashes={}
    for name in files:
        dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HERE/name,dest)
        hashes[name]=sha256(dest)
        if hashes[name]!=sha256(HERE/name):raise ValueError('Staging changed: '+name)
    manifest={'status':'staged_locally_not_uploaded','repo_id':'Shrikes/repair-sbn',
              'dataset_version':'fluent-groups-v1','files_sha256':hashes}
    (stage/'stage.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'stage':str(stage),'status':manifest['status'],'files':len(files)},indent=2))


if __name__=='__main__':main()
