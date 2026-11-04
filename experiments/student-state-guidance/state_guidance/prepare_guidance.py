"""Build and verify all fixed training references before condition comparisons."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from state_guidance.data import read_rows
from state_guidance.guides import build_guidance


def make(row):return {**row,**build_guidance(row)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();folder=a.root/'data'/a.task;dest=folder/'guidance.jsonl'
    if dest.exists():raise RuntimeError('Existing references retained; do not overwrite')
    rows=read_rows(folder/'train.jsonl');start=time.time()
    part=dest.with_suffix('.partial')
    with ProcessPoolExecutor(max_workers=a.workers) as pool,part.open('w') as f:
        for i,row in enumerate(pool.map(make,rows,chunksize=8)):
            f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
            if (i+1)%1024==0:print(json.dumps({'task':a.task,'verified_references':i+1}),flush=True)
    part.replace(dest)
    manifest={'task':a.task,'count':len(rows),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
              'train_sha256':hashlib.sha256((folder/'train.jsonl').read_bytes()).hexdigest(),
              'seconds':time.time()-start,'teacher_only':True,'errors':'fixed verified-error rules; answer-only fallback'}
    (folder/'guidance_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
