"""Atomic state, identity checks, and physical allocation accounting for G3."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def read(path):
    for i in range(4):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):
            if i==3:raise
            time.sleep(.25)


def identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();tail=text[text.rfind(')')+2:].split()
        return {'pid':pid,'start_ticks':int(tail[19]),'pgrp':int(tail[2]),'session':int(tail[3]),
                'cwd':str(Path(f'/proc/{pid}/cwd').resolve()),
                'cmdline':Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
    except OSError:return None


def same_process(saved):
    current=identity(saved['pid'])
    return bool(current and all(current[k]==saved[k] for k in ['pid','start_ticks','pgrp','session','cwd'])
                and current['cmdline']==saved['cmdline'])


def stop_owned(saved):
    if same_process(saved):os.kill(saved['pid'],signal.SIGTERM)


def ledger(root):
    entries=[];now=time.time()
    for subdir in ['runs','checks','evaluations']:
        base=Path(root)/subdir
        if not base.exists():continue
        # Launchers live directly under each route/job directory. Walking their
        # checkpoints and validation trees can stall shared-storage heartbeats.
        for path in sorted(base.glob('*/launcher_record.json')):
            rec=read(path);n=rec['n_gpus']
            charge=rec.get('allocated_gpu_hours')
            if charge is None:
                assert rec['status'] in ('starting','running'),str(path)
                charge=n*max(0,now-rec['started_unix'])/3600
            entries.append({'record':str(path.relative_to(root)),'gpu_hours':charge,
                            'running_estimate':rec['status'] in ('starting','running')})
    history=Path(root)/'HISTORICAL_COST.json'
    prior=read(history)['allocated_gpu_hours'] if history.exists() else 0.
    assert isinstance(prior,(int,float)) and 0<=prior<float('inf')
    return {'allocated_gpu_hours':prior+sum(x['gpu_hours'] for x in entries),'records':entries,
            'historical_allocated_gpu_hours':prior,
            'updated_unix':now,'budget_cap':None,'historical_study_cost_included':history.exists()}
