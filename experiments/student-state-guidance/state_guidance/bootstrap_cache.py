"""Restore only officially verified model files; never overwrite a differing file."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    a = p.parse_args()
    a.state.parent.mkdir(parents=True, exist_ok=True)
    lock = (a.state.parent/'cache.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.time()
    expected = json.loads(a.manifest.read_text())
    state = dict(status='running', pid=os.getpid(), started_unix=start,
                 source=str(a.source), cache=str(a.cache), files=[], n_gpus=0)
    atomic(a.state, state)
    try:
        for item in expected['files']:
            source = a.source/item['model']/item['file']
            dest = a.cache/item['model']/item['file']
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                temp = dest.with_name(dest.name+'.partial')
                shutil.copyfile(source, temp)
                if temp.stat().st_size != item['bytes'] or digest(temp) != item['sha256']:
                    raise RuntimeError('Official manifest mismatch: '+str(source))
                temp.replace(dest)
            elif dest.stat().st_size != item['bytes'] or digest(dest) != item['sha256']:
                raise RuntimeError('Existing cache differs: '+str(dest))
            state['files'].append(item)
            atomic(a.state, state)
            print(json.dumps({'verified': item['model']+'/'+item['file']}), flush=True)
        state.update(status='completed', finished_unix=time.time(), seconds=time.time()-start)
        atomic(a.cache/'CACHE_MANIFEST.json', state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        atomic(a.state, state)


if __name__ == '__main__':
    main()
