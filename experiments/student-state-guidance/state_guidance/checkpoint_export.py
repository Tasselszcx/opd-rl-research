"""Copy a completed local checkpoint to an unpublished shared staging directory."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

def export_verified(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    descriptors=[];jobs=[]
    try:
        for path in sorted(source.rglob('*')):
            relative=path.relative_to(source);target=destination/relative
            if path.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            assert path.is_file(), str(path)
            target.parent.mkdir(parents=True,exist_ok=True)
            inp=os.open(path,os.O_RDONLY);out=os.open(target,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
            descriptors.extend([inp,out]);size=os.fstat(inp).st_size;os.ftruncate(out,size)
            for offset in range(0,size,16*1024**2):jobs.append((str(relative),inp,out,offset,min(16*1024**2,size-offset)))
        def copy_chunk(job):
            name,inp,out,offset,size=job;data=os.pread(inp,size,offset);assert len(data)==size
            written=0
            while written<size:
                n=os.pwrite(out,data[written:],offset+written);assert n>0;written+=n
            digest=hashlib.sha256(data).hexdigest()
            assert hashlib.sha256(os.pread(out,size,offset)).hexdigest()==digest
            return {'path':name,'offset':offset,'bytes':size,'sha256':digest}
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:chunks=list(pool.map(copy_chunk,jobs))
        for fd in descriptors[1::2]:os.fsync(fd)
        (destination/'EXPORT_VERIFIED.json').write_text(json.dumps({'status':'verified','chunks':chunks,'bytes':sum(x['bytes'] for x in chunks)},indent=2)+'\n')
        return sum(x['bytes'] for x in chunks)
    finally:
        for fd in descriptors:os.close(fd)
