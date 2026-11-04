"""Account for every real training/evaluation attempt, including archived failures."""
from pathlib import Path
import json

def charged_gpu_hours(root, migration_root):
    paths=list(Path(root).rglob('launcher_record.json'))
    paths+=list((Path(migration_root)/'validation/smoke').rglob('launcher_record.json'))
    events=[json.loads(p.read_text()) for p in paths]
    return sum(e.get('allocated_gpu_hours',0) for e in events)
