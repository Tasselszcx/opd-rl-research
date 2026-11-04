"""Standard-library-only control primitives for the frozen G2 diagnostic."""
from pathlib import Path
import hashlib
import json
import math
import os
import signal
import tempfile
import time

ENDPOINTS = ('plain-opd', 'plain-rl', 'key_step-opd', 'key_step-rl',
             'full_solution-opd', 'full_solution-rl')
# The user explicitly removed the G2 budget ceiling before deployment.
# Keep the JSON fields so provenance remains explicit rather than implicit.
CAP = None
THRESHOLD = None
GPUS = 4
CLEANUP_RESERVE_SECONDS = 120


def read_json(path, attempts=5, delay=0.4):
    """BeeGFS may transiently fail after an atomic rename; never treat that as absence."""
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_plan(plan):
    required = {'gpu_hour_cap', 'termination_threshold', 'questions', 'samples',
                'max_tokens', 'baseline_root', 'endpoints', 'frozen_files',
                'source_commit', 'diagnostic'}
    if not required <= plan.keys():
        raise ValueError('Incomplete frozen plan')
    expected = {'gpu_hour_cap': CAP, 'termination_threshold': THRESHOLD,
                'questions': 512, 'samples': 4, 'max_tokens': 4096, 'diagnostic': True}
    for key, value in expected.items():
        if plan[key] != value:
            raise ValueError('Frozen protocol mismatch: ' + key)
    if tuple(item['name'] for item in plan['endpoints']) != ENDPOINTS:
        raise ValueError('Endpoint sequence differs from the frozen six endpoints')
    if any(not Path(item['model']).is_absolute() for item in plan['endpoints']):
        raise ValueError('Model paths must be explicit absolute paths')
    if not plan['frozen_files'] or len(plan['source_commit']) != 40:
        raise ValueError('Frozen source/files missing')
    paths = [item['path'] for item in plan['frozen_files']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate frozen file entries')
    for item in plan['frozen_files']:
        if not Path(item['path']).is_absolute() or len(item['sha256']) != 64:
            raise ValueError('Invalid frozen file identity')


def verify_frozen_files(plan):
    checked = []
    for item in plan['frozen_files']:
        digest = sha256(item['path'])
        if digest != item['sha256']:
            raise RuntimeError('Frozen file changed: ' + item['path'])
        checked.append({'path': item['path'], 'sha256': digest})
    return checked


def process_identity(pid, expected_script=None, proc_root=Path('/proc')):
    """Return Linux start ticks and argv; a reused PID cannot match an old identity."""
    proc = Path(proc_root) / str(pid)
    try:
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
    except FileNotFoundError:
        return None
    if stat[0] == 'Z':
        return None
    if expected_script is not None and str(expected_script) not in argv:
        raise RuntimeError('PID does not identify the expected task: ' + str(pid))
    return {'pid': int(pid), 'start_ticks': int(stat[19]), 'argv': argv,
            'session': int(stat[3]), 'pgrp': int(stat[2])}


def same_process(expected, expected_script=None):
    current = process_identity(expected['pid'], expected_script)
    return current is not None and current['start_ticks'] == expected['start_ticks']


def stop_verified_owner(identity, script):
    if same_process(identity, script):
        os.kill(identity['pid'], signal.SIGTERM)
        return True
    return False


def signal_owned_group(identity, signum):
    """Only a session created for one recorded worker can be signalled."""
    current = process_identity(identity['pid'])
    if current is not None:
        if current['start_ticks'] != identity['start_ticks']:
            raise RuntimeError('Worker PID reused; refusing to signal its group')
        if current['session'] != identity['pid'] or current['pgrp'] != identity['pid']:
            raise RuntimeError('Worker is not the recorded session leader')
    else:
        # A failed leader can leave vLLM descendants. Its session cannot be reused
        # while such descendants survive; verify their start times before cleanup.
        members = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            except (FileNotFoundError, PermissionError):
                continue
            if fields[0] != 'Z' and int(fields[3]) == identity['pid'] and int(fields[2]) == identity['pid']:
                members.append({'start_ticks': int(fields[19])})
        if not members:
            return False
        if any(member['start_ticks'] < identity['start_ticks'] for member in members):
            raise RuntimeError('Worker session identity mismatch')
    try:
        os.killpg(identity['pid'], signum)
    except ProcessLookupError:
        return False
    return True


def _record_paths(folder):
    """Unlike Path.rglob, do not swallow filesystem traversal errors."""
    result = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                result.extend(_record_paths(entry.path))
            elif entry.name == 'launcher_record.json':
                result.append(Path(entry.path))
    return result


def record_cost(record, now):
    if record.get('n_gpus') != GPUS or record.get('status') not in ('running', 'completed', 'failed'):
        raise ValueError('Invalid allocation record')
    start = float(record['started_unix'])
    if not math.isfinite(start) or start > now + 10:
        raise ValueError('Invalid allocation start time')
    if record['status'] == 'running':
        if 'allocated_gpu_hours' in record:
            raise ValueError('Running record must not masquerade as settled cost')
        return GPUS * max(0, now - start) / 3600
    hours = float(record['allocated_gpu_hours'])
    elapsed = float(record['wall_seconds'])
    finished = float(record['finished_unix'])
    if not all(math.isfinite(x) and x >= 0 for x in (hours, elapsed, finished)):
        raise ValueError('Invalid settled allocation')
    if abs(hours - GPUS * elapsed / 3600) > 1e-8 or abs(finished - start - elapsed) > 0.1:
        raise ValueError('Inconsistent settled allocation')
    return hours


class CostLedger:
    """Every observed attempt remains required, including archived failed attempts."""
    def __init__(self, root):
        self.folder = Path(root) / 'evaluations'
        self.known = {}

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        records = {}
        for path in sorted(_record_paths(self.folder)):
            record = read_json(path)
            identity = record['attempt_id']
            if not identity or identity in records:
                raise ValueError('Missing or duplicate allocation attempt identity')
            records[identity] = (path, record)
        missing = set(self.known) - set(records)
        if missing:
            raise OSError('Previously observed allocation disappeared: ' + ', '.join(sorted(missing)))
        for identity, (path, record) in records.items():
            stable = (record['started_unix'], record['n_gpus'])
            if identity in self.known and self.known[identity] != stable:
                raise ValueError('Allocation identity was rewritten: ' + str(path))
            self.known[identity] = stable
        return sum(record_cost(record, now) for _, record in records.values()), len(records)


def phase_timeout(charged, cap=CAP):
    if cap is None:
        return None
    seconds = (cap - charged) * 3600 / GPUS - CLEANUP_RESERVE_SECONDS
    if seconds <= 0:
        raise RuntimeError('Budget exhausted including required cleanup reserve')
    return seconds


def validate_coverage(folder, entries, model=None):
    folder = Path(folder)
    if len(entries) != 512 or {x['index'] for x in entries} != set(range(512)):
        raise ValueError('Expected the entire frozen 512-question test')
    hashes = {x['index']: x['question_sha256'] for x in entries}
    expected_paths = {folder / f'rank_{rank}.jsonl' for rank in range(GPUS)}
    if set(folder.glob('rank_*.jsonl')) != expected_paths:
        raise ValueError('Expected exactly four evaluation output shards')
    seen = set()
    for rank in range(GPUS):
        done = read_json(folder / f'rank_{rank}.done.json')
        if (done.get('status') != 'completed' or done.get('rank') != rank
                or done.get('rows') != 512 or done.get('samples') != 4):
            raise ValueError('Invalid worker completion marker')
        if model is not None and done.get('model') != str(model):
            raise ValueError('Worker model differs from the fixed endpoint')
        rank_seen = set()
        with (folder / f'rank_{rank}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                key = (row['index'], row['sample'])
                if key in seen or row['index'] not in hashes or row['index'] % GPUS != rank:
                    raise ValueError('Duplicate, unexpected, or incorrectly sharded question')
                if row['sample'] not in range(4) or row['sampling_seed'] != 42 + row['sample']:
                    raise ValueError('Sampling protocol mismatch')
                if row['suite'] != 'pilot_test' or row['question_sha256'] != hashes[row['index']]:
                    raise ValueError('Frozen test identity mismatch')
                if row['score'] not in (0, 1) or not 0 <= row['response_tokens'] <= 4096:
                    raise ValueError('Invalid evaluation result')
                seen.add(key)
                rank_seen.add(key)
        if len(rank_seen) != 512:
            raise ValueError('Incomplete evaluation shard')
    if seen != {(i, j) for i in range(512) for j in range(4)}:
        raise ValueError('Incomplete 512 x 4 evaluation coverage')
    return len(seen)
