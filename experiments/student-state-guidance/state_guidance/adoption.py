"""Adopt an identity-verified launcher after a controlled scheduler handoff."""
import hashlib
import subprocess
import time
from state_guidance.control import read, same_process


class AdoptedProcess:
    def __init__(self, saved, record, matches=same_process):
        self.saved, self.record, self.matches = saved, record, matches
        self.pid = saved['pid']

    def poll(self):
        if self.matches(self.saved):
            return None
        record = read(self.record)
        if record['status'] == 'completed':
            return 0
        if record['status'] == 'failed' and 'allocated_gpu_hours' in record:
            return record.get('return_code') or 1
        raise RuntimeError('Adopted launcher died without a settled record: ' + str(self.record))

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(['adopted', str(self.pid)], timeout)
            time.sleep(.1)


def adoption_items(root, manifest, all_phases, slots, matches=same_process):
    assert manifest['schema'] == 1 and manifest['reason'] == 'explicit_user_scope_revision'
    assert not matches(manifest['previous_coordinator']), 'Previous scheduler must be stopped first'
    by_name = {p['name']: p for p in all_phases}
    active = {}
    for item in manifest['retained']:
        slot, name = item['slot'], item['name']
        assert slot in slots and slot not in active and name in by_name
        phase = by_name[name]
        folder = root / 'runs' / name
        record_path = folder / 'launcher_record.json'
        record = read(record_path)
        assert record['started_unix'] == item['attempt_started_unix']
        for key in ['launcher_identity', 'child_identity']:
            assert all(record[key][field] == item[key][field] for field in ['pid', 'start_ticks'])
        assert (record['task'], record['mode'], record['seed'], record['start_step'], record['end_step']) == (
            phase['task'], phase['mode'], phase['seed'], phase['start'], phase['start'] + phase['steps'])
        assert record['gpu_ids'] == list(map(int, slots[slot].split(',')))
        assert hashlib.sha256((folder / 'request.json').read_bytes()).hexdigest() == item['request_sha256']
        assert record['status'] in ('running', 'completed')
        if record['status'] == 'running':
            assert matches(item['launcher_identity']) and matches(item['child_identity'])
        active[slot] = {'phase': phase, 'process': AdoptedProcess(item['launcher_identity'], record_path, matches),
                        'identity': item['launcher_identity'], 'check': False, 'started': record['started_unix'],
                        'log': (root / 'logs' / (name + '.launcher.log')).open('a')}
    return active
