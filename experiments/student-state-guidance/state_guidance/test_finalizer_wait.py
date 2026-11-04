import unittest
from pathlib import Path
from state_guidance.finalizer_wait import training_snapshot


class FakeStop:
    def __init__(self, clock, stop_after=None):
        self.clock = clock
        self.stopped = False
        self.waits = []
        self.stop_after = stop_after
    def is_set(self):
        return self.stopped
    def wait(self, duration):
        self.waits.append(duration)
        self.clock[0] += duration
        if self.stop_after is not None and len(self.waits) >= self.stop_after:
            self.stopped = True
        return self.stopped


class FinalizerWaitTests(unittest.TestCase):
    def run_read(self, reads, *, stop_after=None, reporter=None, limit=15):
        now = [0.]
        stop = FakeStop(now, stop_after)
        events = []
        def read(_):
            value = reads.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        def report(stage, **data):
            events.append((stage, data))
            if reporter:
                reporter(stage, **data)
        value = training_snapshot(Path('/shared/study_state.json'), stop, report,
                                  read_state=read, clock=lambda: now[0], max_wait_seconds=limit)
        return value, stop, events

    def test_healthy_snapshot_is_immediate(self):
        value, stop, events = self.run_read([{'status': 'running'}])
        self.assertEqual(value, {'status': 'running'})
        self.assertEqual(stop.waits, [])
        self.assertEqual(events, [])

    def test_transient_missing_snapshot_recovers(self):
        value, stop, events = self.run_read([FileNotFoundError('temporary'), OSError('stale lookup'), {'status': 'running'}])
        self.assertEqual(value['status'], 'running')
        self.assertEqual(stop.waits, [5., 5.])
        self.assertTrue(all(stage == 'waiting_training_snapshot_retry' for stage, _ in events))

    def test_invalid_json_or_schema_never_releases_evaluation(self):
        value, _, events = self.run_read([ValueError('partial JSON'), {}, {'status': 'training_complete'}])
        self.assertEqual(value['status'], 'training_complete')
        self.assertEqual(len(events), 2)

    def test_persistent_missing_state_fails_within_bound(self):
        with self.assertRaisesRegex(RuntimeError, 'bounded retry'):
            self.run_read([FileNotFoundError('missing')] * 4)

    def test_user_stop_interrupts_retry_without_more_reads(self):
        value, stop, events = self.run_read([FileNotFoundError('missing')], stop_after=1)
        self.assertIsNone(value)
        self.assertTrue(stop.stopped)
        self.assertEqual(len(events), 1)

    def test_status_write_failure_does_not_turn_transient_read_into_crash(self):
        def failed_report(*args, **kwargs):
            raise OSError('temporary status write failure')
        value, _, _ = self.run_read([FileNotFoundError('missing'), {'status': 'running'}], reporter=failed_report)
        self.assertEqual(value['status'], 'running')

    def test_programming_errors_are_not_silently_retried(self):
        with self.assertRaises(TypeError):
            self.run_read([TypeError('bug')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
