"""Cost discovery must include attempts without walking model or output trees."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from state_guidance.control import atomic, ledger


class CostDiscoveryTests(unittest.TestCase):
    def test_all_route_archives_checks_and_evaluations_are_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path, charge in [('runs/route', 2), ('runs/route.failed-health-1', 3),
                                 ('checks/opd-update', 4), ('evaluations/test-1', 5)]:
                atomic(root / path / 'launcher_record.json',
                       {'n_gpus': 4, 'status': 'failed', 'allocated_gpu_hours': charge})
            # A checkpoint's unrelated metadata must not be visited or charged.
            nested = root / 'runs/route/checkpoints/deep/launcher_record.json'
            nested.parent.mkdir(parents=True)
            nested.write_text('not a launcher record')
            result = ledger(root)
            self.assertEqual(result['allocated_gpu_hours'], 14)
            self.assertEqual(len(result['records']), 4)

    def test_running_time_and_settled_attempt_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic(root / 'runs/a/launcher_record.json',
                   {'n_gpus': 4, 'status': 'running', 'started_unix': 100})
            atomic(root / 'runs/a.failed-1/launcher_record.json',
                   {'n_gpus': 4, 'status': 'failed', 'allocated_gpu_hours': 7})
            with patch('state_guidance.control.time.time', return_value=1000):
                result = ledger(root)
            self.assertEqual(result['allocated_gpu_hours'], 8)
            self.assertEqual(sum(x['running_estimate'] for x in result['records']), 1)

    def test_corrupt_direct_record_is_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runs/a/launcher_record.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            with self.assertRaises(KeyError):
                ledger(Path(tmp))


if __name__ == '__main__':
    unittest.main()
