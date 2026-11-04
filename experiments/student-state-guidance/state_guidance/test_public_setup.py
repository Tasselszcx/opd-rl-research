import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from state_guidance.public_setup import initialize, metadata, fetch_file, check_model_cache


class PublicSetupTests(unittest.TestCase):
    def test_fresh_preset_is_eighteen_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = initialize(Path(tmp))
            plan, scope = metadata(root)
            self.assertEqual(scope['training_routes'], 18)
            self.assertEqual(scope['training_seeds'], [42])
            self.assertEqual(plan['global_endpoints'], [180, 240])

    def test_conflicting_scope_does_not_partially_initialize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ACTIVE_SCOPE.json').write_text('existing experiment')
            with self.assertRaises(RuntimeError):
                initialize(root)
            self.assertFalse((root / 'study_plan.json').exists())
            self.assertEqual((root / 'ACTIVE_SCOPE.json').read_text(), 'existing experiment')

    def test_existing_execution_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'study_state.json').write_text('retained')
            with self.assertRaises(RuntimeError):
                initialize(root)
            self.assertFalse((root / 'study_plan.json').exists())

    def test_changed_plan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = initialize(Path(tmp))
            with (root / 'study_plan.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaises(ValueError):
                metadata(root)

    def test_bad_download_never_becomes_final_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            with self.assertRaises(RuntimeError):
                fetch_file('https://example.invalid/input', path, '0' * 64,
                           opener=lambda *a, **k: io.BytesIO(b'wrong'))
            self.assertFalse(path.exists())
            self.assertEqual(len(list(Path(tmp).glob('*.partial'))), 1)

    def test_corrupt_cached_input_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            path.write_bytes(b'old')
            with self.assertRaises(RuntimeError):
                fetch_file('unused', path, hashlib.sha256(b'new').hexdigest())
            self.assertEqual(path.read_bytes(), b'old')

    def test_model_rehash_catches_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model').mkdir()
            (root / 'model/weights').write_bytes(b'bad')
            expected = [{'model': 'model', 'file': 'weights', 'bytes': 3,
                         'sha256': hashlib.sha256(b'yes').hexdigest()}]
            (root / 'CACHE_MANIFEST.json').write_text(json.dumps({'status': 'completed', 'files': expected}))
            with self.assertRaises(ValueError):
                check_model_cache(root, expected, rehash=True)


if __name__ == '__main__':
    unittest.main()
