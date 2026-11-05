import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from state_guidance.public_setup import initialize, metadata, fetch_file, check_model_cache


class PublicSetupTests(unittest.TestCase):
    def test_fresh_preset_is_eight_comparisons_with_repaired_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = initialize(Path(tmp))
            plan, scope = metadata(root)
            self.assertEqual(scope['training_routes'], 9)
            self.assertEqual(scope['tasks'], ['kk'])
            self.assertEqual(scope['formal_branches'], 8)
            self.assertEqual(scope['training_seeds'], [42])
            self.assertEqual(plan['global_endpoints'], [180, 240])
            self.assertFalse(scope['legacy_reference_allowed'])
            self.assertTrue((root/'CLEAN_REFERENCE_REQUIREMENT.json').exists())
            self.assertEqual(json.loads((root/'PROMPT_FORMAT.json').read_text())['cot_suffix_by_task']['kk'],"Let's think step by step.\n")
            from state_guidance.coordinator import phases
            from state_guidance.run_phase import command
            routes=phases(root)
            self.assertEqual([p['start'] for p in routes],[0]+[30]*4+[90]*4)
            for phase in routes:
                resume=Path(phase['resume']) if phase['resume'] else None
                _,cfg,_,_=command(root,'kk',phase['mode'],42,root/'runs'/phase['name'],resume)
                self.assertEqual(cfg['algorithm.opd_eos_mode'],'semantic')
                self.assertEqual(cfg['data.cot_suffix'],"Let's think step by step.\n")
                self.assertFalse(cfg.get('algorithm.kk_teacher_output_contract',False))

    def test_clean_initialization_is_idempotent_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=initialize(Path(tmp));before=(root/'CLEAN_REFERENCE_REQUIREMENT.json').read_bytes()
            initialize(root)
            self.assertEqual((root/'CLEAN_REFERENCE_REQUIREMENT.json').read_bytes(),before)
            metadata(root)

    def test_public_reference_origin_uses_portable_source_identity(self):
        from state_guidance.clean_reference import check_reference_origin
        from state_guidance.run_phase import command
        with tempfile.TemporaryDirectory() as tmp:
            root=initialize(Path(tmp));rule=json.loads((root/'CLEAN_REFERENCE_REQUIREMENT.json').read_text())
            folder=root/'runs/kk-s42-reference';folder.mkdir(parents=True)
            _,cfg,_,_=command(root,'kk','plain',42,folder)
            (folder/'request.json').write_text(json.dumps({'config':cfg,'source_resume':None}))
            record={'start_step':0,'end_step':150,'started_unix':rule['approved_unix']+1,
                    'source_path':str(Path(__file__).resolve().parents[1])}
            (folder/'launcher_record.json').write_text(json.dumps(record));check_reference_origin(root)
            record['reuse_only']=True;(folder/'launcher_record.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(AssertionError,'legacy'):check_reference_origin(root)

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
