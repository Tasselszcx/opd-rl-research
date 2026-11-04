import json,tempfile,unittest
from pathlib import Path
from hydra.core.override_parser.overrides_parser import OverridesParser
from state_guidance.eval_worker import prompt_suffix

class PromptTests(unittest.TestCase):
    def test_new_campaign_eval_matches_corrected_training_command(self):
        from state_guidance.run_phase import command
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,cfg,_,_=command(root,'kk','plain',42,root/'runs/branch',root/'global_step_30')
            actual=OverridesParser.create().parse_override(next(a for a in args if a.startswith('++data.cot_suffix='))).value()
            self.assertEqual(actual,cfg['data.cot_suffix']);self.assertTrue(actual.endswith('\n'))
            self.assertFalse(actual.endswith('\\n'))
            (root/'PROMPT_FORMAT.json').write_text(json.dumps({'status':'frozen_training_runtime','cot_suffix_by_task':{'kk':actual}}))
            self.assertEqual(prompt_suffix(root,'kk').encode(),actual.encode())
    def test_eval_matches_resolved_training_argument(self):
        arg='++data.cot_suffix='+json.dumps("Let's think step by step.\n")
        actual=OverridesParser.create().parse_override(arg).value()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'PROMPT_FORMAT.json').write_text(json.dumps({'status':'frozen_training_runtime','cot_suffix_by_task':{'kk':actual}}))
            self.assertEqual(prompt_suffix(r,'kk').encode(),actual.encode())
    def test_repaired_run_requires_format_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'PROTOCOL_AMENDMENT.json').write_text('{}')
            with self.assertRaises(AssertionError):prompt_suffix(r,'kk')
    def test_math_has_no_extra_suffix(self):
        with tempfile.TemporaryDirectory() as d:self.assertEqual(prompt_suffix(Path(d),'gsm8k'),'')

if __name__=='__main__':unittest.main()
