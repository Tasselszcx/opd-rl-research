import json,tempfile,unittest
from pathlib import Path
from hydra.core.override_parser.overrides_parser import OverridesParser
from state_guidance.run_phase import command,hydra_value

class HydraValueTests(unittest.TestCase):
    def parse(self,value):
        return OverridesParser.create().parse_override('++data.test='+hydra_value(value)).value()
    def test_real_newline_round_trip(self):
        text="Let's think step by step.\n"
        self.assertEqual(self.parse(text),text)
        self.assertNotEqual(OverridesParser.create().parse_override('++data.test='+json.dumps(text)).value(),text)
    def test_unicode_quotes_backslash_and_tab_round_trip(self):
        for text in ['\u6d4b\u8bd5', 'a"b', "a'b", 'literal\\n', 'tab\tend', 'C:\\folder\\name']:
            with self.subTest(text=text):self.assertEqual(self.parse(text),text)
    def test_scalar_and_list_types_are_preserved(self):
        for value in [None,True,False,42,1e-6,['console',30,90],['\u6d4b\u8bd5','a\nb']]:
            with self.subTest(value=value):self.assertEqual(self.parse(value),value)
    def test_actual_training_command_has_the_intended_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);args,cfg,_,_=command(r,'kk','answer_only',42,r/'runs/probe',r/'global_step_30')
            arg=next(x for x in args if x.startswith('++data.cot_suffix='))
            self.assertEqual(OverridesParser.create().parse_override(arg).value(),cfg['data.cot_suffix'])
            self.assertTrue(cfg['data.cot_suffix'].endswith('\n'))
    def test_nul_is_rejected(self):
        with self.assertRaises(ValueError):hydra_value('a\x00b')

if __name__=='__main__':unittest.main()
