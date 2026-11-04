"""CPU-only coverage, scoring, and fixed-matrix statistics checks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
from state_guidance.analyze import pass_curve,bootstrap
from state_guidance.coordinator import phases
from state_guidance.finalize import jobs,validate,training_finished
from state_guidance.data import sha_text,score_answer
from state_guidance.eval_worker import suite_settings


class FinalizerTests(unittest.TestCase):
    def test_complete_training_scope(self):
        p=phases(Path('/tmp/unit'));self.assertEqual(len(p),132)
        self.assertEqual(sum(x['resume'] is not None for x in p),120)
        self.assertEqual({x['seed'] for x in p},{42,43,44})
    def test_eval_scope_and_transfer(self):
        j=jobs(Path('/tmp/unit'));self.assertEqual(len(j),144)
        self.assertEqual(sum(x['samples']==16 for x in j),24)
        self.assertEqual(sum('math500' in x['suites'] for x in j),30)
    def test_diagnostic_samples_independent(self):
        self.assertEqual(suite_settings(16,'diagnostic'),(16,2000))
        self.assertEqual(suite_settings(32,'diagnostic'),(16,3000))
        self.assertEqual(suite_settings(32,'test'),(32,42))
    def test_curve_all_correct(self):self.assertTrue(all(v==1 for v in pass_curve([32,32],32).values()))
    def test_curve_all_wrong(self):self.assertTrue(all(v==0 for v in pass_curve([0,0],32).values()))
    def test_curve_count_formula(self):
        curve=pass_curve([16],32);self.assertEqual(curve['1'],.5)
        self.assertAlmostEqual(curve['2'],1-(16*15)/(32*31));self.assertEqual(curve['32'],1)
    def test_paired_zero_interval(self):self.assertEqual(bootstrap(np.zeros(8),draws=500)['paired_question_ci95'],[0,0])
    def test_paired_fixed_difference(self):self.assertEqual(bootstrap(np.ones(8)*.25,draws=500)['difference'],.25)
    def test_math_equivalence(self):
        entry={'task':'math500','answer':r'\frac{1}{2}'}
        self.assertEqual(score_answer('0.5',entry),1);self.assertEqual(score_answer('1',entry),0)
    def test_never_recover_user_stopped(self):
        self.assertFalse(training_finished(Path('/tmp/unit'),{'status':'stopped'}))
    def test_never_recover_missing_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(training_finished(Path(tmp),{'status':'failed','error':"ValueError('min() arg is an empty sequence')"}))
    def test_coverage_rejects_missing_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data/gsm8k';data.mkdir(parents=True);out=root/'eval';out.mkdir()
            entries=[{'index':i,'task':'gsm8k','question':str(i),'question_sha256':sha_text(str(i)),
                      'answer':'12','metadata':{}} for i in range(8)]
            (data/'test.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
            for rank in range(4):
                rows=[]
                for entry in entries:
                    if entry['index']%4!=rank:continue
                    for sample in range(2):rows.append({'suite':'test','index':entry['index'],'sample':sample,
                         'question_sha256':entry['question_sha256'],'sampling_seed':42+sample,'score':1.,
                         'predicted_answer':'12','token_ids':[1],'response_tokens':1})
                (out/f'rank_{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
                (out/f'rank_{rank}.done.json').write_text(json.dumps({'status':'completed'}))
            job={'task':'gsm8k','suites':['test'],'samples':2}
            self.assertEqual(validate(root,job,out,rescore=True),16)
            path=out/'rank_0.jsonl';lines=path.read_text().splitlines();path.write_text('\n'.join(lines[:-1])+'\n')
            with self.assertRaises(AssertionError):validate(root,job,out)


if __name__=='__main__':unittest.main(verbosity=2)
