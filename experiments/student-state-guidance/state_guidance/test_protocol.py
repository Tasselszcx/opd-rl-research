"""CPU-only protocol checks. They are not experimental results."""
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
from state_guidance.data import arithmetic,extract_final,score_answer,sha_text,FrozenTaskDataset
from state_guidance.guides import build_guidance,error_hint,reference_text
from state_guidance.prepare_data import generate,canonical
from state_guidance.run_phase import command


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kk=generate(('kk',5000000));cls.kk['index']=0
        cls.zebra=generate(('zebra',10000000));cls.zebra['index']=0
        cls.cd=generate(('countdown',15000000));cls.cd['index']=0
        cls.math={'index':0,'task':'gsm8k','question':'A synthetic arithmetic check.','answer':'12',
                  'metadata':{'reference_solution':'Four groups of three give <<4*3=12>>12.'}}
        cls.math['question_sha256']=sha_text(cls.math['question'])

    def test_safe_arithmetic(self):self.assertEqual(arithmetic('(3+1)*3'),12)
    def test_exact_rational(self):self.assertEqual(arithmetic('1/3+2/3'),1)
    def test_reject_code(self):
        with self.assertRaises(ValueError):arithmetic('__import__("os").system("true")')
    def test_reject_large_numbers(self):
        with self.assertRaises(ValueError):arithmetic('9'*40)
    def test_numeric_scoring(self):
        self.assertEqual(score_answer('12.0',self.math),1);self.assertEqual(score_answer('13',self.math),0)
    def test_no_arithmetic_expression_as_final_numeric(self):self.assertEqual(score_answer('3*4',self.math),0)
    def test_extract_last_answer(self):self.assertEqual(extract_final('<answer>1</answer><answer>12</answer>','gsm8k'),'12')
    def test_extract_math_fallback(self):self.assertEqual(extract_final('work\n#### 12','gsm8k'),'12')
    def test_logic_requires_answer_format(self):self.assertEqual(extract_final('some person','zebra'),'')
    def test_oracle_scores_all_tasks(self):
        for row in [self.kk,self.zebra,self.cd,self.math]:self.assertEqual(score_answer(row['answer'],row),1)
    def test_wrong_empty_scores(self):
        for row in [self.kk,self.zebra,self.cd,self.math]:self.assertEqual(score_answer('',row),0)
    def test_kk_full_certificate(self):
        g=build_guidance(self.kk);checks=g['proof']['assignment_checks']
        self.assertEqual(len(checks),64);self.assertEqual(sum(x['first_violated_speaker'] is None for x in checks),1)
    def test_zebra_full_certificate(self):
        g=build_guidance(self.zebra);self.assertTrue(g['proof']['all_verified_by_sat'])
        self.assertEqual(len(g['proof']['entailed_placements']),16)
    def test_countdown_reference(self):self.assertTrue(build_guidance(self.cd)['proof']['exact_rational_steps'])
    def test_math_key_verified(self):self.assertTrue(build_guidance(self.math)['proof']['key_arithmetic_verified'])
    def test_answers_shared(self):
        row={**self.kk,**build_guidance(self.kk)}
        for mode in ['answer_only','full_reasoning','key_steps','error_targeted']:
            text,_=reference_text(row,mode,'');self.assertTrue(text.startswith('Reference answer: '+row['answer']))
    def test_correct_error_fallback(self):
        hint,status=error_hint(self.kk,'<answer>'+self.kk['answer']+'</answer>')
        self.assertIsNone(hint);self.assertEqual(status,'correct_answer_fallback')
    def test_unparseable_error_fallback(self):self.assertIsNone(error_hint(self.kk,'unparseable')[0])
    def test_verified_math_error(self):self.assertEqual(error_hint(self.math,'3 * 4 = 13 <answer>13</answer>')[1],'verified_arithmetic_error')
    def test_no_guessing_semantic_error(self):self.assertIsNone(error_hint(self.math,'It should be thirteen <answer>13</answer>')[0])
    def test_countdown_wrong_multiset(self):self.assertEqual(error_hint(self.cd,'<answer>999+999</answer>')[1],'verified_number_multiset_error')
    def test_branch_horizons_and_sampler(self):
        for start,end in [(30,180),(90,240)]:
            _,cfg,got,actual=command(Path('/tmp/unit'), 'kk','plain',42,Path('/tmp/unit/run'),Path('/tmp/ref/global_step_'+str(start)))
            self.assertEqual((got,actual),(start,end));self.assertEqual(cfg['data.seed'],1000042)
            self.assertTrue(cfg['trainer.reset_dataloader_on_fork']);self.assertEqual(cfg['trainer.total_training_steps'],end)
    def test_reference_save_points(self):
        _,cfg,start,end=command(Path('/tmp/unit'),'kk','plain',42,Path('/tmp/unit/run'))
        self.assertEqual((start,end),(0,150));self.assertEqual(cfg['trainer.extra_save_steps'],[30,90])
    def test_paired_engine_seed(self):
        _,cfg,_,_=command(Path('/tmp/unit'),'gsm8k','plain',43,Path('/tmp/unit/run'))
        self.assertEqual(cfg['actor_rollout_ref.rollout.engine_seed'],43);self.assertEqual(cfg['data.max_response_length'],8192)
    def test_slim_rollout_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl';path.write_text(json.dumps(self.zebra)+'\n')
            ds=FrozenTaskDataset(path);self.assertNotIn('cnf',ds[0]['metadata'])
            self.assertEqual(ds.score_answer(self.zebra['answer'],ds[0]),1)
    def test_jsonl_unicode_line_separator(self):
        from state_guidance.data import read_rows
        row={**self.math,'question':'A\u2028B\u0085C'};row['question_sha256']=sha_text(row['question'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl';path.write_text(json.dumps(row,ensure_ascii=False)+'\n')
            self.assertEqual(read_rows(path)[0]['question'],row['question'])
    def test_conditioned_response_alignment(self):
        import torch,numpy as np
        from verl import DataProto
        from state_guidance.teacher_inputs import conditioned_reference_batch
        class Tokenizer:
            pad_token_id=0
            def apply_chat_template(self,chat,**kwargs):return json.dumps(chat)
            def encode(self,text,**kwargs):return list(text.encode())
        row={**self.kk,**build_guidance(self.kk)}
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'guidance.jsonl';f.write_text(json.dumps(row)+'\n')
            b=DataProto.from_single_dict({'input_ids':torch.tensor([[0,4,10,11]]),'attention_mask':torch.tensor([[0,1,1,1]]),
                 'responses':torch.tensor([[10,11]]),'prompts':torch.tensor([[0,4]]),
                 'position_ids':torch.tensor([[0,0,1,2]]),'index':np.array([0],dtype=object),
                 'question':np.array([row['question']],dtype=object),
                 'raw_prompt':np.array([[{'role':'user','content':row['question']}]],dtype=object)})
            out,m=conditioned_reference_batch(b,Tokenizer(),'answer_only',str(f),8192,None)
            self.assertTrue(torch.equal(out.batch['responses'],b.batch['responses']))
            self.assertTrue(torch.equal(out.batch['attention_mask'][:,-2:],b.batch['attention_mask'][:,-2:]))
            self.assertEqual(m['guidance/response_alignment_ok'],1)


if __name__=='__main__':unittest.main(verbosity=2)
