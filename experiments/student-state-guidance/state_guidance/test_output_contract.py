import json,hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from verl import DataProto
from state_guidance.data import sha_text
from state_guidance.teacher_inputs import conditioned_reference_batch,KK_OUTPUT_CONTRACT


class OutputContractTests(unittest.TestCase):
    def test_formal_policy_applies_to_all_eight_commands(self):
        from state_guidance.run_phase import command
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'TEACHER_OUTPUT_POLICY.json').write_text(json.dumps({
                'status':'validated_for_formal','kind':'kk_teacher_completion_contract_v1',
                'reminder_sha256':hashlib.sha256(KK_OUTPUT_CONTRACT.encode()).hexdigest()}))
            for state in [30,90]:
                for mode in ['plain','answer_only','full_reasoning','error_targeted']:
                    _,cfg,start,end=command(root,'kk',mode,42,root/'runs/branch',root/f'global_step_{state}')
                    self.assertTrue(cfg['algorithm.kk_teacher_output_contract'])
                    self.assertEqual((start,end),(state,state+150))
                    self.assertEqual(cfg['data.cot_suffix'],"Let's think step by step.\n")
            with self.assertRaises(AssertionError):command(root,'gsm8k','plain',42,root/'runs/other')

    def test_teacher_only_and_plain_has_no_answer(self):
        class Tokenizer:
            pad_token_id=0
            def apply_chat_template(self,chat,**kwargs):
                self.chat=deepcopy(chat)
                return json.dumps(chat)
            def encode(self,text,**kwargs):return list(text.encode())
        question='Assign the six roles.'
        row={'index':0,'question_sha256':sha_text(question),'answer':'SECRET ROLE ASSIGNMENT'}
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'guidance.jsonl';f.write_text(json.dumps(row)+'\n')
            for mode in ['plain','answer_only']:
                b=DataProto.from_single_dict({
                    'input_ids':torch.tensor([[0,4,10,151643,0]]),
                    'attention_mask':torch.tensor([[0,1,1,1,0]]),
                    'responses':torch.tensor([[10,151643,0]]),
                    'prompts':torch.tensor([[0,4]]),'position_ids':torch.tensor([[0,0,1,2,0]]),
                    'index':np.array([0],dtype=object),'question':np.array([question],dtype=object),
                    'raw_prompt':np.array([[{'role':'user','content':question}]],dtype=object)})
                before={k:v.clone() for k,v in b.batch.items()};raw=deepcopy(b.non_tensor_batch['raw_prompt'])
                tok=Tokenizer();out,metrics=conditioned_reference_batch(b,tok,mode,str(f) if mode!='plain' else '/nonexistent',2048,'\n',kk_output_contract=True)
                self.assertIn(KK_OUTPUT_CONTRACT,tok.chat[-1]['content'])
                self.assertEqual('SECRET ROLE ASSIGNMENT' in tok.chat[-1]['content'],mode=='answer_only')
                for k,v in before.items():self.assertTrue(torch.equal(b.batch[k],v),k)
                self.assertEqual(b.non_tensor_batch['raw_prompt'].tolist(),raw.tolist())
                self.assertTrue(torch.equal(out.batch['input_ids'][:,-3:],before['responses']))
                self.assertTrue(torch.equal(out.batch['attention_mask'][:,-3:],before['attention_mask'][:,-3:]))
                self.assertEqual(metrics['guidance/output_contract_sequences'],1)
                self.assertEqual(metrics['guidance/conditioned_sequences'],int(mode!='plain'))


if __name__=='__main__':unittest.main()
