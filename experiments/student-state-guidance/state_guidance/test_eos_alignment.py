import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from verl.utils.eos_alignment import semantic_eos_log_probs

tree=ast.parse((Path(__file__).parents[1]/'verl/verl/trainer/ppo/ray_trainer.py').read_text())
ns={'torch':torch,'DataProto':object,'Enum':Enum}
for name in ['AdvantageEstimator','compute_advantage']:
    node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),ns)

class EosTests(unittest.TestCase):
    def test_mass_fold_and_non_eos_preserved(self):
        logits=torch.tensor([[0.,1.,2.,3.]]).repeat(4,1);labels=torch.arange(4)
        old=logits.log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
        new=semantic_eos_log_probs(logits,labels,old,2,[3])
        self.assertTrue(torch.equal(new[:2],old[:2]))
        self.assertAlmostEqual(new.exp().sum().item(),1.,places=6)
        self.assertAlmostEqual(new[2].exp().item(),old[2:].exp().sum().item(),places=6)
        self.assertTrue(torch.isfinite(new).all())
    def test_duplicate_alias_is_not_double_counted(self):
        logits=torch.randn(3,5);ids=torch.tensor([1,2,3]);lp=logits.log_softmax(-1).gather(-1,ids[:,None]).squeeze(-1)
        self.assertTrue(torch.equal(semantic_eos_log_probs(logits,ids,lp,2,[2,3,3]),semantic_eos_log_probs(logits,ids,lp,2,[3])))
    def test_low_precision_normalizer_fails_closed(self):
        with self.assertRaises(ValueError):
            semantic_eos_log_probs(torch.zeros(1,4),torch.tensor([2]),torch.zeros(1,dtype=torch.bfloat16),2,[3])
    def run_adv(self,mode,aligned=True):
        d=SimpleNamespace(batch={'responses':torch.tensor([[2,0],[1,2],[1,1]]),'attention_mask':torch.tensor([[1,0],[1,1],[1,1]]),
            'old_log_probs':torch.tensor([[-.1,0.],[-1.,-.1],[-1.,-1.]]),
            'ref_log_prob':torch.tensor([[-20.,0.],[-.5,-.01],[-.5,-.5]])},meta_info={'opd_eos_semantic':aligned})
        ns['compute_advantage'](d,ns['AdvantageEstimator'].OPD,opd_eos_mode=mode)
        return d.batch['advantages']
    def test_premature_eos_gets_negative_signal(self):self.assertEqual(self.run_adv('semantic')[0,0],-5.)
    def test_valid_stop_is_not_artificially_penalized(self):self.assertGreater(self.run_adv('semantic')[1,1],0.)
    def test_padding_stays_zero(self):self.assertEqual(self.run_adv('semantic')[0,1],0.)
    def test_truncated_non_eos_keeps_signal(self):self.assertEqual(self.run_adv('semantic')[2,1],.5)
    def test_legacy_control_retains_exact_behavior(self):self.assertEqual(self.run_adv('legacy_mask')[0,0],0.)
    def test_missing_alignment_fails_closed(self):
        with self.assertRaises(AssertionError):self.run_adv('semantic',False)

if __name__=='__main__':unittest.main()
