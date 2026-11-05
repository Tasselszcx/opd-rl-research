"""Observe actual rollout sampling arguments without constructing a GPU engine."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace,MethodType
import unittest
import numpy as np
import torch
from verl import DataProto


class Captured(Exception):pass


class DevelopmentSeedTests(unittest.TestCase):
    def capture(self,seed=4200,validate=True,previous=None):
        source=(Path(__file__).parents[1]/'verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py').read_text()
        tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='generate_sequences' for f in n.body))
        ns={'torch':torch,'np':np,'DataProto':DataProto,'contextmanager':contextmanager,'vllm_version':'0.10.1'}
        for node in cls.body:
            if isinstance(node,ast.FunctionDef) and node.name in ['generate_sequences','update_sampling_params']:
                exec(compile(ast.Module(body=[node],type_ignores=[]),node.name,'exec'),ns)
        captured={}
        def generate(**kwargs):
            params=kwargs['sampling_params'];captured.update({k:getattr(params,k) for k in ['seed','n','temperature','top_p','top_k']});raise Captured()
        val=SimpleNamespace(n=8,temperature=.7,top_p=.8,top_k=20)
        if seed is not None:val.seed=seed
        val.get=lambda key,default=None:getattr(val,key,default)
        obj=SimpleNamespace(config=SimpleNamespace(val_kwargs=val),sampling_params=SimpleNamespace(seed=previous,n=8,temperature=1.,top_p=1.,top_k=-1),inference_engine=SimpleNamespace(generate=generate))
        obj.update_sampling_params=MethodType(ns['update_sampling_params'],obj)
        batch=DataProto.from_dict(tensors={'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.long),'position_ids':torch.tensor([[0,1,2]])},non_tensors={'raw_prompt_ids':np.array([[1,2,3]],dtype=object)},meta_info={'validate':validate,'do_sample':True,'eos_token_id':151643})
        with self.assertRaises(Captured):ns['generate_sequences'](obj,batch)
        return captured

    def test_explicit_development_seed_reaches_engine(self):
        self.assertEqual(self.capture(),{'seed':4200,'n':8,'temperature':.7,'top_p':.8,'top_k':20})
    def test_zero_seed_is_not_dropped(self):self.assertEqual(self.capture(seed=0)['seed'],0)
    def test_unspecified_seed_preserves_existing_sampling(self):self.assertEqual(self.capture(seed=None,previous=123)['seed'],123)
    def test_training_sampling_is_unchanged(self):
        self.assertEqual(self.capture(validate=False,previous=123),{'seed':123,'n':8,'temperature':1.,'top_p':1.,'top_k':-1})


if __name__=='__main__':unittest.main()
