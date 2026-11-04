import ast,contextlib,json,os,tempfile,unittest,warnings
from pathlib import Path
from types import SimpleNamespace
from state_guidance.run_phase import command,validate
from state_guidance.control import atomic

ROOT=Path(__file__).parents[1]
POLICY={'status':'approved_by_user','policy':'fork_states_full_terminal_model_only'}

class StorageTests(unittest.TestCase):
    def test_reference_preserves_full_fork_points(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);atomic(r/'STORAGE_POLICY.json',POLICY)
            _,c,_,_=command(r,'kk','plain',42,r/'runs/ref')
            self.assertEqual(c['trainer.extra_save_steps'],[30,90]);self.assertEqual(c['trainer.save_freq'],150)
            self.assertTrue(c['trainer.g3_terminal_model_only'])
    def test_branches_have_only_terminal_save_and_keep_dev_cadence(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);atomic(r/'STORAGE_POLICY.json',POLICY)
            for step in [30,90]:
                _,c,start,end=command(r,'kk','full_reasoning',42,r/'runs/branch',r/f'global_step_{step}')
                self.assertEqual((start,end),(step,step+150));self.assertEqual(c['trainer.save_freq'],150)
                self.assertEqual(c['trainer.extra_save_steps'],[]);self.assertEqual(c['trainer.test_freq'],25)
    def test_entry_checks_remain_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);atomic(r/'STORAGE_POLICY.json',POLICY)
            _,c,_,_=command(r,'kk','plain',42,r/'checks/probe',steps=1,check=True)
            self.assertEqual(c['trainer.save_freq'],1);self.assertFalse(c.get('trainer.g3_terminal_model_only',False))

    def exercise_writer(self,model_only):
        tree=ast.parse((ROOT/'verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FSDPCheckpointManager')
        save=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint')
        load=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='load_checkpoint')
        calls=[];written=[]
        def state(name):calls.append(name);return {'type':name}
        def write(value,path):written.append(Path(path).name);Path(path).write_text(json.dumps(value))
        ns={'os':os,'json':json,'warnings':warnings,'torch':SimpleNamespace(save=write,distributed=SimpleNamespace(barrier=lambda:None)),
            'FSDP':SimpleNamespace(state_dict_type=lambda *a,**kw:contextlib.nullcontext()),
            'StateDictType':SimpleNamespace(SHARDED_STATE_DICT=1),
            'ShardedStateDictConfig':lambda **kw:None,'ShardedOptimStateDictConfig':lambda **kw:None}
        for node in [save,load]:exec(compile(ast.Module(body=[node],type_ignores=[]),node.name,'exec'),ns)
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            def hf(path):Path(path,'config.json').write_text('{}')
            fake=SimpleNamespace(world_size=4,rank=0,local_mkdir=lambda p:p,
                model=SimpleNamespace(state_dict=lambda:state('model'),_fsdp_wrapped_module=SimpleNamespace(config=SimpleNamespace(save_pretrained=hf))),
                optimizer=SimpleNamespace(state_dict=lambda:state('optimizer')),
                lr_scheduler=SimpleNamespace(state_dict=lambda:state('scheduler')),
                processing_class=SimpleNamespace(save_pretrained=lambda path:Path(path,'tokenizer_config.json').write_text('{}')),
                get_rng_state=lambda:state('rng'))
            ns['save_checkpoint'](fake,str(r),180,model_only=model_only)
            if model_only:
                self.assertEqual(calls,['model']);self.assertEqual(written,['model_world_size_4_rank_0.pt'])
                with self.assertRaisesRegex(ValueError,'model weights only'):ns['load_checkpoint'](fake,str(r))
            else:
                self.assertEqual(set(calls),{'model','optimizer','scheduler','rng'})
                self.assertEqual(set(written),{'model_world_size_4_rank_0.pt','optim_world_size_4_rank_0.pt','extra_state_world_size_4_rank_0.pt'})
    def test_model_only_writer_omits_optimizer_and_rng_and_rejects_resume(self):self.exercise_writer(True)
    def test_full_writer_keeps_optimizer_and_rng(self):self.exercise_writer(False)

    def fixture(self,r,start):
        atomic(r/'STORAGE_POLICY.json',POLICY);folder=r/'runs/fixture';cp=folder/'checkpoints';cp.mkdir(parents=True)
        rows=[{'step':i,'prompt_indices':list(range(128)), 'metrics':{
            'reproduction/teacher_used_in_objective':1,'guidance/response_alignment_ok':1,
            'actor/grad_norm':1.,'guidance/conditioned_sequences':0}} for i in range(start+1,start+151)]
        (cp/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        def endpoint(step):
            p=cp/f'global_step_{step}';(p/'actor/huggingface').mkdir(parents=True)
            (p/'data.pt').write_bytes(b'metadata')
            (p/'actor/huggingface/config.json').write_text('{}')
            for rank in range(4):(p/'actor'/f'model_world_size_4_rank_{rank}.pt').write_bytes(b'weights')
            atomic(p/'checkpoint_kind.json',{'kind':'model_only'});atomic(p/'actor/checkpoint_contents.json',{'model_only':True})
        endpoint(start+150);return folder,endpoint
    def test_terminal_validation_accepts_model_only(self):
        with tempfile.TemporaryDirectory() as d:
            folder,_=self.fixture(Path(d),30)
            self.assertEqual(validate(folder,30,180,'plain')[1],150)
    def test_reference_start_points_cannot_be_model_only(self):
        with tempfile.TemporaryDirectory() as d:
            folder,endpoint=self.fixture(Path(d),0);endpoint(30);endpoint(90)
            with self.assertRaisesRegex(AssertionError,'Branch start states'):validate(folder,0,150,'plain')

if __name__=='__main__':unittest.main()
