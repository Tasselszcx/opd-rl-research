import json,tempfile,unittest
from pathlib import Path
from state_guidance.coordinator import phases,state_candidates
from state_guidance.control import atomic,ledger
from state_guidance.run_phase import command

class RepairedScopeTests(unittest.TestCase):
    def make(self,r):
        scope={'schema':1,'authorization':'explicit_user_scope_revision','tasks':['kk'],'training_seeds':[42],
            'modes':['plain','answer_only','full_reasoning','error_targeted'],'states':[30,90],'training_routes':9,
            'ordered_modes':['plain','answer_only','full_reasoning','error_targeted'],'sequential_student_states':True}
        atomic(r/'ACTIVE_SCOPE.json',scope);return phases(r)
    def test_nine_routes_and_explicit_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make(Path(d));self.assertEqual(len(p),9)
            self.assertEqual([(x['start'],x['mode']) for x in p[1:]],[(s,m) for s in [30,90] for m in ['plain','answer_only','full_reasoning','error_targeted']])
    def test_running_state30_blocks_state90_even_if_lane_free(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.make(Path(d));done={x['name'] for x in p[:4]}
            self.assertEqual(state_candidates(p[5:],p,done,42,True),[])
            done.add(p[4]['name']);self.assertEqual(state_candidates(p[5:],p,done,42,True),p[5:])
    def test_historical_cost_counted_once(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);atomic(r/'HISTORICAL_COST.json',{'allocated_gpu_hours':100.})
            atomic(r/'runs/reference/launcher_record.json',{'n_gpus':0,'status':'completed','allocated_gpu_hours':0})
            atomic(r/'runs/new/launcher_record.json',{'n_gpus':4,'status':'completed','allocated_gpu_hours':2.})
            self.assertEqual(ledger(r)['allocated_gpu_hours'],102.)
    def test_all_formal_guidance_modes_use_same_eos_fix(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);atomic(r/'PROTOCOL_AMENDMENT.json',{'status':'approved_user_repair','method_revision':'semantic_eos_v1'})
            for mode in ['plain','answer_only','full_reasoning','error_targeted']:
                _,cfg,_,_=command(r,'kk',mode,42,r/'runs'/mode,r/'global_step_30')
                self.assertEqual(cfg['algorithm.opd_eos_mode'],'semantic')
                self.assertEqual(cfg['actor_rollout_ref.ref.eos_semantic_teacher_ids'],[151643,151645])

if __name__=='__main__':unittest.main()
