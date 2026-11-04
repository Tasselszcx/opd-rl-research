import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from state_guidance.control import atomic, identity
from state_guidance.scope import load_scope, seed_summary
from state_guidance.coordinator import phases, next_candidate, SLOTS
from state_guidance.adoption import AdoptedProcess, adoption_items
from state_guidance.finalize import jobs, training_finished
from state_guidance.analyze import batch_question_order


def core_scope(root):
    data={'schema':1,'authorization':'explicit_user_scope_revision','tasks':['kk','gsm8k'],
          'training_seeds':[42],'modes':['plain','answer_only','full_reasoning','error_targeted'],
          'states':[30,90],'training_routes':18,'baseline_plan_sha256':'0'*64}
    atomic(root/'ACTIVE_SCOPE.json',data)
    return data


class ScopeTests(unittest.TestCase):
    def test_core_has_eighteen_routes_and_no_cancelled_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);core_scope(root);p=phases(root)
            self.assertEqual(len(p),18)
            self.assertEqual(sum(x['resume'] is None for x in p),2)
            self.assertEqual({x['seed'] for x in p},{42})
            self.assertEqual({x['task'] for x in p},{'kk','gsm8k'})
            self.assertNotIn('key_steps',{x['mode'] for x in p})
            self.assertTrue(all(x['steps']==150 for x in p))

    def test_subset_keeps_original_within_task_counterbalance_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=phases(root);core_scope(root)
            expected=[p['name'] for p in original if p['task'] in ['kk','gsm8k'] and p['seed']==42
                      and p['mode']!='key_steps']
            self.assertEqual([p['name'] for p in phases(root)],expected)

    def test_eval_is_twenty_fixed_models_and_eight_math_transfer_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);core_scope(root);j=jobs(root)
            self.assertEqual(len(j),20)
            self.assertEqual(sum(x['samples']==16 for x in j),4)
            self.assertEqual(sum('math500' in x['suites'] for x in j),8)
            self.assertEqual({x['seed'] for x in j},{42})

    def test_one_seed_summary_does_not_invent_seed_variance(self):
        s=seed_summary([.1],[42])
        self.assertIsNone(s['seed_std'])
        self.assertEqual(s['interaction_by_training_seed'],{'42':.1})

    def test_scope_counts_and_unknown_conditions_rejected(self):
        for change in [{'training_routes':19},{'modes':['plain','answer_only','unknown']},
                       {'authorization':'implicit'},{'training_seeds':[42,42]}]:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data=core_scope(root);data.update(change);atomic(root/'ACTIVE_SCOPE.json',data)
                with self.assertRaises(AssertionError):load_scope(root)

    def test_finalizer_rejects_different_training_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);core_scope(root)
            with self.assertRaisesRegex(AssertionError,'scope mismatch'):
                training_finished(root,{'status':'training_complete','scope_sha256':'different'})

    def test_free_lane_prefers_the_other_benchmark(self):
        c=[{'task':'kk','name':'next-kk'},{'task':'gsm8k','name':'next-math'}]
        selected=next_candidate(c,{'B':{'phase':{'task':'kk'}}})
        self.assertEqual(selected['name'],'next-math')

    def test_batch_matching_ignores_balancing_but_not_cross_step_changes(self):
        first=[{'prompt_indices':[2,1]},{'prompt_indices':[4,3]}]
        balanced=[{'prompt_indices':[1,2]},{'prompt_indices':[3,4]}]
        shifted=[{'prompt_indices':[3,4]},{'prompt_indices':[1,2]}]
        self.assertEqual(batch_question_order(first),batch_question_order(balanced))
        self.assertNotEqual(batch_question_order(first),batch_question_order(shifted))


class AdoptionTests(unittest.TestCase):
    def fixture(self,root):
        core_scope(root);phase=next(p for p in phases(root) if p['name']=='kk-s42-state30-full_reasoning')
        folder=root/'runs'/phase['name'];folder.mkdir(parents=True);(root/'logs').mkdir()
        request=folder/'request.json';request.write_text('{"unchanged":true}')
        launcher={'pid':8,'start_ticks':80};child={'pid':9,'start_ticks':90}
        atomic(folder/'launcher_record.json',{'status':'running','task':'kk','mode':'full_reasoning','seed':42,
               'start_step':30,'end_step':180,'gpu_ids':[4,5,6,7],'started_unix':100.,
               'launcher_identity':launcher,'child_identity':child})
        item={'slot':'B','name':phase['name'],'attempt_started_unix':100.,'launcher_identity':launcher,
              'child_identity':child,'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest()}
        manifest={'schema':1,'reason':'explicit_user_scope_revision','previous_coordinator':{'pid':7},'retained':[item]}
        return phase,folder,manifest

    def test_adoption_keeps_identity_and_original_operational_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);phase,folder,manifest=self.fixture(root)
            active=adoption_items(root,manifest,phases(root),SLOTS,lambda s:s['pid'] in [8,9])
            self.assertEqual(active['B']['identity']['pid'],8)
            self.assertEqual(active['B']['started'],100.)
            self.assertIsNone(active['B']['process'].poll())
            active['B']['log'].close()

    def test_running_old_coordinator_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,manifest=self.fixture(root)
            with self.assertRaises(AssertionError):
                adoption_items(root,manifest,phases(root),SLOTS,lambda _:True)

    def test_wrong_identity_lane_or_request_cannot_be_adopted(self):
        for field,value in [('request_sha256','wrong'),('slot','A'),('launcher_identity',{'pid':88,'start_ticks':80})]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,_,manifest=self.fixture(root);manifest['retained'][0][field]=value
                with self.assertRaises(AssertionError):
                    adoption_items(root,manifest,phases(root),SLOTS,lambda s:s['pid'] in [8,9,88])

    def test_dead_unsettled_launcher_requires_investigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            record=Path(tmp)/'launcher_record.json';atomic(record,{'status':'running'})
            p=AdoptedProcess({'pid':8},record,lambda _:False)
            with self.assertRaisesRegex(RuntimeError,'without a settled record'):p.poll()

    @unittest.skipUnless(Path('/proc/self').exists(),'Linux process identity test')
    def test_real_cpu_launcher_finishes_after_adoption_without_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            record=Path(tmp)/'launcher_record.json';atomic(record,{'status':'running'})
            code='import time,json,sys;time.sleep(.3);open(sys.argv[1],"w").write(json.dumps({"status":"completed"}))'
            child=subprocess.Popen([sys.executable,'-c',code,str(record)])
            try:
                p=AdoptedProcess(identity(child.pid),record)
                self.assertEqual(p.wait(timeout=3),0)
            finally:child.wait(timeout=3)


if __name__=='__main__':unittest.main()
