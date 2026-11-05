import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from state_guidance.clean_reference import check_configuration,check_reference_origin,check_fork_parent
from state_guidance.run_phase import command
from state_guidance.control import atomic


class CleanReferenceTests(unittest.TestCase):
    def setup_root(self, path):
        rule={'status':'clean_reference_required','approved_unix':100.,'training_release':'new-clean-release'}
        atomic(path/'CLEAN_REFERENCE_REQUIREMENT.json',rule)
        atomic(path/'ACTIVE_SCOPE.json',{'clean_reference_requirement_sha256':hashlib.sha256((path/'CLEAN_REFERENCE_REQUIREMENT.json').read_bytes()).hexdigest()})
        atomic(path/'PROTOCOL_AMENDMENT.json',{'status':'approved_user_repair','method_revision':'semantic_eos_v1'})

    def test_reference_and_all_eight_commands_use_repaired_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.setup_root(root)
            _,cfg,start,end=command(root,'kk','plain',42,root/'runs/kk-s42-reference')
            self.assertEqual((start,end),(0,150));self.assertEqual(cfg['trainer.resume_mode'],'disable')
            for state in [30,90]:
                for mode in ['plain','answer_only','full_reasoning','error_targeted']:
                    _,cfg,start,end=command(root,'kk',mode,42,root/'runs/branch',root/f'runs/kk-s42-reference/checkpoints/global_step_{state}')
                    self.assertEqual((start,end),(state,state+150));self.assertEqual(cfg['data.seed'],1000042)
                    self.assertEqual(cfg['algorithm.opd_eos_mode'],'semantic');self.assertTrue(cfg['trainer.reset_dataloader_on_fork'])

    def test_external_legacy_parent_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.setup_root(root)
            with self.assertRaisesRegex(AssertionError,'Legacy or external'):
                command(root,'kk','plain',42,root/'runs/branch',root/'old-campaign/global_step_30')

    def test_legacy_eos_and_literal_newline_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.setup_root(root)
            _,cfg,_,_=command(root,'kk','plain',42,root/'runs/kk-s42-reference')
            for field,value in [('algorithm.opd_eos_mode','legacy_mask'),('data.cot_suffix',"Let's think step by step.\\n")]:
                bad=dict(cfg);bad[field]=value
                with self.assertRaises(AssertionError):check_configuration(root,bad,None)

    def test_reference_origin_and_full_optimizer_parent_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.setup_root(root);folder=root/'runs/kk-s42-reference'
            _,cfg,_,_=command(root,'kk','plain',42,folder)
            atomic(folder/'request.json',{'config':cfg,'source_resume':None})
            record={'status':'completed','start_step':0,'end_step':150,'started_unix':101.,'source_path':'/tmp/new-clean-release'}
            atomic(folder/'launcher_record.json',record);check_reference_origin(root)
            parent=folder/'checkpoints/global_step_30';(parent/'actor').mkdir(parents=True)
            for file in [parent/'data.pt']+[parent/'actor'/f'{kind}_world_size_4_rank_{rank}.pt' for kind in ['model','optim','extra_state'] for rank in range(4)]:file.write_bytes(b'fixture')
            check_fork_parent(root,parent)
            (parent/'actor/optim_world_size_4_rank_0.pt').unlink()
            with self.assertRaisesRegex(AssertionError,'optimizer'):check_fork_parent(root,parent)
            for overrides in [{'reuse_only':True},{'started_unix':99.},{'source_path':'/tmp/old-release'}]:
                atomic(folder/'launcher_record.json',dict(record,**overrides))
                with self.assertRaises(AssertionError):check_reference_origin(root)


if __name__=='__main__':unittest.main()
