"""Regression checks for false completion, historical reuse, and score attribution."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from build_g3a_snapshot import build, write


def fixture():
    point = dict(additional_step=25, global_step=25, questions=512, samples=8,
        pass_at_1=.3, pass_at_8=.75, mean_response_tokens=1200,
        output_limit_fraction=.05, no_final_before_limit_fraction=0,
        question_scores=['PRIVATE_QUESTION_EVIDENCE'])
    return dict(campaign='student-state-guidance-03-kk-clean-v4', scope_sha256='scope',
        checked_beijing='2026-11-06T04:30:00+08:00', pending_or_invalid=[],
        active_scope=dict(tasks=['kk'], training_seeds=[42], states=[30, 90],
            modes=['plain','answer_only','full_reasoning','error_targeted'],
            formal_branches=8, branch_additional_steps=150, legacy_reference_allowed=False),
        runs=[dict(task='kk', training_seed=42, start_step=0, guidance='plain', status='running',
            public_config=dict(clean_reference=True, eos_mode='semantic', starting_checkpoint_origin='official_base'),
            train=[dict(additional_step=i, global_step=i) for i in range(1,31)], dev=[point])])


class SnapshotChecks(unittest.TestCase):
    def test_reference_score_is_separate_and_private_details_excluded(self):
        summary = build(fixture(), 'a'*40, 'scope')
        self.assertEqual(summary['counts']['pending'], 8)
        self.assertEqual(summary['reference']['latest_development']['pass_at_1_pct'], 30)
        self.assertTrue(all(r['dev_pass_at_1_pct'] is None for r in summary['routes']))
        self.assertNotIn('PRIVATE_QUESTION_EVIDENCE', json.dumps(summary))

    def test_legacy_scope_and_incomplete_completion_are_rejected(self):
        for mutate in [lambda s:s.update(campaign='legacy'),
                       lambda s:s.update(scope_sha256='other'),
                       lambda s:s['runs'][0].update(reused_reference=True),
                       lambda s:s['runs'][0].update(status='completed')]:
            snapshot=fixture();mutate(snapshot)
            with self.assertRaises(AssertionError):build(snapshot,'a'*40,'scope')

    def test_branch_step0_is_not_a_trained_result(self):
        snapshot=fixture();reference=snapshot['runs'][0]
        reference['status']='completed'
        reference['train']=[dict(additional_step=i,global_step=i) for i in range(1,151)]
        branch=deepcopy(reference)
        branch.update(start_step=30,status='running',train=[])
        branch['public_config']['starting_checkpoint_origin']='clean_reference_v4'
        branch['dev'][0].update(additional_step=0,global_step=30)
        snapshot['runs'].append(branch)
        summary=build(snapshot,'a'*40,'scope')
        self.assertEqual(summary['counts']['completed'],0)
        self.assertEqual(summary['counts']['running'],1)
        self.assertIsNone(summary['routes'][0]['latest_development_update'])
        self.assertTrue(all(p['route_kind']=='reference' for p in summary['development']))
        with TemporaryDirectory() as folder:
            document = write(summary, Path(folder))
        self.assertIn('初始化／起始验证', document)


if __name__=='__main__':unittest.main()
