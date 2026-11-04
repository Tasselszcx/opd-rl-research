"""Analyze frozen endpoints only after every guidance arm and test is complete."""
from pathlib import Path
import csv
import json
import math
import os
import numpy as np
from guidance.costs import charged_gpu_hours

ROOT=Path(os.environ['REPRO_ROOT'])
MIGRATION=Path(os.environ['MIGRATION_ROOT'])
MODES=['plain','key_step','full_solution']

def main():
    reference=None
    for mode in MODES:
        order={}
        for phase,steps in [('opd',list(range(1,21))),('rl',list(range(21,41)))]:
            folder=ROOT/'runs'/(mode+'-'+phase)
            record=json.loads((folder/'launcher_record.json').read_text())
            assert record['status']=='completed' and record['validation_status']=='passed'
            rows=[json.loads(x) for x in (folder/'checkpoints/metrics.jsonl').read_text().splitlines()]
            assert [x['step'] for x in rows]==steps
            assert all(x['metrics']['reproduction/teacher_used_in_objective']==(phase=='opd') for x in rows)
            order.update({x['step']:x['prompt_indices'] for x in rows})
        if reference is None:reference=order
        else:assert reference==order,'Training order differs across guidance arms'
    expected={x['index']:x['question_sha256'] for x in map(json.loads,(ROOT/'data/pilot_test.jsonl').read_text().splitlines())}
    arrays={};metrics={};question_rows=[]
    for phase in ['opd','rl']:
        arrays[phase]={};metrics[phase]={}
        for mode in MODES:
            folder=ROOT/'evaluations'/(mode+'-'+phase)
            assert json.loads((folder/'launcher_record.json').read_text())['status']=='completed'
            scores={}
            for f in folder.glob('rank_*.jsonl'):
                for line in f.open():
                    x=json.loads(line);key=(x['index'],x['sample']);assert key not in scores
                    assert x['question_sha256']==expected[x['index']] and x['score'] in (0,1)
                    assert x['suite']=='pilot_test' and x['sampling_seed']==42+x['sample']
                    scores[key]=x['score']
            assert set(scores)=={(i,j) for i in range(512) for j in range(4)}
            matrix=np.array([[scores[i,j] for j in range(4)] for i in range(512)])
            counts=matrix.sum(axis=1).astype(int);arrays[phase][mode]={'pass_at_1':matrix.mean(axis=1),'pass_at_4':(counts>0).astype(float)}
            curve={str(k):float(np.mean([1 if k>4-c else 1-math.comb(4-int(c),k)/math.comb(4,k) for c in counts])) for k in range(1,5)}
            metrics[phase][mode]={'pass_at_1':float(matrix.mean()),'pass_at_4':float((counts>0).mean()),'pass_at_k':curve,'questions':512,'responses':2048}
            question_rows.extend({'phase':phase,'mode':mode,'index':i,'question_sha256':expected[i],'correct_count':int(c),'samples':4} for i,c in enumerate(counts))
    rng=np.random.default_rng(20261023);comparisons={}
    for phase in arrays:
        comparisons[phase]={}
        for mode in ['key_step','full_solution']:
            comparisons[phase][mode]={}
            for metric in ['pass_at_1','pass_at_4']:
                delta=arrays[phase][mode][metric]-arrays[phase]['plain'][metric]
                draws=rng.choice(delta,(10000,512),replace=True).mean(axis=1)
                comparisons[phase][mode][metric]={'difference':float(delta.mean()),'paired_question_ci95':np.quantile(draws,[.025,.975]).tolist()}
    total=charged_gpu_hours(ROOT,MIGRATION)
    result={'status':'completed','metrics':metrics,'comparisons_vs_plain':comparisons,
            'total_allocated_gpu_hours':total,'gpu_hour_cap':24,'single_training_seed':True,
            'guidance_on_teacher_only':True,'training_order_matches':True,
            'primary_outcome':'Post-RL independent-test performance relative to same-start plain guidance; not RL-only increment',
            'scope':'Short controlled pilot with oracle-derived training guidance, not a learned teaching policy or full paper reproduction'}
    (ROOT/'study_results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'per_question_scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(question_rows[0]));writer.writeheader();writer.writerows(question_rows)
    report='# K&K 指导方式小试验 G1\n\n固定 teacher；指导仅提供给 teacher，student 始终只看原题。每种方式20步OPD后接20步RL，单训练种子；新独立集512题×4次采样。\n'
    for phase,title in [('opd','OPD 后'),('rl','后续 RL 后')]:
        report+='\n## '+title+'\n\n| 指导 | pass@1 | pass@4 |\n|---|---:|---:|\n'
        for mode,row in metrics[phase].items():report+=f"| {mode} | {row['pass_at_1']:.2%} | {row['pass_at_4']:.2%} |\n"
        for mode,values in comparisons[phase].items():
            for metric,v in values.items():
                lo,hi=v['paired_question_ci95'];report+=f"\n{mode} 相对 plain 的 {metric}：{100*v['difference']:+.2f} 个百分点，95%题目级配对区间 [{100*lo:+.2f}, {100*hi:+.2f}]。\n"
    report+=f'\n总计 {total:.3f} GPU小时，包含当前迁移训练链路检查及本轮失败尝试、训练、评测。三组按步数对齐，不代表总计算量一致。\n'
    report+='\n区间不覆盖训练种子不确定性。指导来自训练题的精确求解器；本轮未训练指导生成器、未在线选择教法，也未用最终测试挑checkpoint或调参。仅比较固定终点，不将RL增量单独当教法价值。\n'
    (ROOT/'PILOT_RESULTS.md').write_text(report)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
