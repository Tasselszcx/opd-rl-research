"""G3 fixed-endpoint, multi-seed analysis. No checkpoint or method selection."""
import csv
import json
import math
import os
from pathlib import Path
import numpy as np
from state_guidance.control import atomic,ledger,read
from state_guidance.coordinator import phases,TASKS,MODES
from state_guidance.finalize import jobs,validate
from state_guidance.eval_worker import entries_for,suite_settings
from collections import Counter
from state_guidance.scope import load_scope,seed_summary


def pass_curve(counts,n):
    return {str(k):float(np.mean([1.0 if n-int(c)<k else 1-math.comb(n-int(c),k)/math.comb(n,k) for c in counts])) for k in range(1,n+1)}


def bootstrap(delta,seed=20261029,draws=10000):
    rng=np.random.default_rng(seed);values=[]
    for begin in range(0,draws,250):
        values.extend(rng.choice(delta,size=(min(250,draws-begin),len(delta)),replace=True).mean(axis=1))
    return {'difference':float(delta.mean()),'paired_question_ci95':np.quantile(values,[.025,.975]).tolist(),
            'draws':draws,'interval_scope':'questions conditional on this training seed; exploratory, no multiplicity correction'}


def batch_question_order(rows):
    # Distributed length balancing can reorder examples within a batch.
    return [sorted(row['prompt_indices']) for row in rows]


def main():
    root=Path(os.environ['REPRO_ROOT']);phase_list=phases(root);orders={};learning={}
    scope=load_scope(root);tasks=scope['tasks'];modes=scope['modes'];seeds=scope['training_seeds']
    for phase in phase_list:
        folder=root/'runs'/phase['name'];rec=read(folder/'launcher_record.json');assert rec['status']=='completed'
        with (folder/'checkpoints/metrics.jsonl').open() as f:rows=[json.loads(x) for x in f]
        assert [x['step'] for x in rows]==list(range(phase['start']+1,phase['start']+151))
        if phase['resume']:
            key=(phase['task'],phase['seed']);order=batch_question_order(rows)
            if key in orders:assert orders[key]==order,'Added training batch question sets differ'
            else:orders[key]=order
        points=[]
        for log in (folder/'val_logs').rglob('metrics.json'):
            v=read(log);step=v['step']-phase['start']
            if 0<=step<=150:points.append((step,v['pass_at_1']/100))
        # metrics.json is rounded for display: compute curve points from actual raw correctness.
        exact={}
        for log in (folder/'val_logs').rglob('val.jsonl'):
            parent=read(log.parent/'metrics.json');step=parent['step']-phase['start']
            if not 0<=step<=150:continue
            with log.open() as f:scores=[json.loads(x)['correctness_score'] for x in f]
            assert len(scores)==512*8;exact[step]=float(np.mean(scores))
        assert set(exact)==set(range(0,151,25)),(phase['name'],sorted(exact))
        x=np.array(sorted(exact));y=np.array([exact[i] for i in x])
        learning[phase['name']]={'points':[{'additional_step':int(i),'pass_at_1':float(v)} for i,v in zip(x,y)],
                                 'mean_auc':float(np.trapz(y,x)/150)}
    all_jobs=jobs(root);arrays={};metrics={};qrows=[];curverows=[]
    for job in all_jobs:
        folder=root/'evaluations'/job['name'];assert read(folder/'launcher_record.json')['status']=='completed'
        validate(root,job,folder)
        suite_scores={s:{} for s in job['suites']};lengths={s:[] for s in job['suites']};cut={s:0 for s in job['suites']};errors=Counter()
        for file in folder.glob('rank_*.jsonl'):
            with file.open() as f:
                for line in f:
                    row=json.loads(line);suite=row['suite'];key=(row['index'],row['sample'])
                    assert key not in suite_scores[suite];suite_scores[suite][key]=row['score']
                    lengths[suite].append(row['response_tokens']);cut[suite]+=int(row['finish_reason']=='length')
                    if suite=='diagnostic':errors[row['error_category']]+=1
        metrics[job['name']]={}
        for suite,lookup in suite_scores.items():
            entries=entries_for(root,job['task'],suite);n,_=suite_settings(job['samples'],suite)
            mat=np.array([[lookup[i,s] for s in range(n)] for i in range(len(entries))]);counts=mat.sum(axis=1).astype(int)
            curve=pass_curve(counts,n);arrays[(job['task'],job['seed'],job['start'],job['mode'],suite)]={
                'p1':mat.mean(axis=1),'pk':(counts>0).astype(float),'counts':counts}
            metrics[job['name']][suite]={'questions':len(entries),'samples':n,'responses':int(mat.size),'pass_at_k':curve,
                'pass_at_1':curve['1'],'pass_at_max':curve[str(n)],'all_failure_fraction':float((counts==0).mean()),
                'mixed_fraction':float(((counts>0)&(counts<n)).mean()),'all_success_fraction':float((counts==n).mean()),
                'mean_response_tokens':float(np.mean(lengths[suite])),'length_stop_fraction':cut[suite]/mat.size,
                'error_categories':dict(errors) if suite=='diagnostic' else None}
            for i,c in enumerate(counts):qrows.append({'endpoint':job['name'],'task':job['task'],'suite':suite,
                'training_seed':job['seed'],'start_step':job['start'],'guidance':job['mode'],'index':i,
                'question_sha256':entries[i]['question_sha256'],'correct_count':int(c),'samples':n})
            curverows.extend({'endpoint':job['name'],'suite':suite,'k':int(k),'pass_at_k':v} for k,v in curve.items())
    comparisons={};interactions={};summaries={}
    for task in tasks:
        for suite in (['test','math500'] if task=='gsm8k' else ['test']):
            for baseline in ['plain','answer_only']:
                for mode in modes:
                    if mode==baseline:continue
                    for metric in ['p1','pk']:
                        group=f'{task}/{suite}/{mode}-vs-{baseline}/{metric}';values=[]
                        for seed in seeds:
                            differences=[]
                            for state in [30,90]:
                                delta=arrays[(task,seed,state,mode,suite)][metric]-arrays[(task,seed,state,baseline,suite)][metric]
                                comparisons[f'{group}/seed{seed}/state{state}']=bootstrap(delta,20261029+seed+state)
                                differences.append(delta)
                            interaction=differences[1]-differences[0]
                            interactions[f'{group}/seed{seed}']=bootstrap(interaction,20261029+seed)
                            values.append(float(interaction.mean()))
                        summaries[group]=seed_summary(values,seeds)
    for filename,rows in [('per_question_scores.csv',qrows),('pass_at_k_curves.csv',curverows)]:
        with (root/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    strata={}
    for task in tasks:
        for seed in seeds:
            for state in [30,90]:
                before=arrays[(task,seed,state,'reference','diagnostic')]['counts']
                groups={'sampled_all_failure':before==0,'sampled_mixed':(before>0)&(before<16),'sampled_all_success':before==16}
                for mode in modes:
                    after=arrays[(task,seed,state,mode,'diagnostic')]
                    strata[f'{task}/seed{seed}/state{state}/{mode}']={name:{'questions':int(mask.sum()),
                        'post_pass_at_1':float(after['p1'][mask].mean()) if mask.any() else None,
                        'post_pass_at_16':float(after['pk'][mask].mean()) if mask.any() else None} for name,mask in groups.items()}
    result={'status':'completed','training_order_matches':True,
            'training_order_scope':'same question set at every added update; balancing may reorder within batch',
            'metrics':metrics,'comparisons':comparisons,
            'state_interactions':interactions,'training_seed_summary':summaries,'learning_curves':learning,
            'active_scope':scope,
            'independent_diagnostic_strata':strata,'diagnostic_sampling':'128 fixed questions x16; reference seeds2000..2015, post-update seeds3000..3015; no guidance during evaluation',
            'costs':ledger(root),'scope':'fixed-teacher post-OPD effects only; no downstream RL or learned generator',
            'limitations':['training state includes optimizer history',f'{len(seeds)} training seed(s)',
                           'question bootstrap does not cover training-seed variability','exploratory multiple comparisons',
                           'guidance differs in content and length','Countdown reference expression already supplies a solution']}
    atomic(root/'study_results.json',result)
    report=f'# G3a：固定 teacher 与学生状态\n\n{len(tasks)}训练任务、2状态、{len(modes)}指导、{len(seeds)}训练种子；每分支追加150步OPD。模型无指导完成固定终测。详细逐seed结果与题目配对区间见JSON/CSV；单seed不估计训练种子方差。\n'
    for task in tasks:
        report+='\n## '+task+'\n\n| 起点 | 指导 | pass@1 | pass@32 |\n|---|---|---:|---:|\n'
        for state in [30,90]:
            for mode in modes:
                vals=[metrics[f'{task}-s{s}-state{state}-{mode}']['test'] for s in seeds]
                report+=f'| {state} | {mode} | {np.mean([v["pass_at_1"] for v in vals]):.2%} | {np.mean([v["pass_at_max"] for v in vals]):.2%} |\n'
    report+='\n主判断是同状态内指导相对普通/答案-only的收益，以及该差值在两个状态间是否改变。不能用早晚绝对分数判断教法；诊断与学习速度另列。多重比较区间为探索性，题目区间不包含训练种子不确定性。后续RL收益尚未验证。\n'
    report+=f'\n本轮分配成本（包含失败检查/训练/评测）：{result["costs"]["allocated_gpu_hours"]:.3f} GPU小时；历史实验未重复计入。\n'
    (root/'G3_RESULTS.md').write_text(report)
    print(json.dumps({'status':'completed','evaluation_jobs':len(all_jobs),'question_rows':len(qrows),'curve_rows':len(curverows)}))


if __name__=='__main__':main()
