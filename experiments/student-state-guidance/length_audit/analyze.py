"""Post-hoc G2 diagnostics: fixed G1 checkpoints, same trajectories at two token budgets."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics
import numpy as np
import reasoning_gym
from reasoning_gym.utils import extract_answer
from transformers import AutoTokenizer
from length_audit.prefix_scoring import prefix_text

ROOT=Path(os.environ['REPRO_ROOT'])
MODES=['plain','key_step','full_solution']
PHASES=['opd','rl']
SCENARIOS=['g1_cached_2048','g2_same_trajectory_2048','g2_full_4096']

def summaries(counts):
    return {str(k):float(np.mean([1-math.comb(4-int(c),k)/math.comb(4,k) for c in counts])) for k in range(1,5)}

def paired(delta,rng):
    draws=rng.choice(delta,size=(10000,512),replace=True).mean(axis=1)
    return {'difference':float(np.mean(delta)),'paired_question_ci95':np.quantile(draws,[.025,.975]).tolist()}

def main():
    plan=json.loads((ROOT/'plan.json').read_text());baseline=Path(plan['baseline_root'])
    expected={x['index']:x for x in map(json.loads,(ROOT/'data/pilot_test.jsonl').read_text().splitlines())}
    assert set(expected)==set(range(512))
    scorer=reasoning_gym.create_dataset('knights_knaves',seed=42,size=1,n_people=6,depth_constraint=2,width_constraint=2)
    tokenizer=AutoTokenizer.from_pretrained(plan['endpoints'][0]['model'],trust_remote_code=False)
    arrays={s:{} for s in SCENARIOS};metrics={s:{} for s in SCENARIOS};question_rows=[];curves=[];diagnostics={}
    for endpoint in plan['endpoints']:
        name=endpoint['name'];folder=ROOT/'evaluations'/name
        assert json.loads((folder/'launcher_record.json').read_text())['status']=='completed'
        matrix={s:np.zeros((512,4)) for s in SCENARIOS};long_rows={};old_rows={}
        for target,directory in [(old_rows,baseline/'evaluations'/name),(long_rows,folder)]:
            for file in sorted(directory.glob('rank_*.jsonl')):
                for row in map(json.loads,file.read_text().splitlines()):
                    i,j=row['index'],row['sample'];assert (i,j) not in target
                    assert row['suite']=='pilot_test' and row['sampling_seed']==42+j
                    assert row['question_sha256']==expected[i]['question_sha256']
                    assert row['expected_answer']==expected[i]['answer']
                    assert row['score'] in (0,1)
                    target[i,j]=row
            assert set(target)=={(i,j) for i in range(512) for j in range(4)}
        gains=losses=0;replacement_count=0
        for (i,j),row in long_rows.items():
            ids=row['response_token_ids'];assert len(ids)==row['response_tokens']<=4096
            assert row['max_tokens']==4096
            full_answer=extract_answer(row['response'],tag_name='answer') or ''
            assert full_answer==row['predicted_answer']
            assert scorer.score_answer(full_answer,expected[i])==row['score']
            prefix=prefix_text(tokenizer,ids,row['response'],prompt_token_ids=row['prompt_token_ids'])
            assert prefix==row['prefix_2048_text']
            prefix_answer=extract_answer(prefix,tag_name='answer') or ''
            assert prefix_answer==row['prefix_2048_answer']
            prefix_score=scorer.score_answer(prefix_answer,expected[i]);assert prefix_score==row['prefix_2048_score']
            assert row['prefix_2048_cut']==(len(ids)>2048)
            if len(ids)<=2048:assert row['score']==prefix_score
            gains+=row['score']>prefix_score;losses+=row['score']<prefix_score
            replacement_count+='\ufffd' in prefix
            matrix['g1_cached_2048'][i,j]=old_rows[i,j]['score']
            matrix['g2_same_trajectory_2048'][i,j]=prefix_score
            matrix['g2_full_4096'][i,j]=row['score']
        for scenario,m in matrix.items():
            counts=m.sum(axis=1).astype(int);curve=summaries(counts)
            arrays[scenario][name]={'pass_at_1':m.mean(axis=1),'pass_at_4':(counts>0).astype(float)}
            metrics[scenario][name]={'pass_at_1':float(m.mean()),'pass_at_4':float((counts>0).mean()),
                                      'pass_at_k':curve,'questions':512,'responses':2048}
            question_rows.extend({'scenario':scenario,'endpoint':name,'index':i,'question_sha256':expected[i]['question_sha256'],
                                  'correct_count':int(c),'samples':4} for i,c in enumerate(counts))
            curves.extend({'scenario':scenario,'endpoint':name,'k':k,'pass_at_k':curve[str(k)]} for k in range(1,5))
        vals=list(long_rows.values());old=list(old_rows.values())
        diagnostics[name]={'mean_tokens_g1':statistics.mean(x['response_tokens'] for x in old),
            'mean_tokens_g2':statistics.mean(x['response_tokens'] for x in vals),
            'g1_length_finish_fraction':statistics.mean(x['finish_reason']=='length' for x in old),
            'g2_length_finish_fraction':statistics.mean(x['finish_reason']=='length' for x in vals),
            'g2_beyond_2048_fraction':statistics.mean(x['response_tokens']>2048 for x in vals),
            'g2_missing_answer_fraction':statistics.mean(not x['predicted_answer'] for x in vals),
            'prefix_missing_answer_fraction':statistics.mean(not x['prefix_2048_answer'] for x in vals),
            'same_trajectory_responses_improved':gains,'same_trajectory_responses_worsened':losses,
            'prefix_replacement_character_count':replacement_count}
    rng=np.random.default_rng(20261024);length_gain={};cached_difference={};guidance={};interactions={}
    for endpoint in plan['endpoints']:
        name=endpoint['name'];length_gain[name]={};cached_difference[name]={}
        for metric in ['pass_at_1','pass_at_4']:
            full=arrays['g2_full_4096'][name][metric]
            length_gain[name][metric]=paired(full-arrays['g2_same_trajectory_2048'][name][metric],rng)
            cached_difference[name][metric]=paired(full-arrays['g1_cached_2048'][name][metric],rng)
    for scenario in SCENARIOS:
        guidance[scenario]={}
        for mode in ['key_step','full_solution']:
            guidance[scenario][mode]={}
            for metric in ['pass_at_1','pass_at_4']:
                guidance[scenario][mode][metric]=paired(arrays[scenario][mode+'-rl'][metric]-arrays[scenario]['plain-rl'][metric],rng)
    for mode in ['key_step','full_solution']:
        interactions[mode]={}
        for metric in ['pass_at_1','pass_at_4']:
            full=arrays['g2_full_4096'][mode+'-rl'][metric]-arrays['g2_full_4096']['plain-rl'][metric]
            prefix=arrays['g2_same_trajectory_2048'][mode+'-rl'][metric]-arrays['g2_same_trajectory_2048']['plain-rl'][metric]
            interactions[mode][metric]=paired(full-prefix,rng)
    costs=[]
    for p in sorted((ROOT/'evaluations').rglob('launcher_record.json')):
        record=json.loads(p.read_text());assert record['status'] in ['completed','failed']
        costs.append({'attempt':str(p.parent.relative_to(ROOT)),'status':record['status'],
                      'allocated_gpu_hours':record['allocated_gpu_hours'],'estimated':record.get('estimated',False)})
    total=sum(x['allocated_gpu_hours'] for x in costs)
    estimated=sum(x['allocated_gpu_hours'] for x in costs if x['estimated'])
    if plan['gpu_hour_cap'] is not None:assert total<=plan['gpu_hour_cap']
    result={'status':'completed','diagnostic':True,'test_previously_observed':True,'training_changed':False,
        'metrics':metrics,'within_trajectory_length_gain':length_gain,'vs_cached_g1_difference':cached_difference,
        'post_rl_guidance_vs_plain':guidance,'post_rl_gap_change_from_extra_tokens':interactions,
        'diagnostics':diagnostics,'cost_records':costs,'total_allocated_gpu_hours':total,'gpu_hour_cap':plan['gpu_hour_cap'],
        'measured_allocated_gpu_hours':total-estimated,'estimated_gpu_hours':estimated,
        'bootstrap':{'resamples':10000,'seed':20261024,'unit':'paired question','multiple_comparison_adjustment':False},
        'scope':'Post-hoc inference-length diagnosis on fixed G1 checkpoints and the already-observed G1 test. No fresh generalization claim; does not isolate training-time truncation.',
        'prefix_definition':'Replay first 2048 generated tokens of the same long sample with installed vLLM V0 detokenization. This is not an independent short-limit decoder run.',
        'plan_sha256':hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()}
    (ROOT/'study_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('per_question_scores.csv',question_rows),('pass_at_k_curves.csv',curves)]:
        with (ROOT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report='# G2：固定checkpoint的输出长度诊断\n\n这是对已看过的G1测试集的事后诊断；不重训、不增加训练种子、不选择checkpoint。仅新增4096上限生成，2048反事实评分来自同一轨迹的前2048 token；旧G1输出另列，不能假定两次生成的前缀完全相同。\n'
    report+='\n| 固定终点 | 旧G1 2048 p@1 / p@4 | 同轨迹2048 p@1 / p@4 | 完整4096 p@1 / p@4 |\n|---|---:|---:|---:|\n'
    for ep in plan['endpoints']:
        name=ep['name'];vals=[]
        for s in SCENARIOS:
            m=metrics[s][name];vals.append(f"{m['pass_at_1']:.2%} / {m['pass_at_4']:.2%}")
        report+='| '+name+' | '+' | '.join(vals)+' |\n'
    report+='\n## 同轨迹延长的差值\n'
    for name,ms in length_gain.items():
        for metric,x in ms.items():
            lo,hi=x['paired_question_ci95'];report+=f"\n{name} {metric}: {100*x['difference']:+.2f}个百分点，95%题目配对区间[{100*lo:+.2f}, {100*hi:+.2f}]。\n"
    report+=f'\n本轮累计计账 {total:.3f} GPU小时，其中实测 {total-estimated:.3f}、保守估计上界 {estimated:.3f}。包含本轮失败尝试；G1历史成本不重复计入。用户已取消预算上限，实验范围仍固定六个终点。\n'
    report+='\n限制：同轨迹对照只衡量推理时继续生成的影响，不能排除训练时2048上限已经改变学习轨迹。单训练种子；题目区间不含训练随机性，且未校正多重比较。指导条件同时改变信息量与表达方式。诊断结果不能冒充新独立测试的泛化提升。\n'
    (ROOT/'LENGTH_AUDIT_RESULTS.md').write_text(report)
    print(json.dumps({'status':'completed','endpoints':6,'new_responses':12288,'cost':total}),flush=True)

if __name__=='__main__':main()
