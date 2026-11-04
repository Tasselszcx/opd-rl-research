"""Frozen G3 endpoint scoring, with no teacher-visible reference in student prompts."""
import argparse
import json
import os
from pathlib import Path
import time
from reasoning_gym.utils import SYSTEM_PROMPTS
from state_guidance.data import read_rows,extract_final,score_answer,sha_text
from state_guidance.guides import error_hint


def suite_settings(samples,suite):
    if suite=='diagnostic':return 16,(2000 if samples==16 else 3000)
    return samples,42


def entries_for(root,task,suite):
    if suite=='math500':
        rows=[]
        with (root/'source_data/math500-test.jsonl').open() as f:
            for i,line in enumerate(f):
                raw=json.loads(line)
                rows.append({'index':i,'task':'math500','question':raw['problem'],'answer':raw['answer'],
                             'question_sha256':sha_text(raw['problem']),'metadata':{'subject':raw.get('subject'),'level':raw.get('level')}})
        assert len(rows)==500
        return rows
    return read_rows(root/'data'/task/(suite+'.jsonl'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--task',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--samples',type=int,required=True);p.add_argument('--suites',nargs='+',required=True)
    p.add_argument('--prompt-cap',type=int,required=True);a=p.parse_args()
    from vllm import LLM,SamplingParams
    root=Path(os.environ['REPRO_ROOT']);max_tokens=8192 if a.task=='gsm8k' else 4096
    path=a.output/f'rank_{a.rank}.jsonl'
    if path.exists():raise RuntimeError('Existing evaluation retained')
    a.output.mkdir(parents=True,exist_ok=True)
    llm=LLM(model=a.model,tensor_parallel_size=1,dtype='bfloat16',seed=42,
            max_model_len=a.prompt_cap+max_tokens,max_num_seqs=64,max_num_batched_tokens=16384,
            gpu_memory_utilization=.6,enforce_eager=True,trust_remote_code=False)
    tokenizer=llm.get_tokenizer();written=0;start=time.time()
    with path.open('w',buffering=1) as f:
        for suite in a.suites:
            rows=[x for x in entries_for(root,a.task,suite) if x['index']%4==a.rank]
            prompts=[]
            for row in rows:
                prompt=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM_PROMPTS['default']},
                    {'role':'user','content':row['question']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                if a.task!='gsm8k':prompt+="Let's think step by step.\n"
                assert len(tokenizer.encode(prompt,add_special_tokens=False))<=a.prompt_cap
                prompts.append(prompt)
            suite_samples,seed_start=suite_settings(a.samples,suite)
            for sample in range(suite_samples):
                params=SamplingParams(n=1,seed=seed_start+sample,temperature=.7,top_p=.8,top_k=20,max_tokens=max_tokens,
                                      stop=['</s>','<|im_end|>','<|endoftext|>'])
                predictions=llm.generate(prompts,params,use_tqdm=False)
                assert len(predictions)==len(rows)
                for row,prediction in zip(rows,predictions):
                    response=prediction.outputs[0];answer=extract_final(response.text,row['task'])
                    score=score_answer(answer,row);assert score in (0.,1.)
                    category=error_hint(row,response.text)[1] if suite=='diagnostic' else None
                    result={'suite':suite,'task':row['task'],'index':row['index'],'question_sha256':row['question_sha256'],
                            'sample':sample,'sampling_seed':seed_start+sample,'score':score,'expected_answer':row['answer'],
                            'predicted_answer':answer,'response':response.text,'token_ids':list(response.token_ids),
                            'response_tokens':len(response.token_ids),'finish_reason':response.finish_reason,
                            'stop_reason':response.stop_reason,'max_tokens':max_tokens,'error_category':category}
                    f.write(json.dumps(result,ensure_ascii=False)+'\n');written+=1
                print(json.dumps({'suite':suite,'rank':a.rank,'sample_completed':sample,'rows':written}),flush=True)
    (a.output/f'rank_{a.rank}.done.json').write_text(json.dumps({'status':'completed','rows':written,
        'seconds':time.time()-start,'model':a.model,'suites':a.suites,'samples':a.samples},indent=2))


if __name__=='__main__':main()
