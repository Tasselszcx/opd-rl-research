"""Audit every frozen training reference before any G3 GPU comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from transformers import AutoTokenizer
from reasoning_gym.utils import SYSTEM_PROMPTS
from state_guidance.data import read_rows
from state_guidance.guides import reference_text


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',required=True);p.add_argument('--tokenizer',required=True)
    a=p.parse_args();d=a.root/'data'/a.task
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    rows=read_rows(d/'guidance.jsonl');stats={k:[] for k in ['plain','answer_only','full_reasoning','key_steps','error_reserved']}
    for row in rows:
        for mode in stats:
            question=row['question']
            if mode!='plain':
                actual_mode='answer_only' if mode=='error_reserved' else mode
                reference,_=reference_text(row,actual_mode)
                if mode=='error_reserved':reference+='\nFeedback on the current attempt:\n'+'x '*512
                question+='\n\n'+reference+'\nUse the reference to reason correctly, without mentioning that it was supplied.'
            prompt=tok.apply_chat_template([{'role':'system','content':SYSTEM_PROMPTS['default']},
                                            {'role':'user','content':question}],tokenize=False,
                                            add_generation_prompt=True,enable_thinking=False)
            if a.task!='gsm8k':prompt+="Let's think step by step.\n"
            stats[mode].append(len(tok.encode(prompt,add_special_tokens=False)))
    maximum=max(max(v) for v in stats.values());limit=max(4096,2048*math.ceil((maximum+16)/2048))
    model=json.loads((Path(a.tokenizer)/'config.json').read_text())
    assert limit+(8192 if a.task=='gsm8k' else 4096)<=model['max_position_embeddings']
    report={'status':'passed','task':a.task,'examples':len(rows),'max_prompt_length':limit,
            'error_hint_reserved_tokens':512,'guidance_sha256':hashlib.sha256((d/'guidance.jsonl').read_bytes()).hexdigest(),
            'stats':{k:{'min':min(v),'max':max(v),'mean':sum(v)/len(v)} for k,v in stats.items()}}
    target=d/'prompt_audit.json'
    if target.exists():assert json.loads(target.read_text())==report
    else:target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
