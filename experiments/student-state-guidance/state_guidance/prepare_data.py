"""Prepare immutable, disjoint G3 splits, recording every generator decision."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from random import Random
import time

from state_guidance.data import sha_text, score_answer

TASK_SEEDS = {'kk': 5000000, 'zebra': 10000000, 'countdown': 15000000}
SPLIT_OFFSETS = {'train': 0, 'diagnostic': 1000000, 'dev': 2000000, 'test': 3000000}
SIZES = {'train': 20000, 'diagnostic': 128, 'dev': 512, 'test': 1024}


def canonical(entry, task):
    m = entry['metadata']
    if task == 'kk': value = m['statements']
    elif task == 'zebra':
        value = [sorted(m['generator_instance']['core_rules']),
                 m['generator_instance']['questions'][0]['question']]
    elif task == 'countdown': value = [sorted(m['numbers']), m['target']]
    else: value = ' '.join(entry['question'].lower().split())
    return sha_text(json.dumps(value, sort_keys=True))


def generate(args):
    task, seed = args
    import reasoning_gym
    if task == 'kk':
        ds = reasoning_gym.create_dataset('knights_knaves', seed=seed, size=1,
                                          n_people=6, depth_constraint=2, width_constraint=2)
        entry = ds[0]
    elif task == 'countdown':
        ds = reasoning_gym.create_dataset('countdown', seed=seed, size=1, min_numbers=3, max_numbers=4)
        entry = ds[0]
    elif task == 'zebra':
        from reasoning_gym.logic.contrib.logic_puzzle.generate import generate_puzzle
        instance, puzzle = generate_puzzle(Random(seed), 4, 4)
        entry = dict(question=str(puzzle)+'\n'+instance['questions'][0]['question']+
                     '? Provide only the name of the person as your final answer.',
                     answer=instance['questions'][0]['answer'], metadata={
                         'source_dataset':'zebra_puzzles', 'difficulty':{'num_people':4,'num_characteristics':4},
                         'generator_instance':instance, 'cnf':puzzle.as_cnf(),
                         'base_cnf':puzzle.constraints,
                         'clues':[{'text':str(c),'cnf':c.as_cnf()} for c in sorted(puzzle.clues)],
                         'solution_positions':{str(k):v for k,v in puzzle.solution.items()}})
    else: raise ValueError(task)
    entry.update(task=task, generator_seed=seed, question_sha256=sha_text(entry['question']))
    entry['structure_sha256'] = canonical(entry, task)
    assert score_answer(entry['answer'], entry) == 1.0, (task, seed, entry['answer'])
    assert score_answer('', entry) == 0.0
    return entry


def freeze(path, rows):
    raw = ('\n'.join(json.dumps(x,sort_keys=True,ensure_ascii=False) for x in rows)+'\n').encode()
    if path.exists(): assert path.read_bytes() == raw, 'Existing frozen split differs'
    else:
        tmp=path.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(path)
    return {'count':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--task',choices=['kk','zebra','countdown','gsm8k'],required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--raw-data',type=Path)
    a=p.parse_args();out=a.root/'data'/a.task;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        for name,item in manifest['splits'].items():
            assert hashlib.sha256((out/(name+'.jsonl')).read_bytes()).hexdigest()==item['sha256']
        print(json.dumps({'status':'already_frozen','task':a.task}));return
    seen_q=set();seen_s=set();splits={};skipped={};start=time.time()
    if a.task=='gsm8k':
        assert a.raw_data
        source={s:[json.loads(x) for x in (a.raw_data/f'gsm8k-{s}.jsonl').read_text().splitlines()] for s in ['train','test']}
        assert len(source['train'])==7473 and len(source['test'])==1319
        order=list(range(7473));Random(20261029).shuffle(order)
        positions={'diagnostic':order[:128],'dev':order[128:640],'train':order[640:],'test':list(range(1319))}
        for name,positions in positions.items():
            rows=[]
            for index,source_index in enumerate(positions):
                raw=source['test' if name=='test' else 'train'][source_index]
                answer=raw['answer'].split('####')[-1].strip()
                entry=dict(index=index,task='gsm8k',question=raw['question'],answer=answer,
                           metadata={'source_index':source_index,'source_split':'test' if name=='test' else 'train',
                                     'reference_solution':raw['answer'].split('####')[0].strip()})
                entry.update(question_sha256=sha_text(entry['question']))
                entry['structure_sha256']=canonical(entry,'gsm8k')
                # Do not silently discard official test questions if overlap is found.
                assert entry['question_sha256'] not in seen_q and entry['structure_sha256'] not in seen_s
                seen_q.add(entry['question_sha256']);seen_s.add(entry['structure_sha256'])
                assert score_answer(answer,entry)==1.0;rows.append(entry)
            splits[name]=freeze(out/(name+'.jsonl'),rows)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for name,count in SIZES.items():
                rows=[];cursor=TASK_SEEDS[a.task]+SPLIT_OFFSETS[name];skipped[name]=0
                while len(rows)<count:
                    candidates=[(a.task,s) for s in range(cursor,cursor+min(256,count-len(rows)))]
                    cursor+=len(candidates)
                    for entry in pool.map(generate,candidates,chunksize=8):
                        if entry['question_sha256'] in seen_q or entry['structure_sha256'] in seen_s:
                            skipped[name]+=1;continue
                        seen_q.add(entry['question_sha256']);seen_s.add(entry['structure_sha256'])
                        entry['index']=len(rows);rows.append(entry)
                    if len(rows)%1024==0:print(json.dumps({'task':a.task,'split':name,'count':len(rows)}),flush=True)
                splits[name]=freeze(out/(name+'.jsonl'),rows)
    manifest={'task':a.task,'splits':splits,'skipped_duplicates':skipped,'zero_question_overlap':True,
              'zero_structure_overlap':True,'canonical_definition':'KK indexed statement trees; Zebra sorted clue text and target question; Countdown number multiset and target; GSM normalized question text. Not a complete logical-equivalence test.',
              'split_seeds':{k:TASK_SEEDS.get(a.task,0)+v for k,v in SPLIT_OFFSETS.items()},
              'gsm8k_partition_seed':20261029,'seconds':time.time()-start}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'completed',**manifest}),flush=True)


if __name__=='__main__':main()
