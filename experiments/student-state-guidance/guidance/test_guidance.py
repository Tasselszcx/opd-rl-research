from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset
from reasoning_gym.utils import SYSTEM_PROMPTS
from guidance.proofs import make_guidance
from guidance.teacher_inputs import conditioned_reference_batch

ROOT = Path(os.environ['REPRO_ROOT'])

@pytest.fixture(scope='module')
def entries():
    return [json.loads(x) for x in (ROOT/'data/train.jsonl').read_text().splitlines()[:8]]

@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained('/tmp/sequential-kk-models/Qwen3-1.7B-Base', local_files_only=True)

def build_batch(entries, tokenizer):
    chats, ids, masks = [], [], []
    for e in entries[:2]:
        chat = [{'role':'system','content':SYSTEM_PROMPTS['default']}, {'role':'user','content':e['question']}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) + "Let's think step by step.\n"
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        chats.append(chat); ids.append([tokenizer.pad_token_id]*(1024-len(tokens))+tokens)
        masks.append([0]*(1024-len(tokens))+[1]*len(tokens))
    prompts = torch.tensor(ids)
    responses = torch.tensor([[10,11,12,tokenizer.pad_token_id],[13,14,tokenizer.pad_token_id,tokenizer.pad_token_id]])
    attention = torch.cat([torch.tensor(masks),torch.tensor([[1,1,1,0],[1,1,0,0]])],dim=-1)
    raw = np.empty(2,dtype=object);raw[:]=chats
    return DataProto.from_dict(tensors={'prompts':prompts,'responses':responses,
        'input_ids':torch.cat([prompts,responses],dim=-1),'attention_mask':attention,
        'position_ids':compute_position_id_with_mask(attention)},non_tensors={
        'index':np.array([e['index'] for e in entries[:2]],dtype=object),
        'question':np.array([e['question'] for e in entries[:2]],dtype=object), 'raw_prompt':raw})

def test_proofs_match_independent_author_solver(entries):
    for e in entries:
        guide=make_guidance(e)
        solutions=KnightsKnavesDataset.find_solution(e['metadata']['statements'])
        assert len(solutions)==1 and list(solutions[0])==guide['proof']['unique_solution']
        assert guide['proof']['prefix_survivor_counts'][-1]==1
        assert e['answer'] in guide['full_solution'] and e['answer'] not in guide['key_step']

@pytest.mark.parametrize('mode',['key_step','full_solution'])
def test_teacher_only_conditioning_keeps_student_and_response_alignment(entries,tokenizer,mode):
    batch=build_batch(entries,tokenizer);before=deepcopy(batch)
    teacher,metrics=conditioned_reference_batch(batch,tokenizer,mode,str(ROOT/'data/guidance.jsonl'),1024,"Let's think step by step.\n")
    for key in batch.batch.keys():assert torch.equal(batch.batch[key],before.batch[key])
    assert batch.non_tensor_batch['raw_prompt'].tolist()==before.non_tensor_batch['raw_prompt'].tolist()
    assert torch.equal(teacher.batch['responses'],batch.batch['responses'])
    assert torch.equal(teacher.batch['input_ids'][:,-4:],batch.batch['responses'])
    assert torch.equal(teacher.batch['attention_mask'][:,-4:],batch.batch['attention_mask'][:,-4:])
    assert not torch.equal(teacher.batch['input_ids'][:,:1024],batch.batch['input_ids'][:,:1024])
    assert metrics['guidance/conditioned_sequences']==2 and metrics['guidance/response_alignment_ok']==1
    for i in range(2):
        original=tokenizer.decode(batch.batch['prompts'][i],skip_special_tokens=False)
        changed=tokenizer.decode(teacher.batch['prompts'][i],skip_special_tokens=False)
        assert 'Reference information for your reasoning:' not in original
        assert 'Reference information for your reasoning:' in changed

def test_plain_is_identity_without_guidance_file(entries,tokenizer):
    batch=build_batch(entries,tokenizer)
    teacher,m=conditioned_reference_batch(batch,tokenizer,'plain','nonexistent',1024,'')
    assert teacher is batch and m['guidance/conditioned_sequences']==0

def test_misaligned_question_is_rejected(entries,tokenizer):
    batch=build_batch(entries,tokenizer);batch.non_tensor_batch['question'][0]='wrong question'
    with pytest.raises(AssertionError):
        conditioned_reference_batch(batch,tokenizer,'key_step',str(ROOT/'data/guidance.jsonl'),1024,'')

def test_no_silent_teacher_truncation(entries,tokenizer):
    with pytest.raises(ValueError,match='exceeds'):
        conditioned_reference_batch(build_batch(entries,tokenizer),tokenizer,'full_solution',str(ROOT/'data/guidance.jsonl'),8,'')

def test_splits_and_guide_hashes():
    sets={}
    for name in ['train','meta_val','pilot_test']:
        rows=[json.loads(x) for x in (ROOT/'data'/(name+'.jsonl')).read_text().splitlines()]
        sets[name]={x['question_sha256'] for x in rows}
        assert all(hashlib.sha256(x['question'].encode()).hexdigest()==x['question_sha256'] for x in rows)
    assert not sets['train'] & sets['meta_val'] and not sets['train'] & sets['pilot_test']
    assert not sets['meta_val'] & sets['pilot_test']
