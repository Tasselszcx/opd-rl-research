"""Condition only the frozen reference model; preserve student response tokens."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
from verl.utils.model import compute_position_id_with_mask

_CACHE = {}

def conditioned_reference_batch(batch, tokenizer, mode, guidance_file, max_prompt_length, cot_suffix):
    if mode == 'plain':
        return batch, {'guidance/conditioned_sequences': 0, 'guidance/response_alignment_ok': 1}
    if mode not in ('key_step', 'full_solution'):
        raise ValueError('Unknown guidance mode: ' + mode)
    path = str(Path(guidance_file).resolve())
    if path not in _CACHE:
        _CACHE[path] = {r['index']: r for r in map(json.loads, Path(path).read_text().splitlines())}
    refs = _CACHE[path]
    all_ids, all_masks, lengths = [], [], []
    for index, question, raw_chat in zip(batch.non_tensor_batch['index'],
                                        batch.non_tensor_batch['question'], batch.non_tensor_batch['raw_prompt']):
        ref = refs[int(index)]
        assert hashlib.sha256(question.encode()).hexdigest() == ref['question_sha256']
        chat = deepcopy(list(raw_chat))
        assert chat[-1]['role'] == 'user' and chat[-1]['content'] == question
        chat[-1]['content'] += ('\n\nReference information for your reasoning:\n' + ref[mode]
                               + '\nUse the reference to reason correctly, without mentioning that it was supplied.')
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) + (cot_suffix or '')
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_prompt_length:
            raise ValueError('Teacher prompt exceeds frozen length; refusing truncation')
        lengths.append(len(ids))
        padding = max_prompt_length-len(ids)
        all_ids.append([tokenizer.pad_token_id]*padding+ids)
        all_masks.append([0]*padding+[1]*len(ids))
    teacher = deepcopy(batch)
    device = batch.batch['input_ids'].device
    prompt_ids = torch.tensor(all_ids, dtype=batch.batch['input_ids'].dtype, device=device)
    prompt_masks = torch.tensor(all_masks, dtype=batch.batch['attention_mask'].dtype, device=device)
    response_length = batch.batch['responses'].shape[-1]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    teacher.batch['input_ids'] = torch.cat([prompt_ids, batch.batch['responses']], dim=-1)
    teacher.batch['attention_mask'] = torch.cat([prompt_masks, response_mask], dim=-1)
    teacher.batch['position_ids'] = compute_position_id_with_mask(teacher.batch['attention_mask'])
    if 'prompts' in teacher.batch:
        teacher.batch['prompts'] = prompt_ids
    assert torch.equal(teacher.batch['responses'], batch.batch['responses'])
    assert torch.equal(teacher.batch['input_ids'][:, -response_length:], batch.batch['responses'])
    assert torch.equal(teacher.batch['attention_mask'][:, -response_length:], response_mask)
    return teacher, {'guidance/conditioned_sequences': len(all_ids),
                     'guidance/teacher_prompt_tokens_mean': sum(lengths)/len(lengths),
                     'guidance/teacher_prompt_tokens_max': max(lengths),
                     'guidance/response_alignment_ok': 1}
