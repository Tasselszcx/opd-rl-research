"""Teacher-only conditioning with exact student response/mask preservation."""
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import torch
from verl.utils.model import compute_position_id_with_mask
from state_guidance.data import sha_text
from state_guidance.guides import reference_text

_CACHE={}


def conditioned_reference_batch(batch,tokenizer,mode,guidance_file,max_prompt_length,cot_suffix):
    if mode=='plain':return batch,{'guidance/conditioned_sequences':0,'guidance/response_alignment_ok':1}
    assert mode in ('answer_only','full_reasoning','key_steps','error_targeted')
    path=str(Path(guidance_file).resolve())
    if path not in _CACHE:
        with Path(path).open() as stream:_CACHE[path]={x['index']:x for x in map(json.loads,stream)}
    refs=_CACHE[path];all_ids=[];all_masks=[];lengths=[];counts=Counter()
    decoded=(tokenizer.batch_decode(batch.batch['responses'].detach().cpu().tolist(),skip_special_tokens=True)
             if mode=='error_targeted' else ['']*len(batch))
    for i,(index,question,raw_chat) in enumerate(zip(batch.non_tensor_batch['index'],batch.non_tensor_batch['question'],batch.non_tensor_batch['raw_prompt'])):
        row=refs[int(index)];assert sha_text(question)==row['question_sha256']
        response=decoded[i]
        reference,status=reference_text(row,mode,response);counts[status]+=1
        chat=deepcopy(list(raw_chat));assert chat[-1]['content']==question
        chat[-1]['content']+='\n\n'+reference+'\nUse the reference to reason correctly, without mentioning that it was supplied.'
        prompt=tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=False)+(cot_suffix or '')
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)>max_prompt_length:raise ValueError('G3 teacher prompt would truncate: '+str(len(ids)))
        lengths.append(len(ids));padding=max_prompt_length-len(ids)
        all_ids.append([tokenizer.pad_token_id]*padding+ids);all_masks.append([0]*padding+[1]*len(ids))
    teacher=deepcopy(batch);device=batch.batch['input_ids'].device
    prompts=torch.tensor(all_ids,dtype=batch.batch['input_ids'].dtype,device=device)
    masks=torch.tensor(all_masks,dtype=batch.batch['attention_mask'].dtype,device=device)
    response_length=batch.batch['responses'].shape[-1];response_mask=batch.batch['attention_mask'][:,-response_length:]
    teacher.batch['input_ids']=torch.cat([prompts,batch.batch['responses']],dim=-1)
    teacher.batch['attention_mask']=torch.cat([masks,response_mask],dim=-1)
    teacher.batch['position_ids']=compute_position_id_with_mask(teacher.batch['attention_mask'])
    if 'prompts' in teacher.batch:teacher.batch['prompts']=prompts
    assert torch.equal(teacher.batch['responses'],batch.batch['responses'])
    assert torch.equal(teacher.batch['input_ids'][:,-response_length:],batch.batch['responses'])
    assert torch.equal(teacher.batch['attention_mask'][:,-response_length:],response_mask)
    metrics={'guidance/conditioned_sequences':len(all_ids),'guidance/response_alignment_ok':1,
             'guidance/teacher_prompt_tokens_mean':sum(lengths)/len(lengths),'guidance/teacher_prompt_tokens_max':max(lengths)}
    metrics.update({'guidance/cases_'+k:v for k,v in counts.items()})
    return teacher,metrics
