"""Verifiable guidance families. Templates are fixed; no learned selector/generator."""
import ast
import json
import re
from itertools import product
from state_guidance.data import arithmetic, extract_final, score_answer
from guidance.proofs import evaluate, make_guidance


def kk_guides(entry):
    m=entry['metadata'];names=m['names'];statements=m['statements'];solution=m['solution']
    evidence=[];lines=['Let T denote a truth-teller and F a liar. In name order '+', '.join(names)+
                      ', a consistent assignment must make each statement equal its speaker role.']
    for assignment in product([False,True],repeat=len(names)):
        bad=next((i for i,s in enumerate(statements) if evaluate(s,assignment)!=assignment[i]),None)
        bits=''.join('T' if v else 'F' for v in assignment)
        if bad is None:
            assert list(assignment)==list(solution)
            lines.append(bits+': every speaker is consistent; this assignment survives.')
        else:
            truth=evaluate(statements[bad],assignment)
            lines.append(f"{bits}: reject; {names[bad]}'s statement is {'true' if truth else 'false'} but its role requires {'true' if assignment[bad] else 'false'}.")
        evidence.append({'assignment':list(assignment),'first_violated_speaker':bad})
    lines.append('All assignments have been checked. The unique survivor gives the stated answer.')
    key=make_guidance(entry)
    return {'full_reasoning':'\n'.join(lines),'key_steps':key['key_step'],
            'proof':{'assignment_checks':evidence,'key_proof':key['proof']}}


def zebra_guides(entry):
    from reasoning_gym.logic.contrib.logic_puzzle.sat_utils import itersolve
    m=entry['metadata'];base=m['base_cnf'];clues=m['clues'];known=[];steps=[]
    solution=m['solution_positions']
    # Every new placement is entailed by the original clues and previously proved placements.
    for variable,house in sorted(solution.items(),key=lambda x:(x[0].startswith('Name.'),x[0])):
        proposition=f'{variable} {house}'
        active=list(range(len(clues)))
        def entails(indices):
            cnf=base+known+[c for i in indices for c in clues[i]['cnf']]+[['~'+proposition]]
            return next(itersolve(cnf),None) is None
        assert entails(active)
        for i in list(active):
            candidate=[j for j in active if j!=i]
            if entails(candidate):active=candidate
        witnesses=[clues[i]['text'] for i in active]
        label=variable.split('.',1)[-1].replace('_',' ')
        reason='; '.join(witnesses) if witnesses else 'the one-value-per-house rule and previous deductions'
        text=f"{label} must be in House {house}. Use: {reason}. Together with prior placements, assuming any other house makes these constraints unsatisfiable."
        steps.append({'variable':variable,'house':house,'clue_indices':active,'text':text})
        known.append([proposition])
    assert len(steps)==len(solution)
    return {'full_reasoning':'Apply each clue together with uniqueness within every attribute category.\n'+
            '\n'.join(f'{i+1}. '+s['text'] for i,s in enumerate(steps)),
            'key_steps':steps[0]['text'],'proof':{'entailed_placements':steps,'all_verified_by_sat':True}}


def countdown_guides(entry):
    expression=entry['metadata']['expression'];tree=ast.parse(expression,mode='eval');steps=[]
    def walk(n):
        if isinstance(n,ast.BinOp):
            walk(n.left);walk(n.right)
            text=ast.unparse(n);value=arithmetic(text)
            steps.append(text+' = '+str(value))
    walk(tree.body)
    assert arithmetic(expression)==entry['metadata']['target'] and steps
    return {'full_reasoning':'Use each given number exactly once in the reference expression.\n'+
            '\n'.join(f'{i+1}. {s}' for i,s in enumerate(steps))+
            '\nThe final value equals the target; the literal-number multiset matches the question.',
            'key_steps':'A useful intermediate calculation is '+steps[0]+'.',
            'proof':{'exact_rational_steps':steps}}


def gsm8k_guides(entry):
    full=entry['metadata']['reference_solution'];candidates=[]
    for match in re.finditer(r'<<([^<>]+)=([^<>]+)>>',full):
        left,right=match.groups()
        try:
            if arithmetic(left.strip())!=arithmetic(right.strip()):continue
        except (SyntaxError,ValueError,ZeroDivisionError,OverflowError):continue
        # Retain the semantic sentence containing the verified calculation.
        start=full.rfind('\n',0,match.start())+1;end=full.find('\n',match.end())
        candidates.append({'equation':match.group(0),'text':full[start:end if end>=0 else len(full)]})
    key=candidates[0]['text'] if candidates else full.split('\n')[0]
    return {'full_reasoning':full,'key_steps':key,
            'proof':{'reference':'official GSM8K training solution','verified_arithmetic':candidates,
                     'key_arithmetic_verified':bool(candidates),
                     'semantic_reasoning':'official reference; not machine-proof-checked'}}


def build_guidance(entry):
    return {'kk':kk_guides,'zebra':zebra_guides,'countdown':countdown_guides,'gsm8k':gsm8k_guides}[entry['task']](entry)


def error_hint(entry,response):
    """Only asserted, mechanically checkable errors; all other cases use answer-only."""
    answer=extract_final(response,entry['task'])
    if score_answer(answer,entry)==1.0:return None,'correct_answer_fallback'
    task=entry['task'];m=entry['metadata']
    if task=='kk':
        from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset
        pairs=KnightsKnavesDataset._normalize_answer(answer)
        roles=m['knight_knave_terms'];true_role=roles['a_knight'].split()[-1].lower()
        false_role=roles['a_knave'].split()[-1].lower();assignment=[]
        for name in m['names']:
            values={r for n,r in pairs if n==name.lower()}
            if len(values)!=1 or not values <= {true_role,false_role}:return None,'unparsed_assignment_fallback'
            assignment.append(next(iter(values))==true_role)
        for i,s in enumerate(m['statements']):
            truth=evaluate(s,assignment)
            if truth!=assignment[i]:
                return (f"The proposed final role assignment is inconsistent: {m['names'][i]}'s statement evaluates to {'true' if truth else 'false'}, whereas the proposed role requires {'true' if assignment[i] else 'false'}. Recheck this speaker's statement against the assigned roles before propagating further deductions.",'verified_role_contradiction')
    elif task=='zebra':
        allowed=[str(v).lower() for v in m['generator_instance']['solution']['table_rows'][0][1:]]
        # A wrong final name is provably incompatible with the unique CSP solution;
        # do not pretend this identifies the first error inside free-form reasoning.
        names=[row[1].lower() for row in m['generator_instance']['solution']['table_rows']]
        if answer.lower().strip() in names:
            return ('The proposed final person does not match the unique assignment satisfying the clues. Recheck the name-to-house placement for the house asked in the question, including uniqueness and adjacency constraints.','verified_final_assignment_conflict')
    elif task=='countdown':
        from collections import Counter
        nums=[int(x) for x in re.findall(r'\b\d+\b',answer)]
        if answer and Counter(nums)!=Counter(m['numbers']):
            return ('The proposed expression does not use the required number multiset. Check for repeated, omitted, or additional numbers; each supplied occurrence must be used exactly once.','verified_number_multiset_error')
        try:
            value=arithmetic(answer)
            if value!=m['target']:
                return (f'The proposed expression evaluates to {value}, not the target {m["target"]}. Recheck operator precedence and the intermediate arithmetic.','verified_target_mismatch')
        except (SyntaxError,ValueError,ZeroDivisionError,OverflowError):pass
    elif task=='gsm8k':
        # Require a complete numeric equality; never extract a truncated numeric suffix.
        for match in re.finditer(r'(?<![\w.])([\d.,() +*/\-]{3,100})\s*=\s*([-+]?\d+(?:\.\d+)?)(?![\w./\d])',response):
            left,right=match.groups();left=left.strip()
            try:
                actual=arithmetic(left);claimed=arithmetic(right)
                if actual!=claimed:
                    return (f'The arithmetic equality "{left} = {right}" is incorrect: its left side equals {actual}. Correct this calculation and propagate the corrected quantity through the remaining steps.','verified_arithmetic_error')
            except (SyntaxError,ValueError,ZeroDivisionError,OverflowError):continue
    return None,'no_verified_error_fallback'


def reference_text(row,mode,response=''):
    common='Reference answer: '+row['answer']
    if mode=='answer_only':return common,'answer_only'
    if mode in ('full_reasoning','key_steps'):return common+'\nReasoning reference:\n'+row[mode],mode
    if mode=='error_targeted':
        hint,status=error_hint(row,response)
        return common+('\nFeedback on the current attempt:\n'+hint if hint else ''),status
    raise ValueError(mode)
