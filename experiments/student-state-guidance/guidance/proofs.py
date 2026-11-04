"""Exact, deterministic guidance from the training puzzle's Boolean constraints."""
from itertools import combinations, product

def evaluate(stmt, assignment):
    op = stmt[0]
    if op == 'telling-truth': return bool(assignment[stmt[1]])
    if op == 'lying': return not assignment[stmt[1]]
    if op == 'not': return not evaluate(stmt[1], assignment)
    if op == 'and': return all(evaluate(s, assignment) for s in stmt[1:])
    if op == 'or': return any(evaluate(s, assignment) for s in stmt[1:])
    if op == '->': return not evaluate(stmt[1], assignment) or evaluate(stmt[2], assignment)
    if op == '<=>': return evaluate(stmt[1], assignment) == evaluate(stmt[2], assignment)
    raise ValueError(op)

def make_guidance(entry):
    meta = entry['metadata']
    names, statements = meta['names'], meta['statements']
    assignments = list(product([True, False], repeat=len(names)))
    valid = [[bool(a[i]) == evaluate(stmt, a) for i, stmt in enumerate(statements)] for a in assignments]
    survivors = [i for i, flags in enumerate(valid) if all(flags)]
    assert len(survivors) == 1
    solution = assignments[survivors[0]]
    assert list(solution) == list(meta['solution'])
    terms = meta['knight_knave_terms']
    roles = [terms['a_knight'] if x else terms['a_knave'] for x in solution]
    # Choose the shortest sufficient subset of statements for one forced role.
    # The tie rule is fixed, independent of model outputs or test scores.
    witness = None
    for size in range(1, len(names)+1):
        for subset in combinations(range(len(names)), size):
            keep = [i for i, flags in enumerate(valid) if all(flags[j] for j in subset)]
            assert keep
            for person in range(len(names)):
                if all(assignments[i][person] == solution[person] for i in keep):
                    witness = (subset, person, keep)
                    break
            if witness: break
        if witness: break
    subset, person, keep = witness
    key = ('Use the consistency rule: a truthful inhabitant makes a true statement and a liar makes a false one. '
           + 'Considering the statements of ' + ', '.join(names[i] for i in subset)
           + f', there are {len(keep)} admissible assignments; all make {names[person]} {roles[person]}. '
           + 'Use this one deduction and work out the remaining roles.')
    remaining = list(range(len(assignments)))
    counts = []
    for speaker in range(len(names)):
        remaining = [i for i in remaining if valid[i][speaker]]
        counts.append(len(remaining))
    full = ('Assign a truth/lie role to each inhabitant. Enforce for each person that their role equals '
            + 'the truth value of their statement. Start with ' + str(len(assignments)) + ' assignments. '
            + 'Adding the statements in listed order leaves: '
            + '; '.join(f'{name}: {count}' for name, count in zip(names, counts)) + '. '
            + 'Exactly one assignment survives: ' + entry['answer'] + ' '
            + 'Verification under this assignment: '
            + '; '.join(f"{name}'s statement is {'true' if solution[i] else 'false'}, matching {roles[i]}"
                        for i, name in enumerate(names)) + '.')
    return {'plain': '', 'key_step': key, 'full_solution': full,
            'proof': {'unique_solution': list(solution), 'prefix_survivor_counts': counts,
                      'key_statement_indices': list(subset), 'key_person': person,
                      'key_survivor_count': len(keep)}}
