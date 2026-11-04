"""Frozen G3 datasets and deterministic task scoring; no test feedback in training."""
import ast
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_rows(path):
    # JSON strings may contain U+2028/U+0085. str.splitlines() splits those inside strings.
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert [r['index'] for r in rows] == list(range(len(rows)))
    assert all(sha_text(r['question']) == r['question_sha256'] for r in rows)
    return rows


def arithmetic(text):
    """Small, bounded arithmetic interpreter: never eval model text."""
    if len(text) > 160:
        raise ValueError('Expression too long')
    tree = ast.parse(text.replace(',', '').replace('×', '*').replace('÷', '/'), mode='eval')
    if len(list(ast.walk(tree))) > 64:
        raise ValueError('Expression too complex')
    def run(node):
        if isinstance(node, ast.Expression): return run(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            val = Fraction(str(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = run(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = run(node.left), run(node.right)
            if isinstance(node.op, ast.Add): val = a+b
            elif isinstance(node.op, ast.Sub): val = a-b
            elif isinstance(node.op, ast.Mult): val = a*b
            else: val = a/b
        else: raise ValueError('Unsupported arithmetic syntax')
        if abs(val.numerator) > 10**30 or val.denominator > 10**30:
            raise ValueError('Arithmetic outside limits')
        return val
    return run(tree)


def extract_final(text, task):
    tags = re.findall(r'<answer>\s*(.*?)\s*</answer>', text or '', flags=re.S)
    if tags: return tags[-1].strip()
    if task in ('gsm8k', 'math500'):
        final = re.findall(r'####\s*([^\n]+)', text or '')
        if final: return final[-1].strip()
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text or '')
        if boxed: return boxed[-1].strip()
    return ''


def score_answer(answer, entry):
    task = entry['task']
    if not isinstance(answer, str) or not answer.strip(): return 0.0
    if task == 'kk':
        from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset
        return float(KnightsKnavesDataset._normalize_answer(answer) ==
                     KnightsKnavesDataset._normalize_answer(entry['answer']))
    if task == 'zebra':
        return float(answer.lower().strip().replace('\n','') == entry['answer'].lower().strip().replace('\n',''))
    if task == 'countdown':
        from collections import Counter
        try:
            # Same literal-number requirement as the original task, exact rational result.
            nums = [int(x) for x in re.findall(r'\b\d+\b', answer)]
            return float(Counter(nums) == Counter(entry['metadata']['numbers']) and
                         arithmetic(answer) == entry['metadata']['target'])
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError): return 0.0
    if task == 'gsm8k':
        # Only a numeric final answer, with optional currency/commas, is accepted.
        candidate = answer.strip().replace(',','').lstrip('$').strip()
        expected = entry['answer'].replace(',','').strip()
        if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[1-9]\d*)?', candidate): return 0.0
        try: return float(Fraction(candidate) == Fraction(expected))
        except (ValueError, ZeroDivisionError): return 0.0
    if task == 'math500':
        from math_verify import parse, verify
        if re.sub(r'\s+','',answer)==re.sub(r'\s+','',entry['answer']):return 1.0
        try: return float(verify(parse('\\boxed{'+entry['answer']+'}'), parse('\\boxed{'+answer+'}')))
        except Exception: return 0.0
    raise ValueError(task)


@dataclass
class FrozenConfig:
    seed: int = 0
    size: int = 0
    def validate(self): assert self.size > 0


class FrozenTaskDataset:
    """The dataset wrapper needs only index, length and score_answer."""
    def __init__(self, path, max_examples=None):
        self.rows = read_rows(path)
        if max_examples is not None: self.rows = self.rows[:max_examples]
        self.config = FrozenConfig(size=len(self.rows))
        self.size = len(self.rows)
        self.seed = 0
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        # Solver certificates stay on disk/in the driver, not in every replicated Ray rollout.
        row=self.rows[index]
        metadata={k:v for k,v in row['metadata'].items() if k in
                  ('numbers','target','source_dataset','source_index','source_split','difficulty')}
        return {k:v for k,v in row.items() if k!='metadata'} | {'metadata':metadata}
    def score_answer(self, answer, entry): return score_answer(answer, entry)
    def extract_answer(self, text): return extract_final(text, self.rows[0]['task'])
