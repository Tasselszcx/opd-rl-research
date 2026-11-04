"""One shard of the pre-registered guidance-pilot test; no teacher inputs."""
from pathlib import Path
import argparse
import json
import os
import time

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from vllm import LLM, SamplingParams

ROOT = Path(os.environ['REPRO_ROOT'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rank', required=True, type=int)
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--suites', nargs='+', default=['pilot_test'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f'rank_{args.rank}.jsonl'
    if output.exists():
        raise RuntimeError('Refusing to overwrite evaluation: ' + str(output))
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype='bfloat16', seed=42,
              max_model_len=6144, max_num_seqs=64, max_num_batched_tokens=16384,
              gpu_memory_utilization=0.6, enforce_eager=True, trust_remote_code=False)
    tokenizer = llm.get_tokenizer()
    scorer = reasoning_gym.create_dataset('knights_knaves', seed=42, size=1,
                 n_people=6, depth_constraint=2, width_constraint=2)
    started = time.time()
    rows_written = 0
    with output.open('w', buffering=1) as handle:
        for suite in args.suites:
            entries = [json.loads(s) for s in (ROOT / 'data' / f'{suite}.jsonl').read_text().splitlines()][:args.count]
            entries = [x for x in entries if x['index'] % args.world_size == args.rank]
            prompts = []
            for entry in entries:
                messages = [{'role': 'system', 'content': SYSTEM_PROMPTS['default']},
                            {'role': 'user', 'content': entry['question']}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                prompts.append(prompt + "Let's think step by step.\n")
            for sample in range(args.samples):
                params = SamplingParams(n=1, seed=42 + sample, temperature=0.7, top_p=0.8, top_k=20,
                            max_tokens=2048, stop=['</s>', '<|im_end|>', '<|endoftext|>'])
                predictions = llm.generate(prompts, params, use_tqdm=False)
                for entry, prediction in zip(entries, predictions):
                    response = prediction.outputs[0]
                    answer = extract_answer(response.text, tag_name='answer') or ''
                    score = scorer.score_answer(answer, entry)
                    assert score in (0.0, 1.0)
                    row = {'suite': suite, 'index': entry['index'], 'question_sha256': entry['question_sha256'],
                           'sample': sample, 'sampling_seed': 42 + sample, 'score': score,
                           'expected_answer': entry['answer'], 'predicted_answer': answer,
                           'response': response.text, 'response_tokens': len(response.token_ids),
                           'finish_reason': response.finish_reason}
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n'); rows_written += 1
                print(json.dumps({'suite': suite, 'sample_completed': sample, 'rank': args.rank,
                                  'rows': rows_written}), flush=True)
    record = {'status': 'completed', 'model': args.model, 'rank': args.rank, 'rows': rows_written,
              'suites': args.suites, 'samples': args.samples, 'seconds': time.time() - started,
              'device': os.environ.get('CUDA_VISIBLE_DEVICES')}
    (args.output / f'rank_{args.rank}.done.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
