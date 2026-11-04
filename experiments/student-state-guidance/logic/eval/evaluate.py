#!/usr/bin/env python
"""Evaluate models on reasoning-gym tasks with pass@k support."""

import argparse
import inspect
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reasoning_gym.factory import DATASETS, create_dataset
from reasoning_gym.utils import SYSTEM_PROMPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_answer(text: str, tag: str = "answer") -> Optional[str]:
    """Extract the last <tag>...</tag> content from model output."""
    matches = list(re.finditer(rf"<{tag}>\s?(.*?)\s?</{tag}>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def estimate_pass_at_k(
    num_samples: Union[int, list[int], np.ndarray],
    num_correct: Union[list[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """Unbiased estimator of pass@k (Codex paper, Chen et al. 2021)."""

    def _estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [_estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def build_prompts(tokenizer, questions: list[str], system_prompt: str) -> list[str]:
    """Apply chat template to questions."""
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt += "Let's think step by step.\n"
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Task config loading
# ---------------------------------------------------------------------------

def load_tasks_from_training_config(config_path: str) -> tuple[list[tuple[str, dict]], Optional[str]]:
    """Parse training YAML and return ([(task_name, config_kwargs), ...], system_prompt_or_None)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tasks = []
    for task_name, task_cfg in cfg["reasoning_gym"]["datasets"].items():
        kwargs = task_cfg.get("config", {}) or {}
        tasks.append((task_name, kwargs))

    # Read developer_prompt setting and resolve to actual prompt text
    developer_prompt_key = cfg.get("reasoning_gym", {}).get("developer_prompt")
    system_prompt = None
    if developer_prompt_key and developer_prompt_key in SYSTEM_PROMPTS:
        system_prompt = SYSTEM_PROMPTS[developer_prompt_key]
    return tasks, system_prompt


def get_task_config_params(task_name: str) -> dict:
    """Get config parameters for a task (for dynamic CLI args)."""
    if task_name not in DATASETS:
        return {}
    _, config_cls = DATASETS[task_name]
    sig = inspect.signature(config_cls.__init__)
    params = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params[name] = {
            "type": param.annotation if param.annotation != inspect.Parameter.empty else str,
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "required": param.default == inspect.Parameter.empty,
        }
    return params


def add_task_cli_args(parser: argparse.ArgumentParser, task_name: str):
    """Dynamically add task-specific config fields to argparse."""
    for name, info in get_task_config_params(task_name).items():
        if name in ("seed", "size"):
            continue
        t = info["type"]
        arg_type = {int: int, float: float, bool: lambda x: x.lower() in ("true", "1", "yes")}.get(t, str)
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=arg_type,
            default=info["default"],
            required=info["required"],
            help=f"Task config: {name} (default: {info['default']})",
        )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _score_sample(idx, entry, resp, dataset, pass_threshold, tokenizer):
    """Build a per-prompt sample dict from a vLLM RequestOutput."""
    sample = {
        "idx": idx,
        "question": entry["question"],
        "expected_answer": entry["answer"],
        "scores": [],
        "passed": [],
        "preds": [],
        "responses": [],
        "finish_reasons": [],
        "response_token_lens": [],
    }
    for out in resp.outputs:
        text = out.text
        extracted = extract_answer(text)
        score = dataset.score_answer(extracted, entry)
        passed = score >= pass_threshold

        fr = getattr(out, "finish_reason", None)
        tok_ids = getattr(out, "token_ids", None)
        tok_len = len(tok_ids) if tok_ids is not None else len(tokenizer.encode(text, add_special_tokens=False))

        sample["scores"].append(float(score))
        sample["passed"].append(bool(passed))
        sample["preds"].append(extracted)
        sample["responses"].append(text)
        sample["finish_reasons"].append(fr)
        sample["response_token_lens"].append(tok_len)
    return sample


def _load_partial(partial_path: Path, expected_header: dict) -> dict[int, dict]:
    """Load completed samples keyed by idx; reset file if header mismatches."""
    if not partial_path.exists():
        return {}
    try:
        with open(partial_path) as f:
            first = f.readline()
            if not first.strip():
                return {}
            header = json.loads(first)
            if header.get("_partial_header") is not True:
                raise ValueError("missing partial header")
            stored = {k: header.get(k) for k in expected_header}
            if stored != expected_header:
                bak = partial_path.with_suffix(partial_path.suffix + ".bak")
                print(f"  WARNING: partial header mismatch; moving {partial_path} -> {bak}")
                print(f"    expected: {expected_header}")
                print(f"    stored:   {stored}")
                partial_path.rename(bak)
                return {}
            completed: dict[int, dict] = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    # truncated line from a killed write; drop the rest
                    print(f"  WARNING: truncated line in {partial_path}, ignoring tail")
                    break
                completed[int(s["idx"])] = s
        return completed
    except Exception as e:
        bak = partial_path.with_suffix(partial_path.suffix + ".bak")
        print(f"  WARNING: failed to parse {partial_path} ({e}); moving to {bak}")
        partial_path.rename(bak)
        return {}


def _append_samples(partial_path: Path, header: dict, samples: list[dict]):
    """Append samples to JSONL; write header if file is new. Flush+fsync."""
    new_file = not partial_path.exists()
    with open(partial_path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(json.dumps({"_partial_header": True, **header}, ensure_ascii=False) + "\n")
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def evaluate_task(
    llm,
    tokenizer,
    task_name: str,
    task_kwargs: dict,
    dataset_size: int,
    seed: int,
    n_sampling: int,
    sampling_params: SamplingParams,
    pass_threshold: float,
    system_prompt: str,
    partial_path: Optional[Path] = None,
    chunk_size: int = 64,
) -> tuple[dict, list]:
    """Evaluate a single task. Returns (metrics, per_sample_results).

    If ``partial_path`` is given, completed samples are appended to a JSONL
    checkpoint file so the run is resumable across job preemption.
    """

    # 1. Create dataset
    dataset = create_dataset(task_name, seed=seed, size=dataset_size, **task_kwargs)
    print(f"  Dataset '{task_name}' created: {len(dataset)} samples")

    # 2. Collect questions and entries
    questions = []
    entries = []
    for i in range(len(dataset)):
        entry = dataset[i]
        questions.append(entry["question"])
        entries.append(entry)

    # 3. Resume: load any previously-completed samples
    header = {
        "task_name": task_name,
        "task_kwargs": task_kwargs,
        "dataset_size": dataset_size,
        "seed": seed,
        "n_sampling": n_sampling,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "max_tokens": sampling_params.max_tokens,
        "pass_threshold": pass_threshold,
    }
    completed: dict[int, dict] = {}
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        completed = _load_partial(partial_path, header)
        if completed:
            print(f"  Resumed: {len(completed)}/{len(dataset)} prompts already done")

    todo_idx = [i for i in range(len(dataset)) if i not in completed]

    # 4. Generate remaining prompts in chunks, checkpointing after each chunk
    sp = SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k if sampling_params.top_k and sampling_params.top_k > 0 else -1,
        stop=sampling_params.stop,
        n=n_sampling,
    )

    if todo_idx:
        print(
            f"  Generating {n_sampling} completion(s) per prompt for "
            f"{len(todo_idx)} prompts in chunks of {chunk_size}..."
        )
        for chunk_start in range(0, len(todo_idx), chunk_size):
            chunk_ids = todo_idx[chunk_start : chunk_start + chunk_size]
            chunk_qs = [questions[i] for i in chunk_ids]
            chunk_entries = [entries[i] for i in chunk_ids]
            chunk_prompts = build_prompts(tokenizer, chunk_qs, system_prompt)
            t_chunk = time.time()
            responses = llm.generate(chunk_prompts, sp)
            new_samples = [
                _score_sample(gi, e, r, dataset, pass_threshold, tokenizer)
                for gi, e, r in zip(chunk_ids, chunk_entries, responses)
            ]
            for s in new_samples:
                completed[s["idx"]] = s
            if partial_path is not None:
                _append_samples(partial_path, header, new_samples)
            done_so_far = len(completed)
            print(
                f"  Chunk {chunk_start // chunk_size + 1}/"
                f"{(len(todo_idx) + chunk_size - 1) // chunk_size}: "
                f"+{len(new_samples)} prompts in {time.time() - t_chunk:.1f}s "
                f"(total {done_so_far}/{len(dataset)})"
            )
    else:
        print("  All prompts already completed; skipping generation.")

    # 5. Assemble final results in idx order and recompute aggregates
    results = [completed[i] for i in range(len(dataset))]

    total_tok = 0
    length_hits = 0
    finish_cnt = 0
    for r in results:
        total_tok += sum(r["response_token_lens"])
        for fr in r["finish_reasons"]:
            if fr is not None:
                finish_cnt += 1
                if isinstance(fr, str) and "length" in fr.lower():
                    length_hits += 1

    n_total = len(results)
    n_completions = sum(len(r["scores"]) for r in results)

    accuracy = sum(1 for r in results if any(r["passed"])) / n_total if n_total else 0
    all_scores = [s for r in results for s in r["scores"]]
    mean_score = float(np.mean(all_scores)) * 100 if all_scores else 0.0

    metrics = {
        "accuracy": accuracy,
        "pass_acc": round(accuracy * 100, 1),
        "mean_score": round(mean_score, 1),
        "num_samples": n_total,
        "n_sampling": n_sampling,
        "pass_threshold": pass_threshold,
        "avg_response_length_tokens": round(total_tok / n_completions, 2) if n_completions else 0,
        "max_length_ratio": round(100.0 * length_hits / finish_cnt, 2) if finish_cnt else 0,
    }

    # pass@k and best-of-k
    if n_sampling > 1:
        num_correct = [sum(r["passed"]) for r in results]
        k_values = [1]
        p = 1
        while 2**p <= n_sampling:
            k_values.append(2**p)
            p += 1
        if k_values[-1] < n_sampling:
            k_values.append(n_sampling)

        pass_at_k = {}
        for k in k_values:
            est = estimate_pass_at_k(n_sampling, num_correct, k)
            pass_at_k[k] = round(float(np.mean(est)) * 100, 1)
        metrics["pass@k"] = pass_at_k

        # best-of-k: for each question, take the max score across k random
        # samples (averaged over all C(n,k) subsets via Monte Carlo), then
        # average across questions.  This is the continuous-reward analog of
        # pass@k and is especially useful for tasks with dense rewards.
        best_of_k = {}
        max_scores = np.array([max(r["scores"]) for r in results])
        all_scores_per_q = [np.array(r["scores"]) for r in results]
        short_samples = [(i, len(s)) for i, s in enumerate(all_scores_per_q) if len(s) < n_sampling]
        if short_samples:
            print(f"  WARNING: {len(short_samples)} samples have fewer than {n_sampling} completions: {short_samples[:5]}")
        for k in k_values:
            if k >= n_sampling:
                # best-of-n is just the max
                bok = float(np.mean(max_scores)) * 100
            elif k == 1:
                # best-of-1 is just the mean score
                bok = float(np.mean([np.mean(s) for s in all_scores_per_q])) * 100
            else:
                # Monte Carlo estimate of E[max of k draws] per question
                n_mc = 1000
                rng = np.random.default_rng(seed)
                bok_per_q = []
                for scores in all_scores_per_q:
                    n_s = len(scores)
                    if n_s <= k:
                        bok_per_q.append(float(np.max(scores)))
                    else:
                        # draw k indices per MC trial (replace=False is per-row with integer input)
                        idx = rng.integers(0, n_s, size=(n_mc, k))
                        draws = scores[idx]
                        bok_per_q.append(float(np.mean(np.max(draws, axis=1))))
                bok = float(np.mean(bok_per_q)) * 100
            best_of_k[k] = round(bok, 1)
        metrics["best_of_k"] = best_of_k

    return metrics, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate models on reasoning-gym tasks")
    parser.add_argument("--model-path", type=str, required=True, help="Model path")
    parser.add_argument("--task-name", type=str, default=None, help="Task name (ignored if --training-config)")
    parser.add_argument("--training-config", type=str, default=None, help="Training YAML config path")
    parser.add_argument("--dataset-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sampling", type=int, default=1, help="Completions per question for pass@k")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--pass-threshold", type=float, default=1.0, help="Score threshold for pass@k")
    parser.add_argument("--output-dir", type=str, default="eval_result")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Prompts per generation chunk for checkpointing (default: 64)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing partial checkpoints and start fresh",
    )
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="Keep the .partial.jsonl checkpoint file after a task completes",
    )
    return parser


def main():
    parser = create_parser()

    # Two-pass parse for dynamic task args
    known_args, remaining = parser.parse_known_args()

    # Determine tasks
    config_system_prompt = None
    if known_args.training_config:
        tasks, config_system_prompt = load_tasks_from_training_config(known_args.training_config)
        args = parser.parse_args()  # no extra task args expected
    elif known_args.task_name:
        add_task_cli_args(parser, known_args.task_name)
        args = parser.parse_args()
        # Collect task-specific kwargs
        config_params = get_task_config_params(args.task_name)
        task_kwargs = {}
        for name in config_params:
            if name in ("seed", "size"):
                continue
            if hasattr(args, name) and getattr(args, name) is not None:
                task_kwargs[name] = getattr(args, name)
        tasks = [(args.task_name, task_kwargs)]
        config_system_prompt = None
    else:
        parser.error("Must provide either --task-name or --training-config")
        return

    args = known_args  # use known_args for the common fields

    # Load model
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Model loaded (TP={args.tensor_parallel_size})")

    # Priority: CLI --system-prompt > training config developer_prompt > DEFAULT_SYSTEM_PROMPT
    system_prompt = args.system_prompt or config_system_prompt
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"],
    )

    model_name = Path(args.model_path).name

    # Evaluate each task
    for task_name, task_kwargs in tasks:
        out_dir = Path(args.output_dir) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        detail_path = out_dir / f"{task_name}.json"
        metrics_path = out_dir / f"{task_name}_metrics.json"
        partial_path = out_dir / f"{task_name}.partial.jsonl"

        # Skip if already fully done (both final files present)
        if detail_path.exists() and metrics_path.exists():
            print(f"\n[SKIP] {task_name}: final outputs exist at {detail_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {task_name}")
        print(f"  Config: {task_kwargs}")
        print(f"  Samples: {args.dataset_size}, n_sampling: {args.n_sampling}")
        print(f"  Partial: {partial_path}")
        print(f"{'='*60}")

        if args.no_resume and partial_path.exists():
            bak = partial_path.with_suffix(partial_path.suffix + ".bak")
            print(f"  --no-resume: moving existing partial -> {bak}")
            partial_path.rename(bak)

        t0 = time.time()
        metrics, results = evaluate_task(
            llm=llm,
            tokenizer=tokenizer,
            task_name=task_name,
            task_kwargs=task_kwargs,
            dataset_size=args.dataset_size,
            seed=args.seed,
            n_sampling=args.n_sampling,
            sampling_params=sampling_params,
            pass_threshold=args.pass_threshold,
            system_prompt=system_prompt,
            partial_path=partial_path,
            chunk_size=args.chunk_size,
        )
        elapsed = time.time() - t0

        # Print results
        print(f"\n  Accuracy: {metrics['pass_acc']}%")
        print(f"  Mean score: {metrics['mean_score']}%")
        print(f"  Avg response tokens: {metrics['avg_response_length_tokens']}")
        print(f"  Max length ratio: {metrics['max_length_ratio']}%")
        if "pass@k" in metrics:
            for k, v in metrics["pass@k"].items():
                print(f"  Pass@{k}: {v}%")
        if "best_of_k" in metrics:
            for k, v in metrics["best_of_k"].items():
                print(f"  Best-of-{k}: {v}%")
        print(f"  Time: {elapsed:.1f}s")

        # Save final results
        output_data = {
            "model": args.model_path,
            "task_name": task_name,
            "task_config": task_kwargs,
            "dataset_size": args.dataset_size,
            "seed": args.seed,
            "n_sampling": args.n_sampling,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "pass_threshold": args.pass_threshold,
            "metrics": metrics,
            "evaluation_time": elapsed,
            "results": results,
        }
        with open(detail_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        metrics_data = {
            "model": args.model_path,
            "task_name": task_name,
            "task_config": task_kwargs,
            "metrics": metrics,
            "evaluation_time": elapsed,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        print(f"  Results: {detail_path}")
        print(f"  Metrics: {metrics_path}")

        # Final outputs persisted -> remove the partial checkpoint
        if not args.keep_partial and partial_path.exists():
            try:
                partial_path.unlink()
                print(f"  Removed partial checkpoint: {partial_path}")
            except OSError as e:
                print(f"  WARNING: could not remove {partial_path}: {e}")


if __name__ == "__main__":
    import multiprocessing
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    multiprocessing.set_start_method("spawn", force=True)
    main()
