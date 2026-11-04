> 这是历史论文脚本说明。当前K&K八组使用[当前代码README](../README.md)和`run_g3.sh`；下文的EOS屏蔽描述只对应legacy模式，修复版对齐结束符并保留监督。

# Running the experiments

```bash
MODEL_DIR=/path/to/models scripts/train_math.sh  <method>
MODEL_DIR=/path/to/models scripts/train_logic.sh <method> [task]
```

`method` is one of `grpo`, `opd`, `opd_then_rl`, `sft_then_rl`, `kdrl`,
`kdrl_anneal`, `kdrl_mask`, `hdpo`, `srpo`, `rlsd`, `trrd`; `task` is one of
`countdown`, `kk`, `zebra`, `mini_sudoku` and defaults to `kk`. Everything that
distinguishes one method from another is the `case` block near the top of each
script.

`opd_then_rl` starts from an OPD checkpoint, so run `opd` first and pass
`INIT_CKPT`; `sft_then_rl` needs `MODEL_PATH` pointing at an SFT checkpoint.

Evaluation takes an HF-format checkpoint, so convert first:

```bash
# math
python3 verl/scripts/model_merger.py --local_dir <ckpt>/actor --target_dir <hf_dir>
MODEL_PATH=<hf_dir> scripts/eval_math.sh

# logic puzzles
python3 logic/convert_verl_ckpt_to_hf.py <run_checkpoint_dir>
MODEL_PATH=<hf_dir> TASK=kk N_SAMPLING=128 scripts/eval_logic.sh
```

## Settings that depart from upstream

These are what the scripts and task configs set explicitly. Each will change your
results if you drop it.

- **`actor_rollout_ref.ref.path` is the teacher.** In upstream verl the reference
  policy is a KL anchor that defaults to the actor's own weights. Here it is a
  separate, larger frozen model whose log-probs are the distillation signal.
  `use_kl_loss=True` with `kl_loss_coef=0.0` exists only so verl spawns the ref
  worker; no KL penalty is applied.
- **The EOS token is excluded from the teacher term**, everywhere, and TRRD pins the
  teacher to the old policy at that position. The student is a base model and the
  teacher an instruct model, so their end-of-sequence tokens are different tokens
  and their log-probs there are not comparable.
- **Math runs use Qwen3 non-thinking mode** (`data.enable_thinking=False`) and
  therefore switch off the `</think>` format reward
  (`DEEPMATH_REQUIRE_THINK_TAG=0`) -- otherwise every response scores -1, since no
  such tag is produced.
- **Logic prompts carry a chain-of-thought trigger** (`data.cot_suffix`). Without
  it the teacher tends to emit the answer directly rather than reasoning to it.
- **Knights-and-knaves is scored 1.0 / 0.0.** Upstream reasoning-gym gives partial
  credit for a partly-correct assignment, which would change both the correct /
  incorrect split and GRPO's group normalisation.
- **`beta` differs between domains**: 0.002 for math, 0.02 for logic puzzles.
- **The dataloader is seeded** (`data.seed`). The original logic-puzzle runs were
  not, so their exact data order is not recoverable.
- **Evaluation** reports every `k <= n` for pass@k (`--passk_rule all`); upstream
  DeepMath reports only `k` with `2k <= n`.
