[![Venue](https://img.shields.io/badge/EMNLP%202026-Findings-2596be.svg)](https://2026.emnlp.org/)

# Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

Implementations of the methods that combine reinforcement learning with verifiable
rewards (RLVR) and on-policy distillation (OPD), evaluated on math (DeepMath-103K)
and logic reasoning (reasoning-gym): GRPO, OPD, KDRL, KDRL-mask, SRPO, HDPO, TRRD,
RLSD, SFT-then-RL, and OPD-then-RL.

## Methods

Every method here is policy-gradient ascent under the same PPO-clipped surrogate,
`min(ρ·A, clip(ρ, 1±ε)·A)`, and differs only in the importance ratio `ρ` and the
per-token advantage `A`. Writing `Â` for the outcome-reward advantage, `d_t` for the
distillation advantage, `c = 1[R=1]` for a correct rollout and `C(x)` for an
all-wrong group:

| Method | Paradigm | `ρ` | `A_t` |
|---|---|---|---|
| GRPO | — | `r` | `Â` |
| OPD | — | `r` | `d_t` |
| KDRL | additive | `r` | `Â + β d_t` |
| KDRL-mask | additive | `r` | `Â + (1−c) β d_t` |
| SRPO | additive | `r` | `c Â + (1−c) d_t` |
| HDPO | additive | `r` | `Â + C(x) β d_t` |
| TRRD | modulate | `r^α · e^{−(1−α) d_t}` | `Â` |
| RLSD | modulate | `r` | `((1−λ) + λ·clip(e^{sign(Â)d_t}, 1±ε)) Â` |
| OPD-then-RL | sequential | `r` | `d_t`, then `Â` |

KDRL also comes in an annealed variant (`kdrl_anneal`), where `β` decays linearly
with the step. SFT-then-RL is the same two-stage shape as OPD-then-RL with a
supervised first stage.

The estimators live in `verl/verl/trainer/ppo/ray_trainer.py::compute_advantage`,
one `elif` branch each; TRRD instead modifies `compute_policy_loss`.
[`methods/estimators.py`](methods/estimators.py) restates each one without the
surrounding boilerplate, and [`methods/test_estimators.py`](methods/test_estimators.py)
asserts the two agree element-wise.

## Setup

verl, reasoning-gym and DeepMath's evaluation code are vendored, so there is
nothing to fetch separately. The two domains want different vLLM and transformers
versions, so they get separate environments.

```bash
git clone https://github.com/StringNLPLAB/opd-rlvr.git && cd opd-rlvr

# math
python -m venv .venv-math && . .venv-math/bin/activate
pip install -e verl
python3 verl/examples/data_preprocess/deepmath_103k.py   # -> data/{train,test}.parquet

# logic reasoning
python -m venv .venv-logic && . .venv-logic/bin/activate
pip install -e verl && pip install -e logic
```

Then `wandb login`, and point `MODEL_DIR` at a directory holding the student and
teacher weights. The paper uses `Qwen3-1.7B-Base` as the student and `Qwen3-8B` as
the teacher.

## Training and evaluation

```bash
MODEL_DIR=/path/to/models scripts/train_math.sh  srpo
MODEL_DIR=/path/to/models scripts/train_logic.sh srpo kk     # task: countdown | kk | zebra | mini_sudoku

# OPD-then-RL starts from an OPD checkpoint
MODEL_DIR=/path/to/models scripts/train_math.sh opd
MODEL_DIR=/path/to/models INIT_CKPT=models/<opd-run>/global_step_60 \
    scripts/train_math.sh opd_then_rl
```

Checkpoints are written in verl's sharded FSDP format and have to be converted
before evaluation:

```bash
# math: MATH500, AIME24, AIME25, AMC23, MinervaMath
python3 verl/scripts/model_merger.py --local_dir <ckpt>/actor --target_dir <hf_dir>
MODEL_PATH=<hf_dir> scripts/eval_math.sh

# logic reasoning: pass@k on a reasoning-gym task
python3 logic/convert_verl_ckpt_to_hf.py <run_checkpoint_dir>
MODEL_PATH=<hf_dir> TASK=kk N_SAMPLING=128 scripts/eval_logic.sh
```

[`scripts/README.md`](scripts/README.md) lists the settings that depart from
upstream and are needed to reproduce the reported numbers.

## Layout

```
verl/       verl, vendored, with the estimators                (pip install -e verl)
math/       DeepMath's evaluation code, vendored
logic/      reasoning-gym, vendored, with the training harness  (pip install -e logic)
scripts/    training and evaluation entry points
methods/    the reference implementation and its equivalence check
NOTICE      what came from where, and every file this project changed
```

Each vendored tree was committed unmodified first, so `git log -p <path>` shows
exactly what changed in any upstream file.

Tests are CPU-only: `pytest methods/test_estimators.py` and `pytest logic/tests/`.
The analysis and figure-generation code is not part of this release, so the paper's
plots cannot be regenerated here; training and evaluation reproduce in full.

## Acknowledgements

Built on [verl](https://github.com/volcengine/verl),
[reasoning-gym](https://github.com/open-thought/reasoning-gym) and
[DeepMath](https://github.com/zwhe99/DeepMath). [`NOTICE`](NOTICE) records what was
taken from each and every file this project modified.

Released under Apache-2.0; vendored code keeps its own license.
