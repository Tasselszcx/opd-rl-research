#!/usr/bin/env bash
# Train one method on a reasoning-gym task.
#
#   MODEL_DIR=/path/to/models scripts/train_logic.sh <method> [task]
#
# method: grpo | opd | opd_then_rl | sft_then_rl | kdrl | kdrl_anneal
#         kdrl_mask | hdpo | srpo | rlsd | trrd
# task:   countdown | kk | zebra | mini_sudoku   (default kk)
#
# REF_MODEL_PATH is the TEACHER, not a KL anchor.

set -euo pipefail
METHOD=${1:?usage: $0 <method> [task]}
TASK=${2:-${TASK:-kk}}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-$REPO/models}
MODEL_PATH=${MODEL_PATH:-$MODEL_DIR/Qwen3-1.7B-Base}
REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_DIR/Qwen3-8B}

N_GPUS=${N_GPUS:-4}
MICRO_BSZ=${MICRO_BSZ:-1}
LOGPROB_MICRO_BSZ=${LOGPROB_MICRO_BSZ:-1}
RESUME_MODE=${RESUME_MODE:-disable}
EXTRA_ARGS=()

# beta is an order of magnitude larger here than on the math side.
case "$METHOD" in
  grpo)         ADV=grpo ;;
  opd)          ADV=on_policy_distillation ;;
  opd_then_rl)  ADV=grpo
                : "${INIT_CKPT:?set INIT_CKPT to an OPD checkpoint, e.g. .../<opd-run>/global_step_60}"
                RESUME_MODE=$INIT_CKPT ;;
  sft_then_rl)  ADV=grpo
                : "${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint}" ;;
  kdrl)         ADV=grpo_opd;        EXTRA_ARGS=(algorithm.opd_beta=0.02) ;;
  kdrl_anneal)  ADV=grpo_opd;        EXTRA_ARGS=(algorithm.opd_beta_init=2e-1
                                                 algorithm.opd_beta_min=2e-3
                                                 algorithm.opd_beta_delta=1.6e-3) ;;
  kdrl_mask)    ADV=grpo_opd_mask;   EXTRA_ARGS=(algorithm.opd_beta=0.02) ;;
  hdpo)         ADV=grpo_opd_cliff;  EXTRA_ARGS=(algorithm.opd_beta=0.02) ;;
  srpo)         ADV=grpo_srpo;       EXTRA_ARGS=(algorithm.opd_beta=1) ;;
  rlsd)         ADV=grpo_rlsd;       EXTRA_ARGS=(algorithm.rlsd_eps_w=0.2 algorithm.rlsd_lambda=1.0) ;;
  trrd)         ADV=grpo;            EXTRA_ARGS=(actor_rollout_ref.actor.trrd_enabled=True
                                                 actor_rollout_ref.actor.trrd_alpha=0.5) ;;
  *) echo "unknown method: $METHOD" >&2; exit 1 ;;
esac

RUN_NAME=${RUN_NAME:-$(basename "$MODEL_PATH")-$TASK-$METHOD}
export VLLM_ATTENTION_BACKEND=XFORMERS
cd "$REPO/logic/training"

python3 -u train_grpo.py \
    --config-path "$REPO/logic/training/configs/gkd_configs" \
    --config-name "$TASK" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-4096} \
    algorithm.adv_estimator=$ADV \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.ref.path="$REF_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BSZ \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BSZ \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name=opd-rlvr-logic \
    trainer.experiment_name="$RUN_NAME" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.total_epochs=2 \
    trainer.save_freq=${SAVE_FREQ:-5} \
    trainer.test_freq=5 \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
