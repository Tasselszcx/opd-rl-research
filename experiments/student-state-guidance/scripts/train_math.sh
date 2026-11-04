#!/usr/bin/env bash
# Train one method on DeepMath-103K.
#
#   MODEL_DIR=/path/to/models scripts/train_math.sh <method>
#
# method: grpo | opd | opd_then_rl | sft_then_rl | kdrl | kdrl_anneal
#         kdrl_mask | hdpo | srpo | rlsd | trrd
#
# MODEL_PATH / REF_MODEL_PATH pick the student and the teacher. REF_MODEL_PATH is
# the TEACHER, not a KL anchor: KL is wired up only so verl spawns the ref worker,
# and its coefficient is zero.

set -euo pipefail
METHOD=${1:?usage: $0 <method>}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-$REPO/models}
DATA_DIR=${DATA_DIR:-$REPO/data}
MODEL_PATH=${MODEL_PATH:-$MODEL_DIR/Qwen3-1.7B-Base}
REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_DIR/Qwen3-8B}

N_GPUS=${N_GPUS:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MICRO_BSZ=${MICRO_BSZ:-2}
LOGPROB_MICRO_BSZ=${LOGPROB_MICRO_BSZ:-4}
RESUME_MODE=${RESUME_MODE:-auto}
EXTRA_ARGS=()

# Everything that distinguishes one method from another.
case "$METHOD" in
  grpo)         ADV=grpo ;;
  opd)          ADV=on_policy_distillation
                PPO_MINI_BATCH_SIZE=32; MICRO_BSZ=1; LOGPROB_MICRO_BSZ=2; N_GPUS=${N_GPUS_OVERRIDE:-4} ;;
  opd_then_rl)  ADV=grpo
                : "${INIT_CKPT:?set INIT_CKPT to an OPD checkpoint, e.g. \$MODEL_DIR/<opd-run>/global_step_60}"
                RESUME_MODE=$INIT_CKPT; MICRO_BSZ=4; LOGPROB_MICRO_BSZ=8; N_GPUS=${N_GPUS_OVERRIDE:-4} ;;
  sft_then_rl)  ADV=grpo
                : "${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint}" ;;
  kdrl)         ADV=grpo_opd;        EXTRA_ARGS=(algorithm.opd_beta=0.002) ;;
  kdrl_anneal)  ADV=grpo_opd;        EXTRA_ARGS=(algorithm.opd_beta_init=2e-1
                                                 algorithm.opd_beta_min=2e-3
                                                 algorithm.opd_beta_delta=1.6e-3) ;;
  kdrl_mask)    ADV=grpo_opd_mask;   EXTRA_ARGS=(algorithm.opd_beta=0.002) ;;
  hdpo)         ADV=grpo_opd_cliff;  EXTRA_ARGS=(algorithm.opd_beta=0.01) ;;
  srpo)         ADV=grpo_srpo;       EXTRA_ARGS=(algorithm.opd_beta=1) ;;
  rlsd)         ADV=grpo_rlsd;       EXTRA_ARGS=(algorithm.rlsd_eps_w=0.2 algorithm.rlsd_lambda=1.0) ;;
  trrd)         ADV=grpo;            EXTRA_ARGS=(actor_rollout_ref.actor.trrd_enabled=true
                                                 actor_rollout_ref.actor.trrd_alpha=0.5) ;;
  *) echo "unknown method: $METHOD" >&2; exit 1 ;;
esac

RUN_NAME=${RUN_NAME:-$(basename "$MODEL_PATH")-$(basename "$REF_MODEL_PATH")-$METHOD}
mkdir -p "$MODEL_DIR/$RUN_NAME"

export VLLM_ATTENTION_BACKEND=XFORMERS
# Non-thinking mode produces no </think> tag, so the format reward is switched off
# too; unset for the upstream DeepMath behaviour.
export DEEPMATH_REQUIRE_THINK_TAG=0

python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.rm_system_prompt=False \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-8192} \
    data.enable_thinking=False \
    algorithm.adv_estimator=$ADV \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.ref.path="$REF_MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BSZ \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BSZ \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO_BSZ \
    custom_reward_function.path="$REPO/math/utils/reward_utils/reward_func.py" \
    custom_reward_function.name=reward_func \
    custom_reward_function.overlong_buffer.enable=False \
    trainer.critic_warmup=0 \
    trainer.project_name=opd-rlvr-math \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$MODEL_DIR/$RUN_NAME" \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.save_rollout=True \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.test_freq=2 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=${TOTAL_STEPS:-1800} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee -a "$MODEL_DIR/$RUN_NAME/train.log"
