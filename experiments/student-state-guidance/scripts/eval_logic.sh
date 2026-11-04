#!/usr/bin/env bash
# Evaluate one checkpoint on a reasoning-gym task, reporting pass@k.
#
#   MODEL_PATH=/path/to/hf_checkpoint TASK=kk training/opd/eval.sh
#
# The checkpoint must be in HuggingFace format. Convert a whole run with
#   python3 logic/convert_verl_ckpt_to_hf.py <run_checkpoint_dir>
#
# Optional: OUT_DIR, TP, N_SAMPLING, DATASET_SIZE, MAX_TOKENS, and for
# knights-and-knaves the difficulty knobs N_PEOPLE / DEPTH / WIDTH.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_PATH:?set MODEL_PATH to an HF-format checkpoint}"
TASK=${TASK:-kk}
OUT_DIR=${OUT_DIR:-$REPO/eval_result/$TASK}
TP=${TP:-4}
N_SAMPLING=${N_SAMPLING:-32}
DATASET_SIZE=${DATASET_SIZE:-256}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-0.6}
SEED=${SEED:-42}

# reasoning-gym task name for each training config
case "$TASK" in
    kk)          RG_NAME=knights_knaves ;;
    countdown)   RG_NAME=countdown ;;
    zebra)       RG_NAME=zebra_puzzles ;;
    mini_sudoku) RG_NAME=mini_sudoku ;;
    *) echo "unknown TASK: $TASK (expected kk | countdown | zebra | mini_sudoku)" >&2; exit 1 ;;
esac

# Must match developer_prompt: "default" in reasoning_gym/utils.py SYSTEM_PROMPTS,
# which is what the models were trained with.
read -r -d '' DEFAULT_PROMPT <<'PROMPT' || true
Given a problem, your task is to answer the question by thinking step-by-step in a clear and specific manner.
Once you have thought about the reasoning process, provide the answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.
PROMPT

TASK_ARGS=()
if [[ "$TASK" == "kk" ]]; then
    TASK_ARGS=(
        --n-people "${N_PEOPLE:-6}"
        --depth-constraint "${DEPTH:-2}"
        --width-constraint "${WIDTH:-2}"
    )
fi

cd "$REPO/logic/eval"
python3 evaluate.py \
    --model-path "$MODEL_PATH" \
    --task-name "$RG_NAME" \
    --dataset-size "$DATASET_SIZE" \
    --n-sampling "$N_SAMPLING" \
    --max-tokens "$MAX_TOKENS" \
    --tensor-parallel-size "$TP" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --system-prompt "$DEFAULT_PROMPT" \
    "${TASK_ARGS[@]}"
