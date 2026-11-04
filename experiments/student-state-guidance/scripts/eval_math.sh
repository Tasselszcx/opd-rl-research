#!/usr/bin/env bash
# Evaluate one checkpoint on the math benchmarks reported in the paper.
#
#   MODEL_PATH=/path/to/hf_checkpoint scripts/opd/eval.sh
#
# The checkpoint must be in HuggingFace format. verl writes sharded FSDP
# checkpoints, so convert first:
#   python3 verl/scripts/model_merger.py --local_dir <ckpt>/actor --target_dir <hf_dir>
#
# Optional: OUT_DIR, TP (tensor parallel size), N (samples per problem),
# DATASETS (subset of the tags below).

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_PATH:?set MODEL_PATH to an HF-format checkpoint}"
OUT_DIR=${OUT_DIR:-$REPO/evaluation/$(basename "$MODEL_PATH")}
TP=${TP:-2}
N=${N:-32}

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# tag | data_id | split | max_model_len
ALL_DATASETS=(
    "math500|zwhe99/MATH|math500|8192"
    "aime24|zwhe99/aime90|2024|16384"
    "aime25|math-ai/aime25|test|16384"
    "amc23|zwhe99/amc23|test|16384"
    "minerva|zwhe99/simplerl-minerva-math|test|16384"
)
SELECTED=${DATASETS:-}

for ENTRY in "${ALL_DATASETS[@]}"; do
    IFS='|' read -r DS_TAG DS_ID DS_SPLIT MAX_LEN <<< "$ENTRY"
    if [[ -n "$SELECTED" && " $SELECTED " != *" $DS_TAG "* ]]; then
        continue
    fi
    echo "=== $(basename "$MODEL_PATH") / $DS_TAG  (n=$N, max_len=$MAX_LEN, tp=$TP) ==="
    # Sampling follows Qwen3's non-thinking recommendation; enable_thinking=False
    # matches the training setting. passk_rule=all reports every k <= n, which is
    # what the paper's pass@k tables use.
    python3 "$REPO/math/uni_eval.py" \
        --base_model "$MODEL_PATH" \
        --chat_template_name default \
        --system_prompt_name simplerl \
        --output_dir "$OUT_DIR/$DS_TAG" \
        --bf16 True \
        --tensor_parallel_size "$TP" \
        --data_id "$DS_ID" \
        --split "$DS_SPLIT" \
        --max_model_len "$MAX_LEN" \
        --temperature 0.7 \
        --top_p 0.8 \
        --top_k 20 \
        --n "$N" \
        --enable_thinking False \
        --passk_rule all \
        || echo "[FAILED] $DS_TAG"
done
