#!/usr/bin/env bash
set -euo pipefail
G3_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${REPRO_ROOT:?Set REPRO_ROOT to an absolute, persistent run directory}"
: "${G3_MODEL_ROOT:?Set G3_MODEL_ROOT to the verified model-cache directory}"
[[ "$REPRO_ROOT" = /* && "$G3_MODEL_ROOT" = /* ]] || {
  echo "REPRO_ROOT and G3_MODEL_ROOT must be absolute paths" >&2; exit 2;
}
# A restored runtime is optional. Otherwise use the caller's activated environment.
if [[ -n ${MIGRATION_ROOT:-} ]]; then
  [[ -x "$MIGRATION_ROOT/runtime/venv/bin/python" ]] || {
    echo "MIGRATION_ROOT does not contain runtime/venv/bin/python" >&2; exit 2;
  }
  export PATH="$MIGRATION_ROOT/runtime/venv/bin:$PATH"
fi
# Native driver matches the current deployment; compatibility mode is opt-in.
if [[ ${G3_USE_CUDA_COMPAT:-0} == 1 ]]; then
  : "${MIGRATION_ROOT:?CUDA compatibility mode requires MIGRATION_ROOT}"
  G3_COMPAT="$MIGRATION_ROOT/runtime/cuda-compat128/usr/local/cuda-12.8/compat"
  [[ -d "$G3_COMPAT" ]] || { echo "CUDA compatibility libraries are absent" >&2; exit 2; }
  export LD_LIBRARY_PATH="$G3_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export REPRO_ROOT G3_MODEL_ROOT
export PYTHONPATH="$G3_REPO/verl:$G3_REPO/logic:$G3_REPO/logic/training:$G3_REPO"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=1073741824
export MPLBACKEND=Agg WANDB_MODE=offline
unset RAY_ADDRESS TRITON_PTXAS_PATH RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
cd "$G3_REPO"
exec "$@"
