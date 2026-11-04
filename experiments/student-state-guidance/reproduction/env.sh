#!/usr/bin/env bash
# Historical reproduction helper; activate a compatible Python environment first.
set -euo pipefail
REPRO_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${REPRO_ROOT:?Set an absolute run directory explicitly}"
export PYTHONPATH="$REPRO_REPO/verl:$REPRO_REPO/logic:$REPRO_REPO/logic/training:$REPRO_REPO"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
cd "$REPRO_REPO"
exec "$@"
