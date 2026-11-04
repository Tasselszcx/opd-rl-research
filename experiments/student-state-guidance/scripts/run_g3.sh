#!/usr/bin/env bash
set -euo pipefail
G3_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "${1:-}" in
  train) G3_MODULE=state_guidance.coordinator ;;
  finalize) G3_MODULE=state_guidance.finalize ;;
  *) echo "Usage: bash scripts/run_g3.sh train|finalize [explicit recovery options]" >&2; exit 2 ;;
esac
shift
: "${REPRO_ROOT:?Set REPRO_ROOT first}"
: "${G3_MODEL_ROOT:?Set G3_MODEL_ROOT first}"
/bin/bash "$G3_REPO/state_guidance/env.sh" python -m state_guidance.public_setup check \
  --root "$REPRO_ROOT" --models "$G3_MODEL_ROOT"
exec /bin/bash "$G3_REPO/state_guidance/env.sh" python -m "$G3_MODULE" "$@"
