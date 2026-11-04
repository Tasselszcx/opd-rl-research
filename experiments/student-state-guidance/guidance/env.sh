#!/usr/bin/env bash
# Historical G1 helper. G3a uses state_guidance/env.sh directly.
set -euo pipefail
PILOT_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec /bin/bash "$PILOT_REPO/state_guidance/env.sh" "$@"
