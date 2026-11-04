#!/usr/bin/env bash
# Historical G2 helper. Use its own explicit output directory.
set -euo pipefail
AUDIT_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${REPRO_ROOT:?Set a separate historical-audit run directory explicitly}"
exec /bin/bash "$AUDIT_REPO/guidance/env.sh" "$@"
