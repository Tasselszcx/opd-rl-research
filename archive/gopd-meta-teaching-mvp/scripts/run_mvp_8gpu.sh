#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export PYTHONPATH="$task_root/verl:$task_root${PYTHONPATH:+:$PYTHONPATH}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the local student model directory}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to the local teacher model directory}"
task_config="${1:-mvp_configs/smoke_8gpu.json}"
if [[ $# -gt 0 ]]; then shift; fi
exec python -m meta_teaching run --config "$task_config" "$@"
