#!/usr/bin/env bash
set -euo pipefail
G3_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$G3_REPO/verl:$G3_REPO/logic:$G3_REPO/logic/training:$G3_REPO"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
cd "$G3_REPO"
exec "${G3_PYTHON:-python}" -m unittest discover -s state_guidance -p 'test_*.py' -v
