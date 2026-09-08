#!/usr/bin/env bash
# chapter15 vector_add 非交互优化（在 AMD + ROCm 机器上跑）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f activate-rocm.sh ]]; then
  # shellcheck disable=SC1091
  source ./activate-rocm.sh || true
fi
exec uv run python -m kernel_optimize --batch chapter15/fixtures/vector_add --max-steps "${MAX_STEPS:-25}"

