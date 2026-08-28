#!/usr/bin/env bash
# 一键：跑 part3-agent 验收 + 打印结果 + 可视化优化过程
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f activate-rocm.sh ]]; then
  # shellcheck disable=SC1091
  source ./activate-rocm.sh || true
fi
MAX_STEPS="${MAX_STEPS:-12}"
exec uv run python chapter16/run_part3_test.py --max-steps "$MAX_STEPS" "$@"

