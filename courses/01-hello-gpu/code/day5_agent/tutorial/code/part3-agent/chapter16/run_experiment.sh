#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODE="${KERNEL_AGENT_NODE:-node}"
PYTHON="${KERNEL_AGENT_PYTHON:-$PART_ROOT/.venv/bin/python}"
APPROVAL="${KERNEL_AGENT_APPROVAL:-ask}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${KERNEL_AGENT_RUN_DIR:-$SCRIPT_DIR/logs/run-$TIMESTAMP}"
ENV_FILE="${KERNEL_AGENT_ENV_FILE:-$HOME/.config/hello-gpu/kernel-agent.env}"
WORKSPACE="${KERNEL_AGENT_WORKSPACE:-}"

if [[ -z "$WORKSPACE" ]]; then
  echo "KERNEL_AGENT_WORKSPACE must point to a frozen init workspace" >&2
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "model configuration not found: $ENV_FILE" >&2
  exit 2
fi

export KERNEL_AGENT_PYTHON="$PYTHON"

"$NODE" \
  --env-file="$ENV_FILE" \
  "$PART_ROOT/chapter15/cli.ts" \
  optimize \
  --workspace "$WORKSPACE" \
  --run-dir "$RUN_DIR" \
  --approval "$APPROVAL"

RETEST_STATUS=0
"$PYTHON" "$PART_ROOT/chapter14/compare.py" \
  --task "$RUN_DIR/task.json" \
  --candidate "$RUN_DIR/best.py" \
  --incumbent "$RUN_DIR/iteration-00-baseline/candidate.py" \
  --runs 5 \
  > "$RUN_DIR/independent-retest.json" || RETEST_STATUS=$?

"$NODE" "$SCRIPT_DIR/report.ts" "$RUN_DIR"
exit "$RETEST_STATUS"

