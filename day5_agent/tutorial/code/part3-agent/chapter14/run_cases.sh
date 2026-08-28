#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${KERNEL_AGENT_PYTHON:-$PART_ROOT/.venv/bin/python}"
TASK="$PART_ROOT/chapter13/task.json"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

"$PYTHON" "$SCRIPT_DIR/calibrate.py" \
  --task "$TASK" \
  --candidate "$SCRIPT_DIR/baseline.py" \
  --runs 5 \
  > "$LOG_DIR/baseline-calibration.json"

"$PYTHON" "$SCRIPT_DIR/evaluate.py" \
  --task "$TASK" \
  --candidate "$SCRIPT_DIR/fixtures/compile_error.py" \
  --pretty \
  > "$LOG_DIR/compile-error.json"

"$PYTHON" "$SCRIPT_DIR/evaluate.py" \
  --task "$TASK" \
  --candidate "$SCRIPT_DIR/fixtures/wrong_answer.py" \
  --pretty \
  > "$LOG_DIR/wrong-answer.json"

"$PYTHON" "$SCRIPT_DIR/compare.py" \
  --task "$TASK" \
  --candidate "$SCRIPT_DIR/fixtures/fused_reference.py" \
  --incumbent "$SCRIPT_DIR/baseline.py" \
  --runs 5 \
  > "$LOG_DIR/fused-reference-comparison.json"

printf 'Evaluator evidence written to %s\n' "$LOG_DIR"

