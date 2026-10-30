#!/bin/bash
# DataFlow Task4 — AgenticRAG pipeline 一键复现（vm-60 实测）
# 该端点对 OneClick 用户限流 20 requests/minute，故默认 rpm=12、max_workers=2 匀速放行。
# 用法:
#   T4_ROWS=4 bash run_agentic_rag.sh main
#   T4_ROWS=10 bash run_agentic_rag.sh full
set -u
PROFILE="${1:-${T4_PROFILE:-main}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_SH="${T4_ENV_SH:-/home/simin/dataflow-task4/env.sh}"
source "$ENV_SH"

PKG=/home/simin/.venv-df3/lib/python3.10/site-packages/dataflow
SRC="$PKG/example/AgenticRAGPipeline/eval_test_data.jsonl"

export T4_PREFIX="agentic_rag_task4"
export T4_CACHE="$HERE/cache_${PROFILE}"
export T4_RPM="${T4_RPM:-12}"
export T4_MAX_WORKERS="${T4_MAX_WORKERS:-2}"
export T4_MAX_RETRIES="${T4_MAX_RETRIES:-6}"
export T4_MAX_PER_TASK="${T4_MAX_PER_TASK:-2}"
export T4_MAX_QUESTION="${T4_MAX_QUESTION:-1}"

# 用前 T4_ROWS 条文档做输入子集（保证可复现：不随机抽样）
ROWS="${T4_ROWS:-4}"
export T4_INPUT="$HERE/input_${PROFILE}.jsonl"
python - "$SRC" "$T4_INPUT" "$ROWS" <<'PY'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()][:n]
with open(dst, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("input subset:", dst, len(rows), "rows")
PY

mkdir -p "$T4_CACHE"
echo "=== PROFILE=$PROFILE ROWS=$ROWS RPM=$T4_RPM WORKERS=$T4_MAX_WORKERS CACHE=$T4_CACHE ==="
date -Is

python "$HERE/agentic_rag_task4.py"
RC=$?
echo "=== pipeline exit=$RC ==="
date -Is
if [ $RC -ne 0 ]; then exit $RC; fi

python "$HERE/verify_task4_agentic_rag.py" \
  "$T4_CACHE/agentic_rag_task4_step1.jsonl" \
  "$T4_CACHE/agentic_rag_task4_step2.jsonl" \
  "$HERE/field_check_${PROFILE}.json" \
  "$T4_INPUT"
