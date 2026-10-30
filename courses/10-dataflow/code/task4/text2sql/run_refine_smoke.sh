#!/bin/bash
# DataFlow Task4 Text2SQL 复现脚本（在远端 ~/dataflow-task4/text2sql/ 下运行）
set -euo pipefail
cd "$(dirname "$0")"
source ../env.sh

python text2sql_refine_task4.py \
  --input subset_smoke.jsonl \
  --cache-dir ./cache_smoke \
  --prefix task4_refine_smoke \
  --num-variations 1 \
  --question-candidates 2 \
  --cot-sampling 1 \
  --exec-generations 5 \
  --max-workers 4 \
  --temperature 0.7
