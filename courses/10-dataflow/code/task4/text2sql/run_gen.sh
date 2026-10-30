#!/bin/bash
# DataFlow Task4 Text2SQL 数据合成(gen) 运行脚本（refine 成功后跑，可选）
# 同 refine：DeepSeek-V4-Flash + 2 并发 + 8 次指数退避重试，3 个库各生成 1 条 SQL。
set -euo pipefail
cd "$(dirname "$0")"
source ../env.sh
export DF_MODEL="${DF_MODEL_OVERRIDE:-DeepSeek-V4-Flash}"

python text2sql_gen_task4.py \
  --cache-dir ./cache_gen \
  --prefix task4_gen \
  --generate-num 1 \
  --question-candidates 2 \
  --cot-sampling 1 \
  --exec-generations 3 \
  --max-workers 2 \
  --max-retries 8 \
  --temperature 0.7
