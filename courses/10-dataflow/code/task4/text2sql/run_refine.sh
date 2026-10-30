#!/bin/bash
# DataFlow Task4 Text2SQL 数据优化(refine) 正式运行脚本
# 限流实测（2026-10-29，AMD Radeon OneClick 端点）：
#   1) 该账户全局限流 20 requests/min（所有模型/pipeline 共享），DeepSeek-V4-Flash
#      在 22:00-22:05 出现 token_rate_limit_exceeded 429 风暴；
#   2) 切 GLM-5.3-Flash 后遇到 "concurrency limit (32)" 的 429（其他并发任务占满）；
#   3) 22:36 复测 DeepSeek-V4-Flash 恢复 200，故正式运行回到 DeepSeek-V4-Flash。
# 对策：max_workers=2 + max_retries=8 的指数退避吸收 429；样本 4 条、采样数 1~3。
set -euo pipefail
cd "$(dirname "$0")"
source ../env.sh
export DF_MODEL="${DF_MODEL_OVERRIDE:-DeepSeek-V4-Flash}"

python text2sql_refine_task4.py \
  --input subset_refine.jsonl \
  --cache-dir ./cache_refine \
  --prefix task4_refine \
  --num-variations 1 \
  --question-candidates 2 \
  --cot-sampling 1 \
  --exec-generations 3 \
  --max-workers 2 \
  --max-retries 8 \
  --temperature 0.7
