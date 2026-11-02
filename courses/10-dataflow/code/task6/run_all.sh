
#!/bin/bash
# DataFlow Task6 — 从数据到效果：评估模型能力（模型性能评测 Pipeline，vm-60 实测）
# 官方原型：gpu_pipelines/benchmark_eval/unified_bench_eval_type{1..6}.py（+ semantic 变体）
# 十个 run：
#   本地 Qwen2.5-0.5B-Instruct（transformers serving，全离线）：6 个 eval_type 全量
#   API DeepSeek-V4-Flash（AMD 端点，限流 10rpm）：key2_qa / key2_qa+semantic / key2_q_ma / key3_q_choices_as
# 用法: bash run_all.sh            # 全量
#       bash run_all.sh local      # 只跑本地 6 类型
#       bash run_all.sh api        # 只跑 API 4 run
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_SH="${T6_ENV_SH:-$HERE/env.sh}"
source "$ENV_SH"
cd "$HERE"

WHICH="${1:-all}"
OUT="${T6_EVIDENCE:-$HERE/../evidence/task6}"
mkdir -p "$OUT"
DATA="$HERE/data"

echo "=== DataFlow Task6 :: $WHICH ==="
date -Is

# 0) 评测数据集（官方 example 原样 + sha256 落盘）
python "$HERE/build_eval_data.py" 2>&1 | tee "$OUT/build_data.log"

run_bench () {  # name eval_type model extra_flags...
  local name="$1" etype="$2" model="$3"; shift 3
  local cache="$HERE/cache_${name}"
  local prefix="bench_${name}"
  local input
  case "$etype" in
    key1_text_score)   input="$DATA/unified_bench_eval_type1.jsonl" ;;
    key2_qa)           input="$DATA/unified_bench_eval_type2.jsonl" ;;
    key2_q_ma)         input="$DATA/unified_bench_eval_type3.jsonl" ;;
    key3_q_choices_a)  input="$DATA/unified_bench_eval_type4.jsonl" ;;
    key3_q_choices_as) input="$DATA/unified_bench_eval_type5.jsonl" ;;
    key3_q_a_rejected) input="$DATA/unified_bench_eval_type6.jsonl" ;;
    *) echo "unknown eval_type $etype" >&2; return 1 ;;
  esac
  echo "=== [$name] $etype model=$model $* ==="
  rm -rf "$cache"; mkdir -p "$cache"
  T6_INPUT="$input" T6_CACHE="$cache" T6_PREFIX="$prefix" \
    python "$HERE/bench_eval_task6.py" --eval-type "$etype" --model "$model" "$@" \
    2>&1 | tee "$OUT/run_${name}.log"
  if [ ! -f "$cache/${prefix}_step2.jsonl" ]; then
    echo "[FATAL] $name 没有产出 step2，中止" >&2
    exit 1
  fi
}

if [ "$WHICH" = "all" ] || [ "$WHICH" = "local" ]; then
  echo "=== [1/2] 本地模型 6 类型（Qwen2.5-0.5B-Instruct, transformers serving） ==="
  run_bench t1_local       key1_text_score    local
  run_bench t2_local       key2_qa            local
  run_bench t3_local       key2_q_ma          local
  run_bench t4_local       key3_q_choices_a   local
  run_bench t5_local       key3_q_choices_as  local
  run_bench t6_local       key3_q_a_rejected  local
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "api" ]; then
  echo "=== [2/2] API 模型（DeepSeek-V4-Flash, AMD 端点） ==="
  run_bench t2_api         key2_qa            api
  run_bench t2s_api        key2_qa            api --semantic-judge
  run_bench t3_api         key2_q_ma          api
  run_bench t5_api         key3_q_choices_as  api
fi

# 2b) 度量宽松度探针：官方 type2 自带 model_answer（15 条，4 条故意答错）直接进评测算子
if [ "$WHICH" = "all" ] || [ "$WHICH" = "local" ] || [ "$WHICH" = "probe" ]; then
  echo "=== [probe] 官方预填 model_answer 列（15 条，4 条故意答错）==="
  python "$HERE/probe_reference_answers.py" 2>&1 | tee "$OUT/probe_reference_answers.log"
  pcache="$HERE/cache_t2_ref_local"
  rm -rf "$pcache"; mkdir -p "$pcache"
  T6_INPUT="$DATA/unified_bench_eval_type2_ref.jsonl" T6_CACHE="$pcache" T6_PREFIX="bench_t2_ref_local" \
    python "$HERE/bench_eval_task6.py" --eval-type key2_qa --model local \
    2>&1 | tee "$OUT/run_t2_ref_local.log"
  if [ ! -f "$pcache/bench_t2_ref_local_step2.jsonl" ]; then
    echo "[FATAL] t2_ref_local 没有产出 step2，中止" >&2
    exit 1
  fi
fi

# 3) 核对（id 级重算 + 聚合对账 + ll 抽核 + 跨模型对比）
RUNS_JSON=$(cat <<'JSON'
{
  "t1_local":       {"cache": "@HERE@/cache_t1_local",  "prefix": "bench_t1_local",  "eval_type": "key1_text_score",   "model": "local"},
  "t2_local":       {"cache": "@HERE@/cache_t2_local",  "prefix": "bench_t2_local",  "eval_type": "key2_qa",           "model": "local"},
  "t3_local":       {"cache": "@HERE@/cache_t3_local",  "prefix": "bench_t3_local",  "eval_type": "key2_q_ma",         "model": "local"},
  "t4_local":       {"cache": "@HERE@/cache_t4_local",  "prefix": "bench_t4_local",  "eval_type": "key3_q_choices_a",  "model": "local"},
  "t5_local":       {"cache": "@HERE@/cache_t5_local",  "prefix": "bench_t5_local",  "eval_type": "key3_q_choices_as", "model": "local"},
  "t6_local":       {"cache": "@HERE@/cache_t6_local",  "prefix": "bench_t6_local",  "eval_type": "key3_q_a_rejected", "model": "local"},
  "t2_api":         {"cache": "@HERE@/cache_t2_api",    "prefix": "bench_t2_api",    "eval_type": "key2_qa",           "model": "api"},
  "t2s_api_semantic": {"cache": "@HERE@/cache_t2s_api", "prefix": "bench_t2s_api",   "eval_type": "key2_qa",           "model": "api"},
  "t3_api":         {"cache": "@HERE@/cache_t3_api",    "prefix": "bench_t3_api",    "eval_type": "key2_q_ma",         "model": "api"},
  "t5_api":         {"cache": "@HERE@/cache_t5_api",    "prefix": "bench_t5_api",    "eval_type": "key3_q_choices_as", "model": "api"},
  "t2_ref_local":   {"cache": "@HERE@/cache_t2_ref_local", "prefix": "bench_t2_ref_local", "eval_type": "key2_qa",       "model": "local"}
}
JSON
)
RUNS_JSON="${RUNS_JSON//@HERE@/$HERE}"
echo "$RUNS_JSON" > "$OUT/runs.json"
python "$HERE/verify_task6.py" --runs "$RUNS_JSON" --ll-spotcheck 3 \
  --out "$OUT/verify_task6.json" 2>&1 | tee "$OUT/verify_task6.log"

echo "=== done ==="
date -Is
