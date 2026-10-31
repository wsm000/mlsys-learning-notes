
#!/bin/bash
# DataFlow Task5 — 构建高质量训练数据 一键复现（vm-60 实测）
# 三条官方 pipeline：预训练数据过滤 / SFT 数据过滤 / SFT 数据合成
# 用法: bash run_all.sh            # 全量
#       bash run_all.sh pt         # 只跑预训练过滤
#       bash run_all.sh sft        # 只跑 SFT 过滤
#       bash run_all.sh syn        # 只跑 SFT 合成
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_SH="${T5_ENV_SH:-$HERE/env.sh}"
source "$ENV_SH"

WHICH="${1:-all}"
OUT="${T5_EVIDENCE:-$HERE/../evidence/task5}"
mkdir -p "$OUT"

echo "=== DataFlow Task5 :: $WHICH ==="
date -Is

# 0) 设计语料（本地生成，无外部下载）
python "$HERE/build_corpora.py" --out "$HERE/data"
echo "corpora: $(wc -l < "$HERE/data/pt_input.jsonl") pt rows, $(wc -l < "$HERE/data/sft_input.jsonl") sft rows"

if [ "$WHICH" = "all" ] || [ "$WHICH" = "pt" ]; then
  echo "=== [1/3] 预训练数据过滤 (PTTextFilter_GPUPipeline) ==="
  # 行类过滤器算子级单元核对（refiner 压平换行后的盲区，见 unit_check_line_filters.py）
  python "$HERE/unit_check_line_filters.py" 2>&1 | tee "$OUT/unit_check_line_filters.log"
  export T5_INPUT="$HERE/data/pt_input.jsonl"
  export T5_CACHE="$HERE/cache_pt"
  export T5_PREFIX="pt_filter_task5"
  rm -rf "$T5_CACHE"; mkdir -p "$T5_CACHE"
  python "$HERE/pt_filter_task5.py" 2>&1 | tee "$OUT/pt_filter_run.log"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "sft" ]; then
  echo "=== [2/3] SFT 数据过滤 (SFTTextFilter_GPUPipeline) ==="
  export T5_INPUT="$HERE/data/sft_input.jsonl"
  export T5_CACHE="$HERE/cache_sft"
  export T5_PREFIX="sft_filter_task5"
  rm -rf "$T5_CACHE"; mkdir -p "$T5_CACHE"
  python "$HERE/sft_filter_task5.py" 2>&1 | tee "$OUT/sft_filter_run.log"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "syn" ]; then
  echo "=== [3/3] SFT 数据合成 (TextSFTSynthesis_APIPipeline) ==="
  export T5_CACHE="$HERE/cache_syn"
  export T5_PREFIX="sft_synthesis_task5"
  rm -rf "$T5_CACHE"; mkdir -p "$T5_CACHE"
  python "$HERE/sft_synthesis_task5.py" 2>&1 | tee "$OUT/sft_synthesis_run.log"
fi

# 4) 核对（id 级去向 / 分数分布 / 字段完整性）
python "$HERE/verify_task5.py" \
  --pt-cache "$HERE/cache_pt" --pt-prefix pt_filter_task5 \
  --pt-input "$HERE/data/pt_input.jsonl" \
  --sft-cache "$HERE/cache_sft" --sft-prefix sft_filter_task5 \
  --sft-input "$HERE/data/sft_input.jsonl" \
  --syn-cache "$HERE/cache_syn" --syn-prefix sft_synthesis_task5 \
  --manifest-pt "$HERE/data/manifest_pt.json" \
  --manifest-sft "$HERE/data/manifest_sft.json" \
  --out "$OUT/verify_task5.json" 2>&1 | tee "$OUT/verify_task5.log"

echo "=== done ==="
date -Is
