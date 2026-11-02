#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task6 — 度量宽松度探针：用官方 type2 数据集自带的 model_answer 列（15 条预填答案，
其中 math_006/008/012/014 四条是故意答错的）跑同一评测算子，量化 math_verify 路径的判定行为。

做法：把 model_answer 复制成 generated_ans 列（BenchAnswerGenerator 因输出列已存在且
allow_overwrite=False 会跳过生成），直接进 UnifiedBenchDatasetEvaluator。
若 4 条错答案全部判 0 -> accuracy=11/15，指标对"明确答错"是严格的；
我们的生成答案拿到 15/15 里若有仅靠子串包含得分的行，则说明短标签会被宽松规则放大。
"""
import json
import os

SRC = os.environ.get("T6_INPUT") or "./data/unified_bench_eval_type2.jsonl"
DST = os.environ.get("T6_REF_OUT") or "./data/unified_bench_eval_type2_ref.jsonl"


def main():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["generated_ans"] = r.get("model_answer", "")
            r.pop("model_answer", None)
            rows.append(r)
    with open(DST, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[REF] {len(rows)} rows -> {DST} (generated_ans <- model_answer)")
    for r in rows:
        print(f"  {r['id']}: gt={r['golden_label']!r}")


if __name__ == "__main__":
    main()
