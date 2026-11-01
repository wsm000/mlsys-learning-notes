# Task4 打卡图 / 运行结果图（6 张，1920×1080 PNG）

> 生成方式：`python code/task4/make_figures.py`（只用 Pillow，不联网）
> **诚实说明**：这些图是脚本从 `evidence/task4/` 的真实 JSONL / JSON / 原始日志中读取数字后绘制的可视化，不是终端/网页截图；`runshot_task4_06_日志摘录.png` 是真实日志去 ANSI 后渲染，也明确标注「渲染图，非截屏」。所有数字均可回溯到证据文件（图底部标注了数据来源）。

| 图 | 内容 | 数据来源 |
|---|---|---|
| `runshot_task4_01_打卡总览.png` | 打卡用总览：两条 pipeline 关键指标 + 限流约束 | 全部证据汇总 |
| `runshot_task4_02_agentic_rag_result.png` | AgenticRAG：F1 逐样本、双过滤、字段齐全、漏斗、成本 | `agentic_rag/agentic_rag_task4_step2_FINAL.jsonl`、`field_check_main.json`、`run_main.log` |
| `runshot_task4_03_text2sql_refine.png` | Text2SQL refine：10 步行数漏斗、难度分布、长度、最终样例 | `text2sql/refine/stats_refine.json`、`quality_checks.json`、`final_refine.jsonl`、`refine.log` |
| `runshot_task4_04_text2sql_gen.png` | Text2SQL gen：9 步漏斗、3 库/复杂度、过滤与限制 | `text2sql/gen/summary_gen.json`、`gen.log`、`samples_gen.jsonl` |
| `runshot_task4_05_429_throughput.png` | 限流与吞吐：各次运行 status=429 计数、限速前后对比、成本含义 | 五份原始日志正则统计 `status=429` |
| `runshot_task4_06_日志摘录.png` | 真实日志关键行摘录（AgenticRAG / refine / gen） | `run_main.log`、`refine.log`、`gen.log` |

## 复现

```bash
# 需要 Pillow（DSH 内置 Python 已带）
python courses/10-dataflow/code/task4/make_figures.py
# 输出到 courses/10-dataflow/evidence/task4/figures/
```

## 数字口径

- AgenticRAG：4 文档 → 8 候选任务 → 7 通过 `llm_score<1` → 每文档 `max_question=1` → 最终 4 条；F1=[1.0,1.0,0.7826,0.0]；66 HTTP 请求 / 654.7s。
- Text2SQL refine：4 条输入 → 10 步 → 一致性过滤 4→3 → 最终 3 条；5 分 05 秒；0×429。
- Text2SQL gen：3 库×1 条 SQL → 9 步 → 一致性过滤 3→2；0×429 + 2×504。
- 429 计数：AR pilot 40、smoke step10 29、AR 正式 3、refine 0、gen 0（均来自日志正则统计）。
