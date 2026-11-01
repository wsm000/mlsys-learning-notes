# Task4 证据索引（courses/10-dataflow/evidence/task4/）

2026-10-29 vm-60 真实运行。入口文档：[Task4 笔记](../../notes/task04-构建RAG与Text2SQL数据.md) · [数据需求分析](../../notes/task04-不同AI应用场景数据需求.md) · [打卡](dataflow-task4-checkin.md)

## agentic_rag/ — AgenticRAG 数据合成（4 条文档 → 4 条 QA）

| 文件 | 内容 |
|---|---|
| run_main.log | 完整运行日志（含 CONFIG / 各阶段 / 66 次调用 / 654.7s / 限流 429 原文） |
| agentic_rag_task4_step1.jsonl | 原子生成产物（4 行 × 11 生成字段） |
| agentic_rag_task4_step2_FINAL.jsonl | 最终产物（4 行，含 F1Score=[1.0,1.0,0.7826,0.0]） |
| field_check_main.json | 字段完整性/分布检查（机读） |
| sample_first_row.json | 首行完整样例 |
| input_main_4docs.jsonl | 实际输入（官方 eval_test_data 前 4 条） |
| run_pilot_429_evidence.log | 第一次尝试因 429 失败（缺列报错）的证据，用于记录失败模式 |

## text2sql/ — Text2SQL 数据流水线

| 路径 | 内容 |
|---|---|
| refine/refine.log | 正式 refine 全量日志（10 步，5 分 05 秒，0 次 429） |
| refine/cache/task4_refine_step1..10.jsonl | 每一步 cache 原始产物（step10=最终） |
| refine/final_refine.jsonl | 最终 3 条（10 列全非空） |
| refine/stats_refine.json / summary_refine.json | 每步行数、字段、参数快照、难度分布、429 统计 |
| refine/quality_checks.json / samples_refine.jsonl | 质量检查（prompt/cot 长度、分布）与完整样例 |
| gen/cache/task4_gen_step1..9.jsonl | gen 合成模式逐步产物（3 库×1 SQL → 一致性过滤后 2 行） |
| gen/summary_gen.json / stats_gen.json / samples_gen.jsonl | gen 统计与样例 |
| smoke/ | smoke 链路证据（2 条到 step9；step10 因 429 未完成，日志含原文） |
| db_stats.json | 3 个 SQLite 库的表/列/行数统计 |
| REPORT.md | Text2SQL 运行报告（每步分析、429 时间线、吞吐估算、复现） |

## docs/ — 官方文档本地快照

| 文件 | 内容 |
|---|---|
| agenticrag_pipeline.txt | 官方 AgenticRAG 数据合成文档（2025-07-14） |
| text2sqlpipeline.txt | 官方 Text-to-SQL 数据合成文档（2025-06-17） |

## figures/ — 打卡图 / 运行结果图（6 张）

由 `code/task4/make_figures.py` 从本目录真实 JSONL/JSON/日志生成（非示意数据，详见 [figures/README.md](figures/README.md)）：

- [runshot_task4_01_打卡总览.png](figures/runshot_task4_01_打卡总览.png)：打卡总览（两条 pipeline 关键指标 + 限流约束）
- [runshot_task4_02_agentic_rag_result.png](figures/runshot_task4_02_agentic_rag_result.png)：AgenticRAG F1/双过滤/漏斗
- [runshot_task4_03_text2sql_refine.png](figures/runshot_task4_03_text2sql_refine.png)：refine 10 步漏斗与样例
- [runshot_task4_04_text2sql_gen.png](figures/runshot_task4_04_text2sql_gen.png)：gen 9 步漏斗与限制
- [runshot_task4_05_429_throughput.png](figures/runshot_task4_05_429_throughput.png)：429 计数与吞吐
- [runshot_task4_06_日志摘录.png](figures/runshot_task4_06_日志摘录.png)：真实日志关键行（渲染）

## 其他

- dataflow-task4-checkin.md：打卡文案
- 代码：`code/task4/agentic_rag/`、`code/task4/text2sql/`（改造脚本 + 复现 shell + REPORT）
