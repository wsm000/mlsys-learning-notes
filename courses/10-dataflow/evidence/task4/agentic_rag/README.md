# Task4 · AgenticRAG 证据目录

本目录是 **vm-60 上 2026-10-29 22:33–22:44（北京时间）** 一次真实运行的产物，
对应 Pipeline：DataFlow `AgenticRAGAtomicTaskGenerator` + `AgenticRAGQAF1SampleEvaluator`；
LLM = AMD Radeon OpenAI 兼容端点 `DeepSeek-V4-Flash`。

## 文件

| 文件 | 内容 | 关键数字 |
|---|---|---|
| `run_main.log` | 正式运行完整 stdout（阶段日志、tqdm、每次 429/504 原文） | 654.7 s，66 次 HTTP 请求 |
| `agentic_rag_task4_step1.jsonl` | Step1 原子问答生成输出 | 4 行 × 12 列 |
| `agentic_rag_task4_step2_FINAL.jsonl` | **最终输出**（含 F1Score） | 4 行 × 13 列 |
| `field_check_main.json` | 字段完整性 / 空值率 / F1 分布 / 打分统计 | 11/11 字段，空值率 0%，F1 mean 0.6957 |
| `sample_first_row.json` | 首行完整样例（长文本已标注截断） | id=4864947，F1=1.0 |
| `input_main_4docs.jsonl` | 实际输入子集（可复现，非随机） | 4 条，id=17989685/4864947/12455656/18253508 |
| `run_pilot_429_evidence.log` | **失败证据**：限流 bug 的 pilot 日志 | 40 次 `status=429`，step1 输出 0 行 |

## 复现

见 `courses/10-dataflow/code/task4/agentic_rag/REPORT.md` §6，脚本：
`agentic_rag_task4.py` + `run_agentic_rag.sh` + `verify_task4_agentic_rag.py`。

## 注意

- `llm_score` 全 0、`golden_doc_score` 均 ≥1 是**算子设计的两级过滤**结果，不是数据缺失。
- 第 4 行 `refined_answer="2017-08-30"` vs `golden_doc_answer="August 30, 2017"` → token-F1=0，
  是评分器**格式敏感**的真实反例（见 REPORT §10.1）。
