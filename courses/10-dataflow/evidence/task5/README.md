# Task5 证据目录

| 文件 | 内容 |
|---|---|
| pt_filter_run.log | 预训练数据过滤 25 步漏斗日志（每步 in/out 行数、耗时） |
| sft_filter_run.log | SFT 数据过滤 3 步日志（含 FALLBACK 标注） |
| sft_synthesis_run.log | SFT 数据合成 3 步日志 |
| verify_task5.json | id 级去向核对 + 分数分布 + API 统计（机器可读） |
| unit_check_line_filters.json | 三个行比例过滤器的算子级核对（refiner 前后对照） |
| unit_check_line_filters.log | 同上，控制台版 |
| corpora/ | 设计语料 pt_input.jsonl / sft_input.jsonl + 缺陷清单 manifest_*.json |
| steps_pt/ | 预训练过滤逐步产物（step1-step25，含 PairQualScore/language_label/minhash 列） |
| steps_sft/ | SFT 过滤逐步产物（含 SuperfilteringScore/AlpagasusScore 列） |
| steps_syn/ | SFT 合成逐步产物 + api_calls.jsonl（请求/响应审计，不含密钥） |
| dataflow-task5-checkin.md | 打卡文案 |

运行环境：vm-60（RTX 4090 D 24GB）· open-dataflow 1.0.11 · 2026-10-31 21:35-22:03 CST。
复现：见 ../../notes/task05-构建高质量训练数据.md 第 5 节。