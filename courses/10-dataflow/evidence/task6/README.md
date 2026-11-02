# Task6 证据目录

| 文件 | 内容 |
|---|---|
| run_all.log | 11 个 run 的总日志（数据集构建 → 10 个 bench run → 预填答案探针 → 核对） |
| run_t1_local.log … run_t6_local.log | 本地 Qwen2.5-0.5B-Instruct 六个 eval_type 运行日志（每步 in/out 行数、耗时、RESULT 聚合） |
| run_t2_api.log / run_t2s_api.log / run_t3_api.log / run_t5_api.log | AMD 端点 DeepSeek-V4-Flash 四个 run 日志（含限流与逐步结果） |
| run_t2_ref_local.log | 度量宽松度探针（官方预填 model_answer 列直接进评测算子） |
| probe_reference_answers.log | 探针输入构建日志（15 条预填答案，4 条故意答错） |
| verify_task6.json | 机器可读核对结果：11 个 run 的聚合对账 / 逐 id 重算 / ll 抽核 / 完整性 / API 审计 / 跨模型对比（all_ok=True） |
| verify_task6.log | 核对控制台输出 |
| runs.json | 核对用的 run 清单（cache/prefix/eval_type/model） |
| build_data.log | 数据集构建日志（官方 example 原样 + sha256） |
| env_facts.txt | vm-60 环境事实（GPU/驱动/Python/open-dataflow/torch/transformers/模型文件/端点连通性） |
| data/ | 六个官方数据集（unified_bench_eval_type{1..6}.jsonl）+ 探针输入 _ref.jsonl + manifest.json（sha256）+ official_pipelines/（官方 7 个 pipeline 原型拷贝） |
| steps/cache_*/ | 每个 run 的逐步产物：*_step1.jsonl（生成后）、*_step2.jsonl（评测后，含 eval_score/eval_pred/eval_valid/eval_error）、eval_result/eval_result.jsonl（聚合）、run_meta.json（配置快照）、api_calls.jsonl（API run 的请求/响应审计，不含密钥） |
| figures/runshot_task6_01_run_matrix.png | 11 个 run 总览（类型/指标/模型/结果/核对） |
| figures/runshot_task6_02_local_six_types.png | 本地模型六类型能力表 |
| figures/runshot_task6_03_model_comparison.png | 同一基准跨模型对比（0.5B vs V4-Flash）+ semantic 变体 |
| figures/runshot_task6_04_metric_leniency_probe.png | 度量宽松度探针（预填答案 11/15 vs 生成答案 15/15 + 命中路径归因） |
| figures/runshot_task6_05_ll_metrics_detail.png | loglikelihood 类指标逐 id 明细（单选 argmax / 偏好 ll 大小） |
| figures/runshot_task6_06_verify_checklist.png | 核对清单（含 ll 抽核 3/3） |
| dataflow-task6-checkin.md | 打卡文案 |

运行环境：vm-60（RTX 4090 D 24GB，驱动 570.211.01）· open-dataflow 1.0.11 · torch 2.9.1+cu128 · transformers 4.57.6 · math-verify 0.9.0 · 2026-11-02 15:41-16:14 CST。
runshot 图由 code/task6/render_runshots.py 从本目录的真实日志/JSON 渲染（终端风格，非手工截图）。
复现：见 ../../notes/task06-从数据到效果评估模型能力.md 第 6 节。
