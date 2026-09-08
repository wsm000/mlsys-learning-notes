# 打卡材料（hello-gpu Day1–5）

> 原根目录 `打卡材料/README.md` 的证据清单，随仓库重组迁到这里。根目录 `打卡材料/` 现为指路页。

这个目录只存放真实运行产生的截图和原始输出；不要用教程样例或他人设备输出替代本机证据。

## 建议文件名

| 文件 | 内容 |
| --- | --- |
| `day1_environment_vector_add.png` | GPU/ROCm 信息、运行命令和 `RESULT: PASS`。 |
| `day1_mapping_n10_block4.png` | N=10、Block=4 的完整线程映射，i=10、11 标为越界。 |
| `day1_environment_vector_add.txt` | 第一张截图对应的未裁剪终端输出。 |
| `day2_single_vs_warmup.png` | 单次同步墙钟与 warmup + repeat GPU Event 结果。 |
| `day2_protocol_and_statistics.png` | 输入、warmup、repeat、同步/Event 与完整统计结果。 |
| `day2_benchmark_output.txt` | Day 2 脚本未裁剪的完整终端输出。 |
| `day3_rocprof_kernel_trace.png` | Kernel 名称、Grid/线程划分与 Kernel 时间（trace 输出或 CSV 列）。 |
| `day3_coalesced_vs_strided.png` | 同一协议下连续访存与跨步访存的延迟、有效带宽（可含 Roofline 工作点）。 |
| `day3_hypothesis.txt` | 一条“证据—当前判断—下一步单变量实验”，未裁剪原始输出一并保存。 |
| `day3_performance_record.md` | 脚本重新运行后生成的本机性能记录（可从 logs/day3_rerun_<时间>/performance_record.md 复制）。 |
| `day5_agent_trace.png` | Agent 运行轨迹：候选生成、compile/bench/profile 工具调用与返回（chapter17.ipynb「步骤5」输出，可同屏「步骤4」baseline bench）。 |
| `day5_candidate_verdict.png` | 候选源码（best.py）与裁决：正确性门禁结果 + 接受/拒绝理由（chapter17.ipynb「步骤6」轨迹表 + 工作区 best.py）。 |
| `day5_compare_baseline_day4_agent.png` | 同一输入与计时协议下 baseline、Day 4 人工版、Agent 版的 median/min 与加速比（chapter17.ipynb「步骤8」报告 + 自制三行表）。 |
| `day5_trajectory.jsonl` | 从 runs/part3-agent/chapter17/<run_id>/ 复制，保留每轮 change/status/accepted/latencyMs。 |
| `day5_agent_report.md` | Agent 本轮报告与最终结论（未加速时如实记录原因）。 |

## 提交前检查

- [ ] Day 1 环境截图中有真实 GPU 或 ROCm 信息、命令和正确性结果。
- [ ] Day 1 映射图中有 Block 0: 0--3、Block 1: 4--7、Block 2: 8--11，且 i=10、11 明确写为越界。
- [ ] Day 2 截图显示同一 baseline 的“只运行一次”与“warmup + repeat”结果，而非两段不同代码。
- [ ] Day 2 协议显示 N、warmup、repeat、同步或 GPU Event、mean、median、min。
- [ ] Day 2 结论没有把单个最快数字或 CPU enqueue 时间写成 Kernel 性能。
- [ ] Day 3 trace 截图来自本机 rocprofv3 运行，且能看到 Kernel_Name 与 Grid_Size。
- [ ] Day 3 对比截图使用同一输入规模与计时协议（N、block、warmup、repeat 一致）。
- [ ] Day 3 假设包含具体单变量实验，而不是“需要继续优化”。
- [ ] Day 5 轨迹截图显示候选生成与 compile/bench/profile（或 accept）工具调用及返回，非模型口头加速比。
- [ ] Day 5 候选源码（best.py）与裁决同屏/同日志，标注门禁通过与否及接受/拒绝理由。
- [ ] Day 5 对比表使用同一任务契约输入与计时协议；Day 4 人工版若口径不同必须在本章契约下重测并注明。
- [ ] Day 5 未加速时保留 trajectory.jsonl 与失败原因，不修改输入或评测规则强行出加速。
