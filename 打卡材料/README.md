# 打卡材料

这个目录只存放真实运行产生的截图和原始输出；不要用教程样例或他人设备输出替代本机证据。

## 建议文件名

| 文件 | 内容 |
| --- | --- |
| <code>day1_environment_vector_add.png</code> | GPU/ROCm 信息、运行命令和 <code>RESULT: PASS</code>。 |
| <code>day1_mapping_n10_block4.png</code> | N=10、Block=4 的完整线程映射，i=10、11 标为越界。 |
| <code>day1_environment_vector_add.txt</code> | 第一张截图对应的未裁剪终端输出。 |
| <code>day2_single_vs_warmup.png</code> | 单次同步墙钟与 warmup + repeat GPU Event 结果。 |
| <code>day2_protocol_and_statistics.png</code> | 输入、warmup、repeat、同步/Event 与完整统计结果。 |
| <code>day2_benchmark_output.txt</code> | Day 2 脚本未裁剪的完整终端输出。 |

## 提交前检查

- [ ] Day 1 环境截图中有真实 GPU 或 ROCm 信息、命令和正确性结果。
- [ ] Day 1 映射图中有 Block 0: 0--3、Block 1: 4--7、Block 2: 8--11，且 i=10、11 明确写为越界。
- [ ] Day 2 截图显示同一 baseline 的“只运行一次”与“warmup + repeat”结果，而非两段不同代码。
- [ ] Day 2 协议显示 N、warmup、repeat、同步或 GPU Event、mean、median、min。
- [ ] Day 2 结论没有把单个最快数字或 CPU enqueue 时间写成 Kernel 性能。
