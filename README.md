# hello-gpu 学习记录

这里集中保存 [datawhalechina/hello-gpu](https://github.com/datawhalechina/hello-gpu) 的学习笔记、可复现实验和打卡证据。每个 Day 都遵循仓库根目录的 [项目执行规则](../项目执行规则.md)：先完成打卡必做项，再进入拓展。

## 学习资料

- [教程仓库](https://github.com/datawhalechina/hello-gpu)
- [五日学习指南](https://zcnijjcepfie.feishu.cn/docx/RtWQdHmztoAXvQxmPX6cSUy6nug)
- [学习环境说明](https://zcnijjcepfie.feishu.cn/docx/PurhdJlaaoSzWzxwL26cw497nxd)

## 目录

| 路径 | 用途 |
| --- | --- |
| [Day1_认识GPU并跑通程序.md](Day1_认识GPU并跑通程序.md) | GPU 并行模型、Vector Add、线程映射与 Day 1 打卡材料。 |
| [Day2_学会可信计时.md](Day2_学会可信计时.md) | warmup、同步、GPU Event、统计口径与 Day 2 打卡材料。 |
| [day1_vector_add/vector_add_hip.cpp](day1_vector_add/vector_add_hip.cpp) | 独立 HIP Vector Add：打印设备和 launch 信息并验证正确性。 |
| [day1_vector_add/thread_mapping.py](day1_vector_add/thread_mapping.py) | 输出 N=10、Block=4 的线程映射表，适合制作 Day 1 第二张截图。 |
| [day2_timing/benchmark_vector_add.py](day2_timing/benchmark_vector_add.py) | 以 PyTorch ROCm 测量同一 Vector Add 的单次、错误 CPU enqueue 与可信 GPU Event 结果。 |
| [打卡材料](打卡材料/README.md) | 截图和原始终端输出的命名、存放与提交前检查。 |

## 范围说明

活动学习指南把 Day 1 定义为教程第 1--4 章，把 Day 2 的打卡核心定义为第 5 章的可信计时。第 6 章是 rocprofv3 与性能瓶颈分析，按五日学习指南属于 Day 3；由于本次安排写有“第 5--6 章”，Day 2 笔记保留了第 6 章预习入口，但不把它替代 Day 2 的计时打卡证据。

## 当前环境状态

2026-08-22 在本工作区检查到 Windows 10 和 Python 3.10.2，未发现 <code>rocminfo</code>、<code>hipcc</code>、<code>rocm-smi</code> 或 <code>amd-smi</code>。因此仓库内没有伪造的 GPU 型号、性能数字或“通过”截图。请在有 AMD GPU 的 ROCm Linux 或可用 WSL2 ROCm 环境中运行下面的代码，再把真实输出放入 [打卡材料](打卡材料/README.md)。

## 推荐完成顺序

1. 阅读 Day 1 笔记，完成 ROCm 三道环境验证、编译运行 HIP 程序，并保存两项截图。
2. 阅读 Day 2 笔记，对固定输入执行一次完整计时，保留完整协议和三组结果。
3. 逐项勾选各 Day 的 Gate 0；未完成前不把第 6 章 profiling 或其他优化结果当作打卡替代品。
