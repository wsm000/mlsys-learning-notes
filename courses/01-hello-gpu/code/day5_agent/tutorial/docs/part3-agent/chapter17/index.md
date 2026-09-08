---
title: "第17章 多轮优化实战"
description: "Hello GPU 第17章 · vector_add 真实轨迹 ≈2.19×、失败回退、对比报告"
---

# 第17章 多轮优化实战

## 本章目标、前置知识与产物

本章是算子 Agent 篇的高潮：把第 15 章的工具与第 16 章的循环放在一起，跑一次完整多轮优化，然后——**如实展示结果**。

本章主角是 **`vector_add` fixtures**，硬件为 **Radeon 8060S（`gfx1151`）+ ROCm 7.12**，工作区 `task-20260806-144647`。

先说结论，这也是本书和「Agent 自动优化一切」类教程的区别：

- 在**故意留头寸**的朴素 `vector_add` 上，本轮 Agent 搜索把延迟从约 **0.705 ms → 0.322 ms（≈2.19×）**，10 轮评估中 **5 次接受**；
- 加速主要来自 **更大 `block_size` + 更高 `num_warps`**（减 grid 启动、提高并行度），符合 memory-bound 直觉；
- 教程承诺的「3–5×」针对**有明显优化空间**的算子（融合 softmax、naive reduction/matmul 等）。`vector_add` 靠近带宽墙，用来证明**闭环可信**与**失败回退**，不是证明极限加速。

学完本章，你应该能够：

- 按留痕规则读懂 `trajectory.jsonl`；
- 从性能曲线指出哪几轮提升最大、哪几轮该回退；
- 写一份区分事实与假设的对比报告。

对应代码与产物：

```text
code/part3-agent/chapter15/
├── fixtures/vector_add/
│   ├── baseline.py
│   ├── reference.py
│   └── task.json
└── logs/tasks/task-20260806-144647/   # 本章引用的真实工作区
    ├── trajectory.jsonl
    ├── best.py
    ├── hardware.json
    └── viz/

code/part3-agent/chapter16/
├── run_part3_test.py
├── visualize_trajectory.py
└── run_and_visualize.sh
```

## 17.1 选一个教学算子

| 算子 | 用途 | 预期 |
|---|---|---|
| **vector_add**（本章默认） | 验证 Agent 闭环、评测器、非交互入口 | 中小幅到约 2×（视 baseline 有多朴素）；重在轨迹完整 |
| masked-softmax / reduction / naive matmul | 冲击 3–5× 叙事 | 融合、LDS、向量化等机制有头寸 |

选型原则：

1. 有可执行的 Triton baseline 与 PyTorch reference；
2. `costModel` 能区分 bound 类型；
3. 故意留一点「可改空间」（如偏小 `block_size`），否则 Agent 只能在噪声里打转。

本章任务契约摘要：

| 字段 | 内容 |
|---|---|
| Objective | 元素级向量加，与 reference 一致 |
| Shape / dtype | `4096×2048`，fp16，约 8.4M 元素 |
| Correctness | 与 PyTorch reference 比对 |
| Promotion | 配对中位改进 ≥ 1%，多数配对为正 |
| Constraints | Triton on ROCm |

## 17.2 记录每轮优化的轨迹

优化之前先定「留痕规则」。工作区固定若干证据文件：

```text
task-20260806-144647/
├── task.json / reference.py / baseline.py / best.py
├── hardware.json          # measure_peak 副本
├── trajectory.jsonl       # 每行一轮：change / status / accepted / latencyMs / …
├── run_summary.json
└── viz/                   # 可视化 PNG
```

规则只有一条：**任何候选版本，无论成败，都留下记录。**

本轮 Agent 步内工具序列（历史跑次工具名曾为 `evaluate_candidate` / `profile`；**当前代码**请用三件套 + `accept_candidate`）：

| Agent 步 | 当时工具 | 现今对应 | 关键返回 |
|---|---|---|---|
| 1 | `get_environment` | 同左 | Radeon 8060S / gfx1151 / ROCm 7.12 |
| 1 | `measure_peak` | 同左 | 带宽 **210.5 GB/s**；fp16 **27.8 TFLOPS** |
| 2 | `profile` | `profile_kernel` | AI≈**0.167** → **memory-bound** |
| 3–12 | `evaluate_candidate` ×10 | `compile`+`bench`+`accept` | **5 接受 / 5 拒绝** |

## 17.3 观察性能提升曲线

硬件与算子画像（优化前）：

| 指标 | 数值 |
|---|---|
| 设备 | Radeon 8060S Graphics · `gfx1151` |
| ROCm / torch | 7.12 / `2.10.0+rocm7.12.0` |
| 带宽峰值 | 210.5 GB/s |
| AI / bound | 0.167 · memory-bound |
| Baseline 特征 | `block_size=256`，未设 `num_warps` |
| Baseline 中位延迟（反推） | ≈ **0.705 ms** |

逐轮账本（`improvementFraction` 相对**当时 incumbent**，不是相对最初 baseline）：

| 轮 | 改动 | 延迟 (ms) | 相对 incumbent | 裁决 |
|---|---|---|---|---|
| 1 | `block_size` 256→1024 | 0.378 | **+46.34%** | ✓ 接受 |
| 2 | + `num_stages=2` | 0.372 | -0.32% | ✗ 阈值下 |
| 3 | `block_size`→2048 | 0.342 | -1.01% | ✗ |
| 4 | `block_size`→4096 | 0.338 | **+4.64%** | ✓ 接受 |
| 5 | `block_size`→8192 | 0.405 | -17.34% | ✗ 明显变慢 |
| 6 | 4096 + `num_stages=2` | 0.340 | +0.34% | ✗ &lt;1% |
| 7 | 4096 + `num_warps=8` | 0.329 | **+3.46%** | ✓ 接受 |
| 8 | `num_warps=16` | 0.326 | **+1.24%** | ✓ 接受 |
| 9 | `num_warps=32` | **0.322** | **+1.50%** | ✓ 接受（最终 best） |
| 10 | + `num_stages=2` | 0.323 | +0.50% | ✗ &lt;1% |

相对最初 baseline：

| 指标 | Baseline | Best（R9） | 变化 |
|---|---|---|---|
| 中位延迟 | ≈ 0.705 ms | **0.322 ms** | **≈2.19×**（延迟降约 54%） |
| `block_size` | 256 | **4096** | grid 启动次数约减 16× |
| `num_warps` | 默认 | **32** | 提高 CU 占用与并行度 |

曲线形状：

- **最大单轮收益**来自 R1 放大 `block_size`（配置搜索，减启动开销）；
- R7–R9 的 `num_warps` 阶梯是二次提升；
- R5 `block_size=8192` 是结构/资源边界上的负结果——过大 block 伤害占用或调度。

### 可视化

#### 延迟与配对改进总览

![延迟曲线与配对改进百分比](./images/rounds_overview.png)

上图绿线为 best-so-far latency；下图红虚线为 1% 接受阈值；绿柱接受、橙柱拒绝。R5 大负柱对应 8192 失败试探。

#### 每轮改动时间线

![每轮改动与裁决时间线](./images/process_timeline.png)

#### 状态分布

![接受与拒绝状态分布](./images/status_breakdown.png)

10 轮：**接受 = 5**，**未达阈值 = 5**，本轮无编译失败。

最终 `best.py` 要点：

```python
block_size = 4096
_vadd_kernel[grid](..., block_size, num_warps=32)
```

## 17.4 失败回退机制

回退不是单独的「undo 工具」，而是评测器契约：

```text
候选失败或未过阈值 → accepted=false → 不写 best.py → incumbent 保持上一版
```

本轮最有教学价值的失败是 **R5：`block_size=8192`**——编译与正确性都过，但中位改进 **-17.3%**，明显变慢。Agent 拒绝后继续从 4096 路线搜索 `num_warps`，没有把坏候选写进 best。

失败记录的价值：

> 「过大 block 在本机 vector_add 上是负优化」这条结论，来自一次失败实验，却能阻止后续盲目把 `block_size` 推到极限。

Agent 侧配合：把拒绝原因读进下一轮提示；SOP 要求「连续同类失败就换机制」；轨迹保留失败轮次，便于报告诚实引用。

## 17.5 和人工优化的对比

| 维度 | 本轮 Agent | 人工典型做法 |
|---|---|---|
| 发现方向 | 在 memory-bound SOP 引导下试 block / warps | 一次选定较大 block + 合理 warps |
| 执行与留痕 | 10 轮全自动评测 + `trajectory.jsonl` | 常靠笔记，易丢负结果 |
| 最终结果 | ≈2.19× vs 故意朴素 baseline | 熟练者可能更少轮次到达相近点 |
| 局限 | 未发明新算法结构；步数耗尽即停 | 需要人盯着跑实验 |

诚实结论：

1. Agent 当前强项是 **执行 + 记录**：给定方向空间，能快速产出变体并自动验证；
2. Agent 弱项仍是 **发现全新结构**（线上 fused MLP 对照实验里纯 Agent 贡献接近 0%——见线上第 17.5 节）；
3. 「3–5×」的正确打开方式往往是 **人给关键洞察，Agent 快速验证并留痕**。本轮 2.19× 发生在「baseline 故意很差」的前提下，不要外推到已经接近带宽墙的生产 kernel。

## 17.6 生成对比报告

好的报告回答五个问题：

| 问题 | 本轮回答 |
|---|---|
| 题目是什么 | `vector_add` fp16 `4096×2048`；配对改进 ≥1% |
| 每轮改了什么 | 上表：block_size / num_warps / num_stages |
| 哪些失败了、为什么 | R5 过大 block；多次 `num_stages` 未过阈值 |
| 最终结论 | 0.322 ms，≈2.19×；best = 4096 + num_warps=32 |
| 事实 vs 推测 | 轨迹与 `hardware.json` 是事实；「还能再快」是推测 |

更细的工具调用链与指标说明见同目录 [optimization-report.md](./optimization-report.md)。

### 如何复现

```bash
cd code/part3-agent
uv sync && source ./activate-rocm.sh

# 需要 ~/.config/hello-gpu/kernel-agent.env（勿提交 git）
bash chapter16/run_and_visualize.sh --skip-pytest
```

前置检查：

```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.__version__)"
# 本机实测示例：Radeon 8060S Graphics  2.10.0+rocm7.12.0
```

## 本章小结

- 实战首先证明闭环可信，再追求 3–5×；算子选型决定加速叙事。
- 本轮真实结果：10 评 / 5 接受，**0.705→0.322 ms（≈2.19×）**，有效机制是更大 block + 更高 warps。
- 失败回退由「不更新 best」保证；轨迹 + 可视化是报告账本。
- Agent 擅长执行与留痕；关键结构洞察仍常需人机协作。

## 延伸阅读

- [算子优化 Agent 实战报告 · vector_add](./optimization-report.md)
- `code/part3-agent/chapter14/EXPERIMENT.md`

