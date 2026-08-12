# Task 1：性能分析入门 / Hello Roofline 学习笔记

## 1. 任务定位

Task 1 的正式要求来自 [Datawhale issue #76](https://github.com/datawhalechina/llm-algo-leetcode/issues/76)：理解 Roofline，使用 MLSys·im 分析 ResNet-50 + A100，改变 batch size，并至少再改变一种硬件或模型。本文还完成了三硬件高级对比。Hello Roofline 的核心问题是：

> 给定一个工作负载和一块硬件，判断它更可能受计算吞吐还是内存带宽限制。

官方 Lab 11 Part A 的核心活动是：改变矩阵规模和精度，观察工作负载点如何跨过 ridge point，并记录算术强度、可达吞吐、MFU 和 regime。

参考资料：

- [课程 Lab 11 Track Plan](https://github.com/harvard-edge/cs249r_book/blob/dev/labs/vol1/lab_11_hw_accel.track-plan.md)
- [Roofline 辅助实现](https://github.com/harvard-edge/cs249r_book/blob/dev/labs/mlsysbook_labs/roofline.py)
- [Datawhale Task 1](https://github.com/datawhalechina/llm-algo-leetcode/issues/76)
- [Datawhale Task 0](https://github.com/datawhalechina/llm-algo-leetcode/issues/65)

## 2. Roofline 要回答什么

Roofline 不是直接测量真实 GPU 时间，而是用两个上限估算工作负载的可达性能：

```text
算力上限       = Peak FLOP/s
带宽上限       = Memory Bandwidth × Arithmetic Intensity
可达性能       = min(算力上限, 带宽上限)
```

其中：

```text
Arithmetic Intensity, AI = FLOPs / Bytes moved
Ridge Point              = Peak FLOP/s / Memory Bandwidth
```

判断规则：

```text
AI < Ridge Point  -> Memory-bound，受内存带宽限制
AI >= Ridge Point -> Compute-bound，受计算吞吐限制
```

单位必须统一。例如峰值是 `TFLOP/s`、带宽是 `GB/s` 时，需要把峰值换成 `GFLOP/s`，这样 ridge point 的单位才是 `FLOP/Byte`：

```text
Ridge = Peak_TFLOP/s × 1000 / Bandwidth_GB/s
```

## 3. Hello Roofline：方形 GEMM

对 `N × N` 矩阵乘法 `C = A × B`，使用最简单的一阶模型：

```text
FLOPs       = 2N³
Bytes moved = 3N² × bytes_per_element
AI          = 2N / (3 × bytes_per_element)
```

这里的 `3N²` 表示读取 `A`、读取 `B`、写回 `C` 各一次。真实实现可能因为缓存、分块、重复使用和融合而改变实际流量，所以这是教学用的基础估算。

精度对应的元素大小：

| Precision | Bytes per element |
|---|---:|
| FP32 | 4 |
| FP16 | 2 |
| INT8 | 1 |

由公式可以直接看出：在相同 `N` 下，元素越小，搬运的字节越少，AI 越高。但最终是否 compute-bound，还要同时看该精度在目标硬件上的峰值算力。

## 4. 可运行的最小实验

这个例子只使用 Python 标准库，不要求本地安装 `mlsysim`。示例硬件参数采用 A100 FP16 的教学近似值：312 TFLOP/s 和 1555 GB/s。它用于 Roofline 推理，不代表当前机器真的有 A100。

```python
HARDWARE = {
    "name": "A100 FP16 (teaching estimate)",
    "peak_tflops": 312.0,
    "bandwidth_gbs": 1555.0,
}


def gemm_roofline(dimension, bytes_per_element, hardware=HARDWARE):
    flops = 2 * dimension**3
    bytes_moved = 3 * dimension**2 * bytes_per_element
    arithmetic_intensity = flops / bytes_moved

    peak_gflops = hardware["peak_tflops"] * 1000
    ridge = peak_gflops / hardware["bandwidth_gbs"]
    attainable_gflops = min(
        peak_gflops,
        hardware["bandwidth_gbs"] * arithmetic_intensity,
    )
    regime = "Compute-bound" if arithmetic_intensity >= ridge else "Memory-bound"
    mfu_pct = attainable_gflops / peak_gflops * 100

    return {
        "N": dimension,
        "AI (FLOP/Byte)": arithmetic_intensity,
        "Ridge (FLOP/Byte)": ridge,
        "Attainable (TFLOP/s)": attainable_gflops / 1000,
        "MFU (%)": mfu_pct,
        "Regime": regime,
    }


for n in (256, 512, 1024):
    result = gemm_roofline(n, bytes_per_element=2)
    print(result)
```

预计得到近似结果：

| N | AI (FLOP/Byte) | 可达性能 | MFU | Regime |
|---:|---:|---:|---:|---|
| 256 | 85.3 | 132.7 TFLOP/s | 42.5% | Memory-bound |
| 512 | 170.7 | 265.4 TFLOP/s | 85.1% | Memory-bound |
| 1024 | 341.3 | 312.0 TFLOP/s | 100.0% | Compute-bound |

这个结果展示了 ridge crossing：矩阵规模增大后，数据被复用得更多，AI 上升，工作负载从内存受限跨到计算受限。

## 5. 用一句话解释结果

以 `N=256, FP16` 为例：

```text
AI = 85.3 FLOP/Byte
Ridge = 200.6 FLOP/Byte
85.3 < 200.6
```

因此带宽屋顶更低，估算结果是 `Memory-bound`。即使 A100 的理论算力很高，数据搬运速度仍然限制了这次 GEMM 能达到的吞吐。

以 `N=1024, FP16` 为例：

```text
AI = 341.3 FLOP/Byte
Ridge = 200.6 FLOP/Byte
341.3 > 200.6
```

此时带宽屋顶已经高于算力屋顶，结果被峰值算力限制，属于 `Compute-bound`。

## 6. 与 MLSys·im 代码的对应关系

官方辅助实现中的对应关系是：

- `gemm_workload()`：计算 `2N³`、`3N²×bytes_per_element` 和 AI；
- `hardware_roofline_profile()`：读取硬件峰值算力、内存带宽和显存容量，并计算 ridge point；
- `roofline_point()`：执行 `min(peak, bandwidth×AI)`，并返回 MFU 和瓶颈类型；
- `fusion_traffic()`：估算 kernel fusion 减少内存读写后的搬运时间。

因此，Task 0 里的 `Engine.solve()` 是面向模型的整体分析；Hello Roofline 则先把其中最关键的性能判断拆出来，单独理解数学关系。

## 7. 这一步不应该误解成什么

1. `Compute-bound` 不等于真实程序一定能达到峰值。内核调度、形状对齐、Tensor Core 支持、缓存和利用率都会降低实际性能。
2. `Memory-bound` 不等于只需要换更大显存。Roofline 这里关注的是带宽，容量是另一个可行性约束。
3. `MFU=100%` 是这个一阶模型相对理论峰值的结果，不是实际监控得到的 GPU 利用率。
4. 降低精度通常会减少数据量、提高 AI，但硬件是否支持该格式、峰值算力是否变化、数值质量是否允许，都需要单独检查。
5. Roofline 结果是估算模型，不是实际 GPU benchmark。真实部署还需要用目标硬件和真实内核测量。

## 8. Task 1 学习检查清单

```text
[x] 能说出 AI = FLOPs / Bytes moved
[x] 能说出 Ridge = Peak FLOP/s / Bandwidth
[x] 能用 AI 与 Ridge 判断 Memory-bound 或 Compute-bound
[x] 能解释为什么增大 GEMM 规模会提高 AI
[x] 能解释 precision 为什么会影响内存流量
[x] 能区分显存容量、显存带宽和 GPU 算力
[x] 能运行最小 Python Roofline 示例
[x] 在 Colab 安装 mlsysim 后，用官方模型和硬件对象完成实验
[x] 记录 ResNet-50、Llama-3 8B 与三种硬件的实际仿真输出
```

## 9. 原理练习小结

以下内容仅总结前面的方形 GEMM 原理练习；正式的 Datawhale Task 1 打卡内容见第 11 节。

```text
task1：Hello Roofline

学习内容：
- 理解 Arithmetic Intensity、Ridge Point 和 Roofline 上限
- 完成方形 GEMM 的 FP16 示例
- 观察 N=256、512、1024 的瓶颈变化

关键结果：
- A100 教学参数：312 TFLOP/s，1555 GB/s
- Ridge Point：约 200.6 FLOP/Byte
- N=256：Memory-bound
- N=512：Memory-bound，接近 ridge
- N=1024：Compute-bound

核心结论：
硬件选型不能只看峰值 FLOP/s；必须比较工作负载的算术强度和目标硬件的 ridge point。
```

## 10. Datawhale Task 1 综合报告

上面的 GEMM 是 Roofline 原理的最小手算示例。Datawhale Task 1 还特别要求说明 MLSys·im 仿真系统的使用方法、Roofline 模型原理，并通过修改 batch size、硬件或模型分析性能变化。本节使用 Colab 中重新运行的 `Engine.solve()` 输出完成报告。

### 10.1 MLSys·im 仿真系统如何使用

MLSys·im 是面向机器学习系统的分析与仿真工具。它不要求当前机器实际配备 A100、H100 或 Jetson AGX Orin，而是结合模型的计算量与数据移动量，以及目标硬件的峰值算力、内存带宽和显存容量，估算性能和可行性。

系统中与本实验直接相关的对象包括：

- `mlsysim.Models`：选择要分析的工作负载，例如 `ResNet50` 或 `Llama3_8B`；
- `mlsysim.Hardware`：选择目标硬件，例如 A100、H100 或 Jetson AGX Orin；
- `Engine.solve()`：组合模型、硬件、batch size 和精度，执行一次仿真分析；
- `PerformanceProfile`：保存瓶颈、延迟、吞吐量、算术强度、显存占用和可行性等结果。

在 Colab 中先安装并导入 MLSys·im：

```python
%pip install -q "git+https://github.com/harvard-edge/cs249r_book.git@dev#subdirectory=mlsysim"

import mlsysim
from mlsysim import Engine
```

一次最小分析的写法如下：

```python
profile = Engine.solve(
    model=mlsysim.Models.Vision.ResNet50,
    hardware=mlsysim.Hardware.Cloud.A100,
    batch_size=64,
    precision="fp16",
)

print("Bottleneck:", profile.bottleneck)
print("Latency:", profile.latency.to("ms"))
print("Throughput:", profile.throughput)
print("Arithmetic intensity:", profile.arithmetic_intensity)
print("Compute latency:", profile.latency_compute.to("ms"))
print("Memory latency:", profile.latency_memory.to("ms"))
print("Memory footprint:", profile.memory_footprint)
print("Feasible:", profile.feasible)
```

主要输出字段的含义是：

| 字段 | 含义 |
|---|---|
| `bottleneck` | 当前配置主要受计算能力还是内存带宽限制 |
| `latency` | 完成一个 batch 的估算时间 |
| `throughput` | 单位时间内可以处理的样本数 |
| `arithmetic_intensity` | 每搬运 1 Byte 数据完成的 FLOPs |
| `latency_compute` | 计算能力对应的延迟分量 |
| `latency_memory` | 数据搬运对应的延迟分量 |
| `memory_footprint` | 当前配置所需的显存容量 |
| `feasible` | 模型及当前配置能否放入目标硬件显存 |

本实验的使用流程可以概括为：

```text
选择模型 + 选择硬件 + 设置 batch size/precision
                         ↓
                    Engine.solve()
                         ↓
瓶颈、延迟、吞吐量、AI、显存占用与可行性
```

保持模型和硬件不变、修改 `batch_size`，可以研究吞吐量和瓶颈变化；保持模型不变、替换 `hardware`，可以比较硬件；保持硬件不变、替换 `model`，可以比较不同模型结构。MLSys·im 给出的是基于模型与硬件规格的仿真估算，并非真实硬件 benchmark。

### 10.2 Roofline 模型原理

Roofline 模型同时考虑计算能力上限和内存带宽上限。三个核心关系为：

```text
Arithmetic Intensity (AI) = FLOPs / Bytes moved
Memory ceiling            = Memory Bandwidth × AI
Attainable Performance    = min(Peak FLOP/s, Memory Bandwidth × AI)
```

AI 表示每搬运一个字节的数据能够完成多少次浮点运算。AI 低，说明数据搬运多、计算复用少；AI 高，说明数据能够被重复利用并完成更多计算。

硬件的临界点称为 ridge point：

```text
Ridge Point = Peak FLOP/s / Memory Bandwidth
```

它是 Roofline 图中斜线段与水平线段的交点，也是判断瓶颈的依据：

```text
AI < Ridge Point  -> Memory-bound
AI >= Ridge Point -> Compute-bound
```

当 AI 小于 ridge point 时，带宽屋顶低于计算屋顶，性能主要受数据搬运限制。优化方向通常是减少内存访问、提高缓存或权重复用、提高带宽。AI 大于或等于 ridge point 时，可达性能主要受计算屋顶限制，优化方向通常是提高计算效率、选择更合适的精度或使用算力更强的硬件。

在 MLSys·im 输出中还可以用延迟分量辅助验证：`latency_memory` 占主导时通常为 Memory-bound，`latency_compute` 占主导时通常为 Compute-bound。`Engine.solve()` 已根据模型工作量和硬件参数完成估算，并直接返回 `bottleneck`。

Batch Size 会改变工作负载在 Roofline 图上的位置。batch 增大时，同一份模型权重可以服务更多输入，权重搬运成本被更多样本分摊，因此 AI 通常上升，工作负载向 Roofline 图右侧移动。如果低 batch 的 AI 小于 ridge point，增大 batch 可能使瓶颈从 Memory-bound 转为 Compute-bound；如果 batch 1 已经位于 ridge point 右侧，则不会发生瓶颈转换，但吞吐量接近计算屋顶后会逐渐饱和。

同一个模型、精度和 batch size 在不同硬件上的 AI 基本相同，但每种硬件的峰值算力、内存带宽及 ridge point 不同，所以同一工作负载在不同硬件上可能得到不同的瓶颈判断。

### 10.3 ResNet-50 + A100：预测与验证

实验配置：ResNet-50、A100、FP16、`batch_size=64`。运行前预测为 `compute-bound`，`Engine.solve()` 验证正确：

| 指标 | 结果 |
|---|---:|
| Bottleneck | Compute |
| Latency | 3.879 ms |
| Throughput | 16,498.66 images/s |
| Arithmetic intensity | 1,385.14 flop/B |
| Compute latency | 3.364 ms |
| Memory latency | 0.186 ms |
| MFU | 43.36% |

计算延迟显著大于内存延迟，因此该配置由计算吞吐主导。要在运行前可靠预测，需要知道工作负载 FLOPs、数据移动量、精度、硬件有效算力和内存带宽。

### 10.4 中级目标（a）：A100 的 Batch Size sweep

| Batch | Bottleneck | Latency (ms) | Throughput (1/s) | AI (flop/B) |
|---:|---|---:|---:|---:|
| 1 | Compute | 0.568 | 1,761.92 | 145.60 |
| 2 | Compute | 0.620 | 3,225.14 | 266.93 |
| 4 | Compute | 0.725 | 5,515.29 | 457.59 |
| 8 | Compute | 0.936 | 8,551.46 | 711.81 |
| 16 | Compute | 1.356 | 11,799.19 | 985.58 |
| 32 | Compute | 2.197 | 14,564.98 | 1,220.24 |
| 64 | Compute | 3.879 | 16,498.66 | 1,385.14 |
| 128 | Compute | 7.243 | 17,671.73 | 1,485.51 |
| 256 | Compute | 13.971 | 18,323.13 | 1,541.35 |

A100 在测试范围内从 batch 1 开始就已经是 `Compute-bound`，因此 batch 1--256 之间没有观测到 Memory 到 Compute 的 crossover。batch 1 的 AI 已为 145.60 FLOP/Byte，随后随 batch 增大升至 1,541.35 FLOP/Byte，所有测试点均位于 A100 的计算受限一侧。

Batch Size 增大显著提高了吞吐量，但提升并不与 batch 成正比。吞吐量从 batch 1 的 1,761.92/s 增加到 batch 32 的 14,564.98/s，约提升 8.27 倍；batch 随后从 32 增加到 256，扩大 8 倍，吞吐量却只进一步提高到 18,323.13/s，约提升 25.8%。与此同时，一个 batch 的延迟由 0.568 ms 上升到 13.971 ms。

因此，增大 batch 可以通过权重和数据复用提高 AI 与吞吐量，但计算资源成为主导瓶颈后，性能逐渐接近有效计算屋顶，继续扩大 batch 的边际收益下降。A100 上可观察到的实用吞吐量拐点大约在 batch 32--64，而不是一次瓶颈类型转换。

### 10.5 中级目标（b）：替换为 H100

| Batch | Bottleneck | Latency (ms) | Throughput (1/s) | AI (flop/B) |
|---:|---|---:|---:|---:|
| 1 | Memory | 0.527 | 1,898.21 | 145.60 |
| 2 | Compute | 0.543 | 3,682.12 | 266.93 |
| 4 | Compute | 0.576 | 6,940.47 | 457.59 |
| 8 | Compute | 0.643 | 12,448.28 | 711.81 |
| 16 | Compute | 0.775 | 20,636.68 | 985.58 |
| 32 | Compute | 1.041 | 30,750.40 | 1,220.24 |
| 64 | Compute | 1.571 | 40,731.28 | 1,385.14 |
| 128 | Compute | 2.633 | 48,622.09 | 1,485.51 |
| 256 | Compute | 4.755 | 53,836.98 | 1,541.35 |

H100 的 crossover 位于 batch 1 和 2 之间：batch 1 的 AI 为 145.60 FLOP/Byte，`Engine.solve()` 判断为 Memory-bound；batch 2 的 AI 上升至 266.93 FLOP/Byte，瓶颈转为 Compute-bound。

虽然 H100 的计算和带宽都强于 A100，但其计算峰值相对内存带宽提升得更快，因此算力/带宽比和 ridge point 更高。同一个 ResNet-50、FP16、batch 1 工作负载在 A100 上已经 Compute-bound，在 H100 上却仍略受内存带宽限制。这说明“硬件更强”不意味着任何工作负载都更容易计算受限；瓶颈取决于工作负载 AI 与该硬件 ridge point 的相对位置。

H100 的吞吐量从 batch 1 的 1,898.21/s 增至 batch 256 的 53,836.98/s。batch 128 到 256 虽然翻倍，吞吐量只从 48,622.09/s 增至 53,836.98/s，进一步说明进入计算受限区间后吞吐量的边际增长逐渐降低。

### 10.6 中级目标（c，补充）：Llama-3 8B + A100

配置为 Llama-3 8B、A100、FP16、batch 1，结果为 `Memory-bound` 且可行：

| 指标 | 结果 |
|---|---:|
| Latency | 8.999 ms |
| Throughput | 111.12 /s |
| Arithmetic intensity | 0.91 flop/B |
| Compute latency | 0.103 ms |
| Memory latency | 8.664 ms |
| Memory footprint | 16.06 GB |
| Feasible | True |

低 batch 自回归 LLM 每步需要读取大量权重，但可摊销的计算有限，所以内存带宽成为主导；这与数据复用更高的 ResNet-50 不同。

### 10.7 高级目标：三种硬件对比

对 ResNet-50、FP16，在 A100、H100、Jetson AGX Orin 上比较 batch 1、32、256：

| Hardware | Batch | Bottleneck | Latency (ms) | Throughput (1/s) |
|---|---:|---|---:|---:|
| A100 | 1 | Compute | 0.568 | 1,761.92 |
| A100 | 32 | Compute | 2.197 | 14,564.98 |
| A100 | 256 | Compute | 13.971 | 18,323.13 |
| H100 | 1 | Memory | 0.527 | 1,898.21 |
| H100 | 32 | Compute | 1.041 | 30,750.40 |
| H100 | 256 | Compute | 4.755 | 53,836.98 |
| Jetson AGX Orin | 1 | Memory | 0.976 | 1,024.51 |
| Jetson AGX Orin | 32 | Compute | 2.608 | 12,268.23 |
| Jetson AGX Orin | 256 | Compute | 15.967 | 16,033.16 |

在相同 batch 下，三种硬件对应的 AI 相同：batch 1、32、256 分别为 145.60、1,220.24 和 1,541.35 FLOP/Byte。这是因为 AI 主要由模型、精度和 batch size 决定，而不是由硬件决定。但三种硬件具有不同的计算能力、内存带宽和 ridge point，所以瓶颈判断不同。

A100 在三个 batch 下均为 Compute-bound。H100 和 Jetson AGX Orin 在 batch 1 时为 Memory-bound，在 batch 32 和 256 时转为 Compute-bound。原因是低 batch 能够复用权重的机会有限，AI 较低，数据搬运先成为瓶颈；batch 增大后，同一份权重服务更多样本，AI 提高并跨过相应硬件的 ridge point，瓶颈因而转为计算。

进入 Compute-bound 后，硬件计算能力差异更加明显。在 batch 256 下，H100 的延迟为 4.755 ms，明显低于 A100 的 13.971 ms 和 Jetson AGX Orin 的 15.967 ms；对应吞吐量分别为 53,836.98/s、18,323.13/s 和 16,033.16/s。另一方面，batch 增大虽然提高吞吐量，却同时增加完成一个 batch 的延迟，因此实际部署还需要在吞吐量和延迟目标之间权衡。

综合来看，不能只按 GPU 峰值 FLOP/s 判断性能。完整判断需要同时考虑工作负载 AI、硬件 ridge point、Batch Size、有效计算能力和内存带宽。以上结果均为 MLSys·im 的 Roofline 仿真估算，不是实际 A100、H100 或 Jetson AGX Orin 上的 benchmark。

## 11. Task 1 完成度与可直接打卡内容

对照 issue #76：

- 中级目标（a）：已完成，说明了 ResNet-50 在 A100 上的瓶颈，并分析了 batch 1--256 的吞吐量变化；
- 中级目标（b）：已完成，增加了 H100；
- 中级目标（c）：已完成，增加了 Llama-3 8B；
- 高级目标：已完成，比较了 A100、H100、Jetson AGX Orin 三种硬件在 batch 1、32、256 的延迟和瓶颈。

因此不只是满足最低打卡，而是完成了 issue 中的高级目标。可直接提交：

```text
Task 1：性能分析入门 / Hello Roofline

一、仿真系统的使用方法

我在 Google Colab 中安装并导入 mlsysim，从 mlsysim.Models 选择 ResNet-50 或 Llama-3 8B，从 mlsysim.Hardware 选择 A100、H100 或 Jetson AGX Orin，然后通过 Engine.solve(model=..., hardware=..., batch_size=..., precision="fp16") 运行仿真。返回的 PerformanceProfile 提供 bottleneck、latency、throughput、arithmetic_intensity、latency_compute、latency_memory、memory_footprint 和 feasible 等结果。改变 batch_size 可以观察吞吐量和瓶颈变化；保持模型不变并替换 hardware 可以比较硬件；保持硬件不变并替换 model 可以比较模型。

二、Roofline 模型原理

Roofline 同时考虑计算屋顶和内存带宽屋顶：AI = FLOPs / Bytes moved；带宽屋顶 = Memory Bandwidth × AI；可达性能 = min(Peak FLOP/s, Memory Bandwidth × AI)。硬件的 Ridge Point = Peak FLOP/s / Memory Bandwidth。当 AI 小于 ridge point 时，任务为 Memory-bound；当 AI 大于或等于 ridge point 时，任务为 Compute-bound。batch 增大后，同一份权重可被更多样本复用，AI 通常升高，工作负载在 Roofline 图上向右移动，并可能从内存受限转为计算受限。

三、实验结果与分析

我通过 Engine.solve() 完成了 ResNet-50、Llama-3 8B 与 A100、H100、Jetson AGX Orin 的 Roofline 分析。

ResNet-50 + A100、FP16：batch size 64 的预测为 Compute-bound，Engine.solve() 验证正确（Latency 3.879 ms，Throughput 16,498.66/s，AI 1,385.14 flop/B，Compute latency 3.364 ms，Memory latency 0.186 ms）。A100 上 batch 1--256 始终为 Compute-bound，吞吐量从 1,761.92/s 增至 18,323.13/s，但 batch 32--64 后增长趋缓。

ResNet-50 + H100：batch 1 为 Memory-bound，batch 2 起为 Compute-bound，crossover 位于 1 和 2 之间。H100 的算力相对带宽提升更大，因此 ridge point 更高，低 batch 的低 AI 工作负载先受内存带宽限制。

Llama-3 8B + A100、batch 1：Memory-bound，Latency 8.999 ms，AI 0.91 flop/B，Memory latency 8.664 ms，显存占用约 16.06 GB，Feasible=True。低 batch 自回归推理需要频繁读取大量权重，内存带宽成为瓶颈。

高级目标对比：A100 在 batch 1、32、256 均为 Compute-bound；H100 和 Jetson AGX Orin 在 batch 1 为 Memory-bound、batch 32 和 256 为 Compute-bound。batch 增大提高数据复用和 AI，跨过各硬件 ridge point 后转为计算受限；进入 Compute-bound 后吞吐量的边际增长降低。

以上是 mlsysim 的 Roofline 仿真估算，不等同于真实 A100、H100 或 Jetson 上的 benchmark。
```

### 11.1 打卡配图顺序

建议按以下顺序把本次 Colab 截图插入 GitHub 评论：

1. MLSys·im 安装、导入以及一次 `Engine.solve()` 代码和原始输出，证明系统的实际使用方法；
2. A100 的 batch sweep 数据表、吞吐量曲线和 A100 Roofline 图，对应中级目标（a）；
3. H100 的 batch sweep 数据表和 H100 Roofline 图，对应中级目标（b）；
4. A100、H100、Jetson AGX Orin 在 batch 1、32、256 下的对比表，对应高级目标；
5. Llama-3 8B + A100 的输出可作为中级目标（c）的额外补充。

Roofline 图中应明确区分理论屋顶和仿真数据点。若纵坐标数据由 `Engine.solve()` 的吞吐量换算，应标注 `Engine-derived performance = throughput × FLOPs/image`；如果点只是按 `min(compute ceiling, bandwidth × AI)` 计算，则只能称为理论 Roofline 位置，不能写成实测或 Engine 实际性能。
