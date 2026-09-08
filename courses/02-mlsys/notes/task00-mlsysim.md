# Task 0：MLSys·im 学习笔记

## 1. Task 0 的目标

Task 0 是学习本课程的最小前置，核心目标不是训练模型，而是了解机器学习系统分析工具 MLSys·im，并完成一次基础分析。

官方要求包括：

1. 阅读 MLSys·im 介绍；
2. 阅读 MLSys·im 的设计理念；
3. 按 Getting Started 完成环境配置；
4. 提交环境配置成功的截图。

官方入口：

- [MLSys·im 介绍](https://mlsysbook.ai/mlsysim/)
- [MLSys·im Philosophy](https://mlsysbook.ai/mlsysim/philosophy.html)
- [Getting Started](https://mlsysbook.ai/mlsysim/getting-started.html)
- [Datawhale Task 0 Issue](https://github.com/datawhalechina/llm-algo-leetcode/issues/65)

Task 0 不要求真实 GPU、A100/H100、模型训练、完整多卡实验或 Level 3 经济性分析。

## 2. Colab 环境配置

如果当前 Colab 运行时还没有安装 `mlsysim`，运行：

```python
%cd /content

!if [ ! -d /content/cs249r_book ]; then \
  git clone --depth 1 --branch dev \
  https://github.com/harvard-edge/cs249r_book.git \
  /content/cs249r_book; \
fi

%cd /content/cs249r_book/mlsysim
%pip install -q -e .
```

检查版本：

```python
import sys
import mlsysim

print(f"Python 版本：{sys.version.split()[0]}")
print(f"mlsysim 版本：{mlsysim.__version__}")
```

之前实际安装日志显示：

```text
Python 3.12 系列
mlsysim 0.1.2
```

## 3. 第一次 Roofline 分析

官方示例使用 ResNet50 和 A100：

```python
import mlsysim
from mlsysim import Engine

model = mlsysim.Models.Vision.ResNet50
hardware = mlsysim.Hardware.Cloud.A100

profile = Engine.solve(
    model=model,
    hardware=hardware,
    batch_size=1,
    precision="fp16"
)

print(f"Bottleneck: {profile.bottleneck}")
print(f"Latency: {profile.latency.to('ms'):~.2f}")
print(f"Throughput: {profile.throughput:.0f}")
```

官方示例结果约为：

```text
Bottleneck: Memory
Latency: 0.54 ms
Throughput: 1843 / second
```

这里的 Roofline 分析用于判断工作负载受哪种资源限制：

- `Memory`：受显存带宽或数据搬运限制；
- `Compute`：受 GPU 算力限制。

## 4. `mlsysim eval` 结果

之前在 Colab 中运行过：

```bash
mlsysim eval Llama3_8B H100 --batch-size 32
```

典型输出为：

```text
Scenario: Llama3_8B on H100
Level 1: Feasibility [PASS]
↳ 16.1 GB / 85.9 GB used

Level 2: Performance [PASS]
↳ Memory Bound

Level 3: Macro/Economics [SKIPPED]
↳ No Ops config provided
```

含义：

- Level 1：模型和运行所需内存能够放进目标硬件；
- Level 2：工具估算性能，并判断计算瓶颈或内存瓶颈；
- Level 3：需要额外的 Ops、地区、运行时间和集群配置，用于成本、能耗和碳排放分析。

`SKIPPED` 不是安装失败。`H100` 也是模拟分析目标，不代表 Colab 实际分配了 H100。

## 5. 输出字段

可以查看完整的 `PerformanceProfile`：

```python
fields = [
    "bottleneck",
    "latency",
    "throughput",
    "latency_compute",
    "latency_memory",
    "arithmetic_intensity",
    "energy",
    "memory_footprint",
    "mfu",
    "feasible",
]

for field in fields:
    print(f"{field}: {getattr(profile, field, 'N/A')}")
```

关键字段：

- `latency`：处理一个 batch 的估算时间；
- `throughput`：吞吐量，通常是 `batch_size / latency`；
- `latency_compute`：只考虑计算能力时的延迟；
- `latency_memory`：只考虑显存带宽时的延迟；
- `arithmetic_intensity`：每搬运一个字节需要执行多少运算；
- `memory_footprint`：工作负载需要的总内存；
- `mfu`：Model FLOPs Utilization 的估算值；
- `feasible`：模型是否能够放入目标设备内存。

Roofline 的简化关系是：

```text
计算耗时 = FLOPs / 有效计算能力
内存耗时 = 数据量 / 显存带宽
总耗时   = max(计算耗时, 内存耗时)
```

## 6. MLSys·im 的四类核心对象

`Exploring the Zoo` 中最重要的四类对象是：

### Models

表示要分析的模型或工作负载，例如：

```python
mlsysim.Models.Language.Llama3_8B
mlsysim.Models.Language.Llama3_70B
mlsysim.Models.Vision.ResNet50
```

### Hardware

表示运行模型的硬件，例如：

```python
mlsysim.Hardware.Cloud.A100
mlsysim.Hardware.Cloud.H100
mlsysim.Hardware.Cloud.H200
```

### Infrastructure

表示数据中心和能源环境，例如不同地区的电网、碳强度和能源条件：

```python
mlsysim.Infrastructure.Grids.Quebec
mlsysim.Infrastructure.Grids.US_Avg
mlsysim.Infrastructure.Grids.Poland
```

### Systems

表示多卡、集群和互联网络，例如：

```python
mlsysim.Systems.Fabrics.InfiniBand_NDR
mlsysim.Systems.Fabrics.Ethernet_100G
mlsysim.Systems.Clusters.Frontier_8K
```

四者的关系可以概括为：

```text
Model + Hardware
    -> 单卡显存、延迟、吞吐和瓶颈

+ Systems
    -> 多卡通信、集群扩展和分布式性能

+ Infrastructure
    -> 能耗、碳排放、运行成本和经济性
```

## 7. Efficiency 参数

`efficiency` 用希腊字母 `η` 表示，代表实际达到理论峰值算力的比例。

默认示例：

```python
profile = Engine.solve(
    model=model,
    hardware=hardware,
    batch_size=32,
    precision="fp16",
    efficiency=0.5
)
```

计算能力近似为：

```text
有效计算能力 = 理论峰值算力 × efficiency
```

常见估计范围：

- 优化良好的 FP16 训练：0.35–0.55；
- FP16 推理：0.25–0.45；
- INT8 推理：0.20–0.40。

它不是 GPU 实际利用率，也不是实时监控数据。如果任务是 `Memory Bound`，修改 Efficiency 可能不会明显改变结果。

## 8. Serving 分析

对于 LLM 服务，可以运行：

```bash
mlsysim serve Llama3_70B B200 --batch-size 32
```

常见字段：

- `TTFT`：Time to First Token，首 token 延迟；
- `ITL`：Inter-Token Latency，生成 token 之间的间隔；
- `Tokens/s`：生成吞吐；
- `KV Cache`：推理缓存的 Key/Value 所占内存；
- `Memory Usage`：显存使用比例；
- `Efficiency`：计算效率假设。

## 9. 自定义模型

MLSys·im 也可以分析不在内置 Zoo 中的模型：

```python
from mlsysim import ureg
from mlsysim.models.types import TransformerWorkload

custom_model = TransformerWorkload(
    name="My-Custom-LLM",
    architecture="Transformer",
    parameters=13e9 * ureg.param,
    layers=40,
    hidden_dim=5120,
    heads=40,
    kv_heads=8,
    inference_flops=2 * 13e9 * ureg.flop
)

profile = Engine.solve(
    model=custom_model,
    hardware=mlsysim.Hardware.Cloud.A100,
    batch_size=1
)

print(f"Bottleneck: {profile.bottleneck}")
print(f"Latency: {profile.latency}")
print(f"Memory: {profile.memory_footprint}")
print(f"Feasible: {profile.feasible}")
```

## 10. T 级参数模型的可行性分析

如果模型有 1T 参数，FP16 权重仅需要：

```text
1T × 2 bytes ≈ 2 TB
```

因此在单张 H100 上会得到：

```text
Feasible: False
Bottleneck: Memory
Memory: 2000000000000 B
```

这表示单张 H100 装不下模型，不是代码错误。

粗略估计：

- 80 GB H100：理论上至少约 25 张；
- 86 GB H100：理论上至少约 24 张；
- 192 GB B200：理论上至少约 11 张。

实际还要额外考虑激活值、KV Cache、通信、框架开销和冗余，所以实际卡数会更多。

对于 MoE 模型，要区分：

- 总参数量：主要决定权重存储需求；
- 激活参数量：主要决定每个 token 的计算量。

因此不能只因为每个 token 只激活几十 B 参数，就认为整个 1T 模型只需要存几十 B 参数的显存。

## 11. Task 0 打卡模板

```text
task0 打卡

环境配置：
- Python 版本：3.12
- mlsysim 版本：0.1.2
- 已成功运行一次 mlsysim 分析

运行命令：
mlsysim eval Llama3_8B H100 --batch-size 32

运行结果：
- Level 1: Feasibility [PASS]
- Level 2: Performance [PASS]
- Level 3: Macro/Economics [SKIPPED]

补充：已完成官方 Getting Started 示例，并理解 Memory Bound、Throughput、Efficiency 和 Feasible 等输出字段。

附：环境配置及运行结果截图。
```

## 12. 总结

Task 0 学到的核心不是“如何真正运行一个大模型”，而是建立机器学习系统的估算思路：

```text
模型需要多少计算？
模型需要多少显存？
硬件能否装下？
瓶颈在计算还是数据搬运？
batch、精度和效率如何影响结果？
多卡、集群和能源环境会带来什么成本？
```

MLSys·im 的结果是分析和模拟结果，不等同于真实硬件 benchmark。真正部署时，还需要用实际 GPU、实际模型权重和真实服务流量进行测量验证。
