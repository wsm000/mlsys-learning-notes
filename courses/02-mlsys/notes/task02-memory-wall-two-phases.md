# Task 2：算力墙与内存墙——LLM 推理的性能诊断

## 1. 任务定位

本笔记对应 Datawhale [Task 2 issue #83](https://github.com/datawhalechina/llm-algo-leetcode/issues/83)，实验记录来自我在该 issue 下的两条回复：

- [E1-E4 回复](https://github.com/datawhalechina/llm-algo-leetcode/issues/83#issuecomment-5313281650)
- [E5-E6、O1-O4 回复](https://github.com/datawhalechina/llm-algo-leetcode/issues/83#issuecomment-5313568264)

任务目标是理解 LLM 推理的两个主要性能阶段：

1. Memory Wall：为什么 H100 的峰值算力比 A100 高约 3.2 倍，但 Llama3-8B 的低 batch 推理只获得约 1.6 倍加速；
2. Two Phases：为什么同一个模型的 Prefill 和 Decode 会分别撞上计算墙和内存墙；
3. 性能指标：理解 TTFT（Time to First Token）与 ITL（Inter-Token Latency）；
4. 工程决策：根据模型、上下文、batch 和 SLA 选择硬件或并行方式。

理论教程：

- [The Memory Wall](https://mlsysbook.ai/mlsysim/tutorials/01_memory_wall.html)
- [Two Phases, One Request](https://mlsysbook.ai/mlsysim/tutorials/02_two_phases.html)

本实验使用 MLSys·im 的分析模型，不代表当前 Colab 运行时真的拥有 A100、H100 或 H200。

## 2. 环境与求解器

在 Colab 中安装教程对应的开发版本：

```python
%pip install -q "git+https://github.com/harvard-edge/cs249r_book.git@dev#subdirectory=mlsysim" pandas matplotlib
```

导入模型、硬件和两个求解器：

```python
import mlsysim
from mlsysim.solvers import SingleNodeModel, ServingModel

model = mlsysim.Models.Language.Llama3_8B
a100 = mlsysim.Hardware.Cloud.A100
h100 = mlsysim.Hardware.Cloud.H100
h200 = mlsysim.Hardware.Cloud.H200
```

### 2.1 SingleNodeModel

`SingleNodeModel` 分析一次前向或 decode step 的 Roofline 瓶颈：

```python
single = SingleNodeModel()
result = single.solve(
    model=model,
    hardware=h100,
    batch_size=1,
    precision="fp16",
    efficiency=0.5,
)
```

它返回 `bottleneck`、`latency`、`throughput`、`arithmetic_intensity` 等字段，适合 E1、E2 和 O1。

### 2.2 ServingModel

`ServingModel` 把一次 LLM 请求拆成两个阶段：

- Prefill：并行处理整个输入 prompt，对应 TTFT；
- Decode：每次生成一个新 token，对应 ITL。

```python
serving = ServingModel()
result = serving.solve(
    model=model,
    hardware=h100,
    seq_len=2048,
    batch_size=1,
    precision="fp16",
)
```

它返回 `ttft`、`itl`、`model_weights_size`、`kv_cache_size`、`memory_utilization` 和 `feasible` 等字段，适合 E3、E4、E5、E6 和 O2-O4。

## 3. Roofline 核心概念

### 3.1 Arithmetic Intensity

算术强度表示每搬运一个字节的数据完成多少计算：

```text
Arithmetic Intensity = FLOPs / Bytes moved
```

### 3.2 Ridge Point

硬件的理想脊点是计算屋顶和带宽屋顶的交点：

```text
Ideal Ridge = Peak FLOP/s / HBM Bandwidth
```

如果实验设置了 `efficiency=0.5`，而 MLSys·im 只折损有效计算能力、不折损带宽，则模拟器实际使用的有效脊点为：

```text
Effective Ridge = Peak FLOP/s × efficiency / HBM Bandwidth
```

这里要区分两件事：

- Ideal Ridge 是硬件规格决定的物理分析值；
- Effective Ridge 是考虑效率假设后的模拟器判界。

### 3.3 瓶颈判断

```text
AI < Ridge Point   -> Memory-bound
AI >= Ridge Point  -> Compute-bound
```

Memory-bound 的主要优化方向是提高 HBM 带宽、减少数据搬运、压缩权重和提高权重复用；Compute-bound 的主要优化方向是提高有效算力、使用更高效的计算内核和提高低精度计算吞吐。

## 4. E1：A100 vs H100，batch=1

实验配置：Llama3-8B、FP16、`batch_size=1`。

| GPU | Bottleneck | Latency | Throughput | AI | Ideal Ridge | Effective Ridge |
|---|---|---:|---:|---:|---:|---:|
| A100 | Memory | 8.999 ms | 111.12/s | 0.909 | 153.02 | 76.51 |
| H100 | Memory | 5.603 ms | 178.46/s | 0.909 | 295.22 | 147.61 |

硬件规格比值：

```text
H100/A100 peak compute ratio  = 989 / 312  ≈ 3.17×
H100/A100 bandwidth ratio      = 3.35 / 2.04 ≈ 1.64×
Observed latency speedup       = 8.999 / 5.603 ≈ 1.61×
```

两个 GPU 都是 Memory-bound，因此延迟主要由权重从 HBM 搬运的速度决定，而不是由峰值 FLOP/s 决定。H100 多出来的计算能力没有成为当前工作负载的限制因素。

这回答了任务中的核心问题：

> H100 算力提升 3.2 倍，但低 batch LLM 推理只提升约 1.6 倍，是因为实际加速被 HBM 带宽比限制，而不是被峰值算力比限制。

## 5. E2：batch=1 到 512 的 Ridge/交叉分析

实验结果：A100 和 H100 从 batch 1 扫到 batch 512，都没有出现 Memory-bound 到 Compute-bound 的转换。

这里需要纠正一个容易混淆的表述：

- A100 和 H100 的 Ridge Point 并不是“没有找到”；它们可以直接从硬件规格计算出来；
- 没有找到的是 `critical batch size`，即模型工作点首次跨过 Ridge Point 的 batch。

当前 MLSys·im 对 Llama3-8B decode 的简化流量模型可以写成：

```text
AI(B) = B / (1 + 0.1B)
lim(B -> infinity) AI(B) = 10 FLOP/Byte
```

而四张卡的有效脊点约为：

```text
V100  69.4
A100  76.5
H100 147.6
B200 140.6 FLOP/Byte
```

模型的 AI 上限约为 10，始终低于这些有效脊点，因此即使增大 batch，也只能提高 Arithmetic Intensity，不能让该 Llama3-8B decode 工作负载进入 Compute-bound。

batch size 改变的是工作负载位置，不改变硬件脊点：

```text
batch 增大
    -> 权重搬运成本被更多样本摊薄
    -> Arithmetic Intensity 上升
    -> 工作点向 Roofline 右侧移动
```

但是否能够越过脊点，还取决于 AI 的上限。

## 6. E3：H100 上的 Prefill 与 Decode

实验配置：H100、Llama3-8B、`seq_len=2048`、`batch_size=1`、FP16。

| Phase | 指标 | 典型结果 | 瓶颈 |
|---|---|---:|---|
| Prefill | TTFT | 70.97 ms | Compute-bound |
| Decode | ITL | 5.194 ms/token | Memory-bound |
| Memory | Weights / KV cache | 16.06 / 0.268 GB | 权重占主导 |

### 6.1 为什么 Prefill 是 Compute-bound

Prefill 一次性并行处理 2048 个 prompt token。权重加载一次后，可以被多个 token 的矩阵计算复用，因此算术强度很高，约为几千 FLOP/Byte，远高于 H100 的脊点。

所以 Prefill 主要撞上计算屋顶，TTFT 对峰值算力更敏感。

### 6.2 为什么 Decode 是 Memory-bound

Decode 每一步只处理一个新 token，但需要重复读取大量模型权重。每个 token 的计算量相对较小，算术强度约为 1 FLOP/Byte，远低于 H100 脊点。

所以 Decode 主要撞上内存屋顶，ITL 对 HBM 带宽更敏感。

## 7. E4：A100 -> H100 -> H200 的非对称性

| GPU | TTFT | ITL |
|---|---:|---:|
| A100 | 224.95 ms | 8.328 ms/token |
| H100 | 70.97 ms | 5.194 ms/token |
| H200 | 70.97 ms | 3.722 ms/token |

升级路径的解释：

- A100 -> H100：算力提升约 3.17 倍，TTFT 约提升 3.17 倍；带宽提升约 1.64 倍，ITL 约提升 1.60 倍；
- H100 -> H200：峰值算力基本不变，TTFT 基本不变；带宽从 3.35 TB/s 增至 4.80 TB/s，ITL 提升约 1.40 倍。

结论是：

```text
TTFT 主要是计算墙问题，随 FLOP/s 变化；
ITL 主要是内存墙问题，随 HBM 带宽变化。
```

同一个模型、同一张 GPU、同一个请求，在 Prefill 和 Decode 两个阶段会先后受到不同资源的限制。

## 8. E5：batch size 对两个阶段的影响

实验配置：H100、Llama3-8B、FP16、`seq_len=2048`，扫描 `batch_size={1,4,16,64}`。

| Batch | TTFT | ITL | KV cache | Prefill AI | Decode AI |
|---:|---:|---:|---:|---:|---:|
| 1 | 70.97 ms | 5.194 ms/token | 0.268 GB | 约 2149 | 0.984 |
| 4 | 283.85 ms | 5.435 ms/token | 1.074 GB | 约 8192 | 3.749 |
| 16 | 1135.38 ms | 6.396 ms/token | 4.295 GB | 约 27582 | 12.624 |
| 64 | 4541.47 ms | 10.242 ms/token | 17.180 GB | 约 67562 | 30.922 |

观察结果：

1. Prefill 的 AI 很高，一开始就是 Compute-bound；增大 batch 只是让它更深入计算受限区；
2. Decode 的 AI 随 batch 增大，但在测试范围内仍低于 H100 有效脊点 147.6，因此仍是 Memory-bound；
3. batch 增大提高聚合吞吐，但不一定降低单用户 TTFT 或 ITL；
4. KV Cache 随 batch 线性增长，显存容量会限制可以使用的 batch 上限。

因此，batching 更准确的作用是“提高吞吐、摊薄权重搬运成本”，而不是保证单请求延迟下降或一定发生瓶颈转换。

## 9. E6：Phase Crossover

固定 `batch_size=1`，扫描 `seq_len=128..32768`，定义 crossover 条件：

```text
TTFT(seq_len) > 256 × ITL(seq_len)
```

典型结果：

```text
H100 / Llama3-8B / batch=1
First crossover seq_len ≈ 26,110 tokens
```

短上下文时，TTFT 较小，而 256 个输出 token 的 Decode 总时间主要由重复读取模型权重决定，因此 Decode 主导总延迟。

上下文继续变长后：

- Prefill 的工作量随 prompt 增长；
- Attention 计算带来近似二次增长项；
- Decode 的权重读取仍然有较大的固定成本，随上下文的增长相对缓慢。

于是总延迟会从 Decode 主导逐渐转为 Prefill 主导。

这不是说 Prefill 从 Memory-bound 变成 Compute-bound，或者 Decode 从 Compute-bound 变成 Memory-bound；改变的是两个阶段在端到端请求中的时间占比。

## 10. O1：V100 -> A100 -> H100 -> B200

| GPU | Peak FP16 | HBM BW | Ideal Ridge | Effective Ridge |
|---|---:|---:|---:|---:|
| V100 | 125 TFLOP/s | 0.90 TB/s | 138.9 | 69.4 |
| A100 | 312 TFLOP/s | 2.04 TB/s | 153.0 | 76.5 |
| H100 | 989 TFLOP/s | 3.35 TB/s | 295.2 | 147.6 |
| B200 | 2250 TFLOP/s | 8.00 TB/s | 281.2 | 140.6 |

不同代 GPU 的临界点不同，是因为峰值算力和 HBM 带宽的代际增长并不一致：

- A100 -> H100：算力约提升 3.17 倍，带宽约提升 1.64 倍，计算/带宽比显著变大，Ridge Point 从 153.0 升到 295.2；
- H100 -> B200：算力约提升 2.28 倍，带宽约提升 2.39 倍，带宽增长略快，所以 B200 Ridge Point 反而略低，为 281.2。

因此“新一代 GPU 的 Ridge Point 一定更高”不是规律。真正决定 Ridge Point 的是：

```text
Peak FLOP/s 与 HBM Bandwidth 的相对增长速度
```

对于 Llama3-8B decode，AI 上限约为 10，低于四代 GPU 的有效脊点，因此这些 GPU 在 batch 1 到 512 内都没有出现 Compute-bound crossover。

## 11. O2：FP16、INT8、INT4 的两阶段影响

实验配置：H100、Llama3-8B、`seq_len=2048`、`batch_size=1`。

典型结果：

| Precision | Weights | TTFT | ITL |
|---|---:|---:|---:|
| FP16 | 16.06 GB | 70.97 ms | 5.194 ms/token |
| INT8 | 8.03 GB | 35.47 ms | 2.757 ms/token |
| INT4 | 4.02 GB | 70.97 ms | 1.539 ms/token |

量化的核心收益是减少权重数据量：

- FP16 每个参数通常占 2 字节；
- INT8 每个参数占 1 字节；
- INT4 每个参数占约 0.5 字节。

### 11.1 为什么 Decode 获益明显

Decode 每生成一个 token，都需要反复从 HBM 读取模型权重。权重从 16 GB 降到 8 GB 或 4 GB，读取的数据量同步减少，所以 Memory-bound Decode 的 ITL 会明显下降。

此外，KV Cache 量化也可以减少 Decode 的显存占用和带宽压力，支持更大的 batch 或更长的上下文。

### 11.2 为什么 Prefill 不一定明显加速

Prefill 主要是 Compute-bound，权重只需加载一次，并被大量 prompt token 的计算复用。因此单纯减少权重字节数并不能直接解决主瓶颈。

只有当 GPU 对对应低精度提供更高的实际计算吞吐时，Prefill 的 TTFT 才会明显下降。例如本次 H100 配置中 INT4 的计算能力表回退到 FP16，因此 INT4 主要改善 ITL，而没有改善 TTFT。

总结：

```text
量化必然有利于 Memory-bound Decode；
量化对 Compute-bound Prefill 的收益取决于硬件是否有对应低精度算力。
```

## 12. O3：Llama3-8B -> Llama3-70B

实验配置：H100、FP16、`seq_len=2048`、`batch_size=1`。

典型结果：

| Model | TTFT | ITL | Total memory | 单张 H100 |
|---|---:|---:|---:|---|
| Llama3-8B | 70.97 ms | 5.19 ms/token | 约 16.33 GB | Feasible |
| Llama3-70B | 607.03 ms | 43.15 ms/token | 约 141.87 GB | Infeasible |

模型变大后：

1. 每个 token 的计算量随参数量增加，Prefill 的 TTFT 上升；
2. 每个 token 需要读取的权重变多，Decode 的 ITL 上升；
3. KV Cache 也会增加，显存压力进一步提高；
4. 70B FP16 权重约 141 GB，超过单张 H100 约 85.9 GB 显存，因此单卡结果只能作为反事实分析，不能直接部署。

模型规模变大并不会自动改变两个阶段的基本物理属性：Prefill 仍主要看算力，Decode 仍主要看带宽。它主要是把两种延迟和显存需求都放大，并可能首先撞上容量墙。

## 13. O4：SLA 驱动的硬件选型

实验配置：Llama3-8B、FP16、`seq_len=4096`、`batch_size=1`，SLA 为：

```text
TTFT < 500 ms
ITL < 50 ms/token
```

典型单卡结果：

| GPU | TTFT | ITL | TTFT SLA | ITL SLA | 两项同时满足 |
|---|---:|---:|---|---|---|
| T4 | 2294.7 ms | 52.19 ms/token | Fail | Fail | Fail |
| A100 | 478.1 ms | 8.46 ms/token | Pass | Pass | Pass |
| H100 | 150.8 ms | 5.27 ms/token | Pass | Pass | Pass |
| H200 | 150.8 ms | 3.78 ms/token | Pass | Pass | Pass |

### 13.1 多卡方式的区别

数据并行：

- 每张卡保留一份完整模型；
- 不降低单请求 TTFT/ITL；
- 主要提升并发吞吐。

流水线并行：

- 模型按层切分到多张卡；
- 单请求仍需依次经过各阶段；
- 通信和流水线气泡可能抵消收益；
- 主要用于放下更大模型和提高吞吐。

张量并行：

- 单层权重切分到多张卡；
- 每张卡承担部分计算和权重读取；
- 理想情况下可同时降低 TTFT 和 ITL；
- 是解决单请求延迟 SLA 的主要并行方式。

### 13.2 理想卡数下界

按理想线性张量并行估计：

- T4：TTFT 约需 5 张，ITL 约需 2 张，同时满足两项约需 5 张；
- A100、H100、H200：单卡即可满足本实验的两项 SLA。

这个卡数只是 Roofline 一阶模型下的下界，没有计入 PCIe/NVLink 通信、框架效率、负载不均衡和调度开销。副本数不能被当作降低单请求延迟的卡数。

选型结论：

- 成本敏感且只需满足当前 SLA：A100 已经足够；
- 关注 TTFT：H100 与 H200 算力相近，TTFT 接近；
- 关注 ITL、长输出或未来增大 batch：H200 的高带宽和更大显存更有优势；
- T4 不适合直接满足这组延迟 SLA，堆副本只能提高吞吐，不能解决单请求延迟。

## 14. 总结

本次 Task 2 的核心结论可以压缩为四句话：

1. LLM Decode 通常是 Memory-bound，低 batch 时性能主要由 HBM 带宽决定；
2. LLM Prefill 通常是 Compute-bound，TTFT 主要由有效算力决定；
3. Ridge Point 是硬件的算力/带宽分界，critical batch 是否存在还取决于模型 AI 曲线和其上限；
4. 硬件选型必须结合业务阶段：聊天和长输出优先带宽，长文档 Prefill 优先算力，70B 级模型还必须先解决显存容量和模型并行。

最终的性能诊断逻辑是：

```text
模型规模、上下文长度、batch、精度
                ↓
       每个阶段的 Arithmetic Intensity
                ↓
       与目标 GPU Ridge Point 比较
                ↓
  判断计算墙/内存墙、TTFT/ITL 和 SLA 可行性
```

所有数值均为 MLSys·im 的一阶分析估计，真实部署仍需要在目标 GPU、目标框架和真实流量下验证。
