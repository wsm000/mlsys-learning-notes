# Task1 学习笔记：Prefill 与 Attention Kernel（推理优化 | 202609）

> 课程：[Datawhale · llm-algo-leetcode 推理优化 | 202609](https://github.com/datawhalechina/llm-algo-leetcode) · [Task1 Issue #147](https://github.com/datawhalechina/llm-algo-leetcode/issues/147)
> DataWhale 社区：https://github.com/datawhalechina · 教程地址：https://github.com/datawhalechina/llm-algo-leetcode
> 打卡选项：**(3) 4.1 + 4.2 + 4.3**
> 运行环境：vm-60 · NVIDIA RTX 4090 D（CC 8.9 Ada，114 SM，23.53 GB，L2 72 MB）· torch 2.9.1+cu128 · CUDA 12.8 · Python 3.10.12
> 教材页：Part 01 · [14 FlashAttention 显存模型](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)、Part 02 · [20 FlashAttention 模拟](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)；扩展：[03 GPU 架构与显存](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)、[24 SRAM 优化](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)、[02 Prefill 与 Attention Kernel](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/inference_optimization/02_prefill_and_attention_kernel.md)、[34 前缀缓存与分块预填充](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)
> 代码与证据：[14_solved.ipynb](code/14_solved.ipynb) · [20_solved.ipynb](code/20_solved.ipynb) · [34_solved.ipynb](code/34_solved.ipynb) · [prefill 规模扫描脚本](code/task1_prefill_scaling.py) · [证据图](evidence/task1/task1_20_flashattention_runshot.png)

---

## 0. 先回答主问题：为什么 Prefill 会受访存和中间矩阵影响，FlashAttention 改变了什么

Prefill 一次性把整段 prompt（S 个 token）并行前向，注意力要在每个 query 位置和全部 key 位置之间算匹配度，于是自然产生一个 **S×S 的 score 矩阵**。它不是最终输出，却是这一阶段最大的中间工作集：

- 朴素实现的数据流是"写出 score → 读回来做 softmax → 写出概率 → 读回来乘 V"，**每一步都要在 HBM 里往返一趟 N² 大小的数据**；
- 序列变长时这张表按 S² 增长：S=4096、8 头、bf16 时 score 就有 256 MB，S=16384 时 4 GB，而它只是中间结果；
- GPU 的算力增长远快于显存带宽，于是计算单元大量时间在等数据——Prefill 的 attention 从"矩阵乘法问题"变成了**访存组织问题**。

我在这台 4090D 上做了固定 workload 的扫描（B=1、H=8、D=64、bf16、causal，见下表），朴素物化 attention 的峰值显存几乎严格按 S² 走，延迟也随 S 明显超线性；而 PyTorch SDPA（内部走 flash / memory-efficient 后端）的峰值显存从 13 MB 只涨到 105 MB，延迟涨到 2.4 ms 就停住了：

| seq_len | 理论 score 矩阵 (bf16) | naive 延迟 | naive 峰值显存 | SDPA 延迟 | SDPA 峰值显存 | 加速比 | 显存比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4 MB | 0.162 ms | 18.9 MB | 0.034 ms | 13.2 MB | 4.8× | 1.4× |
| 1024 | 16 MB | 0.176 ms | 46.6 MB | 0.044 ms | 22.3 MB | 4.0× | 2.1× |
| 2048 | 64 MB | 0.778 ms | 151.1 MB | 0.137 ms | 20.2 MB | 5.7× | 7.5× |
| 4096 | 256 MB | 3.378 ms | 558.1 MB | 0.288 ms | 32.3 MB | 11.7× | 17.3× |
| 8192 | 1024 MB | 12.58 ms | 2164.1 MB | 0.753 ms | 56.4 MB | 16.7× | 38.4× |
| 16384 | 4096 MB | 55.13 ms | 8544.1 MB | 2.385 ms | 104.6 MB | 23.1× | 81.7× |

![prefill 规模扫描](evidence/task1/task1_prefill_scaling.png)

用算术强度也能解释同一件事。这台机器实测：bf16 大矩阵乘可达 **141.6 TFLOP/s**，显存拷贝等效带宽 **941 GB/s**，roofline 拐点约 **150 FLOPs/Byte**。

- **朴素路径**：每个 score 元素至少经历"写 S、读 S、写 P、读 P"四次 HBM 往返，按 bf16 算约 8 字节/元素，而它的计算只有 2·d（d=128 时 256 FLOPs）→ 强度 ≈ 32 FLOPs/Byte，**远低于拐点，是 memory-bound**。
- **FlashAttention 路径**：score/P 不再落 HBM，等价于每个元素只分摊到 O(1/N) 量级的 Q/K/V 读取 → 强度被抬到拐点以上，**变成 compute-bound**。

所以 FlashAttention 改变的不是计算量（QKᵀ 仍然是 O(S²·d)），而是**中间结果住在哪一层存储、以及要搬几次**。它就是一篇 IO-aware 的工作。

---

## 4.1 最小打卡

### FlashAttention 的思想是什么，如何解决 attention 问题

一句话：**在保持注意力数学结果完全不变的前提下，用分块把中间结果留在片上，减少最贵的 HBM 搬运。**

做法是把注意力从"先物化整张 S×S 再统一归约"，改成"**边算边归约**"：

1. 固定一个 Q 分块，让它依次接收所有 K/V 分块；
2. 每接收一个 K/V 块，就在块内算 QKᵀ、掩码、softmax 和 PV；
3. 块内结果立刻合并进该 Q 行的三个状态，**块用完即弃**，不写回 HBM；
4. 全部 K/V 块走完，该 Q 行就拿到了完整上下文的输出。

它解决的问题有三层：**显存**（中间工作集从 O(S²) 降到 O(tile)）、**带宽**（消掉 N² 中间矩阵的多次 HBM 往返）、**并行/延迟**（长 prompt 的 TTFT 不再被中间矩阵的搬运拖住）。注意它**不是近似算法**——20 节的 CPU 模拟与标准 attention 的最大误差在 1e-7 量级（见下），等价性是精确的。

### tiling 具体指什么

把 Q、K、V 沿序列维切成能放进片上存储的小块（tile），外层遍历 Q 块、内层遍历 K/V 块，**任何时刻只 hold 一个 score tile，而不是整张 S×S**。

Part 01 · 14 的工作集模型给出的数量级（seq_len=4096、head_dim=128、bf16）：

| tile size | 单个 score tile | 单块工作集（Q/K/V+score+O） | 一维 tile 数 | 物化 / 分块比 |
|---:|---:|---:|---:|---:|
| 64 | 8 KB | 72 KB | 64 | ≈ 4096× |
| 128 | 32 KB | 160 KB | 32 | ≈ 1024× |
| 256 | 128 KB | 384 KB | 16 | ≈ 256× |

tile size 是一个**取舍**：tile 越小，片上工作集越小、能塞进 SRAM 的把握越大，但分块数量和调度/循环开销上升；tile 越大，调度次数下降，但要么放不进 shared memory，要么挤压 occupancy。这也解释了为什么真实 kernel 的 tile size 是跟 head_dim、dtype、SRAM 容量一起调的。

### online softmax 具体指什么

分块之后，**一个 Q 行看不到全部 score**，不能直接做 softmax（分母 l 跨所有 K 块）。online softmax 为每个 Q 行维护三个状态，做到"看一块、算一块、合并一块"：

- `m`：已处理块的最大 score（数值稳定基准）
- `l`：以 m 为基准的指数和（softmax 分母）
- `O`：已累积的加权输出

新的 K/V 块到来时：

```
m_new = max(m_old, m_block)
l_new = l_old · exp(m_old − m_new) + Σ exp(S_block − m_new)
O_new = O_old · (l_old · exp(m_old − m_new) / l_new) + (exp(S_block − m_new) @ V_block) / l_new
```

关键是**基准变了就要把旧状态重标定**（乘 `exp(m_old − m_new)`），这样每一步的 (m, l, O) 都代表"到目前为止全部 K 块"的精确结果，最终输出与标准 softmax 完全一致。m 初始化为 −∞、l 初始化为 0，用 `keepdim` 保持 [S,1] 形状方便广播——这是我在 20 节实现里的写法。

### HBM 和 SRAM 指什么，在 FA 中起什么作用

| 层级 | 是什么 | 这台 4090D 实测/探测值 | 在 FlashAttention 中的角色 |
|---|---|---|---|
| 寄存器 | 线程私有最快存储 | 65536 regs/SM | 保存当前 tile 的累加器与 m/l/O 状态 |
| **SRAM**（shared memory / L1） | 片上可编程高速存储 | **100 KB/SM**，单 block 默认 48 KB（opt-in 99 KB） | **FA 的战场**：Q/K/V tile、score tile、softmax 状态都住这里，中间结果不落 HBM |
| L2 | 全芯片共享缓存 | 72 MB | 跨 SM 复用 K/V，减少 HBM 往返 |
| **HBM** | 片外高带宽显存 | 23.53 GB，实测拷贝带宽 **941 GB/s** | 只负责放全量 Q/K/V、权重和**最终输出** |

两者的量级差距就是 FA 的全部动机：SRAM 比 HBM 快一到两个数量级，但一台 SM 只有 100 KB。FlashAttention 的思路就是"**把 HBM 留给必须长期存在的大对象，把高频复用的中间块挤进 SRAM**"。所以 tile size 的上限不是算法偏好，而是 shared memory 的物理容量。

### Part 02 · 20 CPU 运行证据（补全 TODO 后跑通全部测试）

我补全了 20 节的 `flash_attention_forward_sim`（TODO 1–6 以及可选 casual mask 扩展），在 vm-60 上用 nbconvert 跑通全部测试：

```
[seq=8, dim=4, block=2] 最大误差: 1.192093e-07
[seq=5, dim=3, block=3] 最大误差: 8.940697e-08
[seq=3, dim=2, block=1] 最大误差: 2.980232e-08
block=1: 完整 score=16384; 单 tile=1
block=4: 完整 score=16384; 单 tile=16
block=16: 完整 score=16384; 单 tile=256
✅ Online Softmax 与分块计算逻辑正确！
```

除数值等价外，causal mask（对比带上三角 mask 的标准结果）、float64（atol 1e-10）、大 score 数值稳定性（q=k=100 时输出仍有限）、`block_size<=0` 拒绝校验全部通过。**这里的等价性是精确等价，不是近似**——这正是 FA"精确 attention"的含义。

![20 节运行结果与规模扫描](evidence/task1/task1_20_flashattention_runshot.png)

### 额外做的 GPU 对照（20 节 Step 5，固定 workload）

把 20 节可选的 GPU 对照实验打开，在 4090D 上跑 naive（物化 score）与 SDPA（flash 后端）：

| 实现 | 延迟 | 峰值 allocated | 峰值 reserved | 最大误差 | 状态 |
|---|---:|---:|---:|---:|---|
| naive | 3.153 ms | 556.12 MB | 808.0 MB | 0.0078 | ok |
| SDPA | 0.234 ms | 32.25 MB | 810.0 MB | 0.0078 | ok |

workload：bf16、B=1、H=8、S=4096、D=64、causal、warmup 5 / iters 20。误差 0.0078 是 bf16（8 位尾数）的分辨率级别，两条路径在同精度下结果一致。

边界必须写清楚：**SDPA 的 flash 后端 ≠ FlashAttention-3/4**。FA3 需要 Hopper 的 TMA/WGMMA/FP8，FA4 面向 Hopper/Blackwell，消费级 sm89 上跑的是 FA1/FA2 风格的 kernel；这里只说明"同一 workload 下不物化中间矩阵的收益"，不能据此推断 FA3/FA4 的性能。

---

## 4.2 增项 1：GPU 架构与内存架构、FA 1-2-3-4、SRAM

### GPU 架构与内存架构

把 GPU 看成一座为并行计算建的工厂：

- **SM（Streaming Multiprocessor）是车间**：这台 4090D 有 **114 个 SM**（CC 8.9 / Ada）；每个 SM 内最多 1536 个线程、65536 个寄存器、100 KB 可编程 shared memory。
- **计算单元是工位**：CUDA Core 做标量/向量计算，**Tensor Core 做矩阵乘加**——这正是 attention 里 QKᵀ 和 PV 的落点。
- **warp 是班组**：32 个线程一组，SM 以 warp 为单位调度；24 节的 bank conflict 实验说明，shared memory 的 32 个 bank 一旦被同一 warp 的线程按跨步模式命中，就会把并行访问串行化。

内存层级（由近到远）：

```
寄存器（每线程私有，最快最小）
  └─ shared memory / L1（每 SM 100 KB，块内共享）   ← FlashAttention 的工作区
       └─ L2（全芯片 72 MB，跨 SM 复用）
            └─ HBM（23.53 GB，实测 941 GB/s，容量大但往返最贵）
```

核心矛盾就一句话：**算力涨得比带宽快**。用 03 节的算术强度判据看（该节的教学模型给出共享内存 19 TB/s vs L2/HBM 1.5 TB/s 的对比），elementwise 类算子强度 2 FLOPs/Byte 必然是 memory-bound；attention 的 score 计算本身强度很高（d=128 时 256 FLOPs/Byte），**真正把它拖成 memory-bound 的是被物化的中间矩阵**——03 节的 attention 工作集实验里，随 seq_len 从 512 涨到 8192，attention 矩阵占总工作集的比例从 50% 一路涨到 94%。

### FlashAttention 1-2-3-4 的创新点与硬件要求

| 版本 | 年份 | 核心创新 | 面向硬件 | 论文/来源 |
|---|---|---|---|---|
| **FA1** | 2022 | 提出 tiling + online softmax + 重计算：不物化完整注意力矩阵，用分块和片上归约把 HBM 读写降下来；**精确而非近似** | 通用 CUDA GPU（A100 上验证，sm80+ 可用） | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| **FA2** | 2023 | 更好的 **work partitioning**：减少非矩阵乘 FLOPs、单 head 也能跨 thread block 并行提高 occupancy、块内按 warp 分工减少 shared memory 通信；相比 FA1 约 2×，A100 上达到理论峰值 50–73% | Ampere 及更新（不依赖 Hopper 专属特性） | [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) |
| **FA3** | 2024 | 面向 Hopper 的**异步与低精度**：TMA + WGMMA 异步、warp-specialization 重叠搬运与计算、block quantization + incoherent processing 用上 FP8；H100 上 1.5–2.0×，FP16 达 740 TFLOP/s（75% 利用率），FP8 近 1.2 PFLOP/s | **H100/H800**，CUDA ≥ 12.3（推荐 12.8） | [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)、[官方仓库 hopper 目录](https://github.com/Dao-AILab/flash-attention) |
| **FA4** | 2025 | 用 **CuTeDSL** 重写，做算法与 kernel 流水线的协同设计（论文标题即 "Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"，针对算力/带宽非对称增长） | Hopper + **Blackwell**（H100 / B200），`pip install flash-attn-4`，CUDA 13 建议 `[cu13]` | [官方仓库 FA4 章节](https://github.com/Dao-AILab/flash-attention)、Tri Dao 博客 |

版本演进的主线很清楚：**FA1 解决"要不要物化中间矩阵"，FA2 解决"算力用没用满（并行分工）"，FA3 解决"Hopper 的新硬件单元（TMA/WGMMA/FP8）用没用上"，FA4 解决"在硬件非对称增长下把算法与流水线一起设计、并换用 CuTeDSL 这种更高层的 kernel 构建方式"**。硬件要求也逐代收紧：FA3 之后基本是数据中心 GPU 的专属能力，消费级卡（含本机 sm89 的 4090D）落地的是 FA1/FA2 风格 kernel。

### SRAM 指什么，有什么作用

SRAM 指 GPU 芯片内部、紧挨计算单元的片上存储，在可编程层面主要就是 **shared memory（以及 L1、寄存器）**。它的特点是**容量小（每 SM 100 KB 量级）、延迟低、带宽远高于 HBM**，但**必须由 kernel 显式管理**。

它在 FlashAttention 里的作用是**承载全部中间计算**：Q/K/V tile 搬进来，score tile、mask、online softmax 的 m/l/O 都在片上产生、消费、丢弃，只把最终输出写回 HBM。换句话说，FA 的性能上限很大程度上由"SRAM 里能放多大的 tile"决定。

24 节把 SRAM 的代价讲得很实：片上复用只有在**复用次数足够多、且同步/冲突代价可接受**时才划算。该节的实验输出：

- **bank conflict**：32 个 bank、32 线程时，stride=1 无冲突（max_conflict_degree=1），stride=2 立刻退化到 2 路串行、stride=32 变成 32 路串行——所以 tile 的**内存布局**和 tile 大小一样重要；
- **复用 vs 同步**：复用 1 次不划算（net_gain 0），复用 2 次以上开始有正收益；
- **寄存器 spill**：寄存器压力过大时（net_gain 从 +8.2 直接掉到 −51.0，spill=True），片上复用收益会被 occupancy 下降吃光。

这三条正好解释了真实 FA kernel 里那些"看起来跟算法无关"的工程细节：swizzle 布局、padding 避免 bank conflict、控制寄存器用量。

---

## 4.3 增项 2：Chunked Prefill 与 Prefix Cache

### 标准 Attention + Prefill 基线

基线是：**每个请求把整段 prompt 一次性送进模型做 prefill**，完整建立 KV Cache 并产出第一个 token。它的痛点是

- 中间 score 与 KV 峰值随序列增长，长 prompt 单次请求就可能把显存顶满；
- **请求之间没有复用**：系统提示词、工具说明、RAG 模板、多轮历史这些重复前缀，每个请求都要重新算一遍（重复 FLOPs、重复写 KV）；
- 长 prompt 的 prefill 会独占一段时间的算力，把首 token 延迟（TTFT）抬得很高。

### Chunked Prefill 与 Prefix Cache 分别指什么

| | Prefix Cache（前缀缓存） | Chunked Prefill（分块预填充） |
|---|---|---|
| 解决的重复 | **跨请求的重复前缀**：缓存已完成 prefill 的公共前缀的 KV，命中部分不再重算 | **单次请求内的长输入**：把未命中部分切成固定大小块，逐块执行 prefill |
| 命中规则 | 必须**从开头连续命中**，中间偶然相同的 token 不算命中 | 不涉及命中，只是执行粒度 |
| 收益 | 减少重复 prefill 的算力与 KV 写入，多轮/模板化场景直接降 TTFT | 降低单次显存峰值、便于和 decode 混排调度，改善长 prompt 的调度公平性 |
| 代价 | 额外 KV 显存、缓存淘汰策略、前缀哈希/匹配开销 | 块数变多带来调度开销，块边界可能影响 kernel 效率 |
| 与对方的关系 | 二者解决不同问题，**可以叠加**：先做前缀匹配，再对未命中 suffix 分块 | 同左 |

### 我在 34 节上做的验证

补全并跑通 `PrefixCacheManager`（TODO 1–8），token 级命中账本的行为符合预期：

```
✅ PrefixCacheManager 测试通过
add_prefix([1,2,3]) / add_prefix([1,2,9]) / 重复登记不新增条目
match_prefix([1,2,3,9]) = 3      # 前缀命中
match_prefix([1,2,0])   = 0      # 不是从开头连续命中 -> 不命中
split_prompt([1,2,3,9]) -> prefix=[1,2,3], suffix=[9]
cache_stats([1,2,3,9])  -> {hit_tokens: 3, uncached_tokens: 1, reuse_ratio: 0.75}
chunked_suffix_prefill_plan([1,2,3,9,10]) -> [(9,10)]   # 只对未命中 suffix 切块
```

再用 34 节的可选 GPU 探针（合成张量）看分块对峰值的作用：suffix 32768 token、hidden 2048、fp16 时，**一次性申请峰值 256 MB，按 chunk_size=512 分块后峰值 4 MB**。

![34 节 prefix cache 与分块预填充运行结果](evidence/task1/task1_34_prefix_cache_runshot.png)

这里的证据边界要说清楚：**这是合成张量的容量探针，不是真实 prefill kernel，也不是端到端 TTFT**。它只说明"分块执行能把单次申请峰值按块大小压下来"这一机制。真实收益仍要回到统一口径去测：固定 prompt length / generated tokens，比较 prompt length 增长时 TTFT 怎么变、prefill_share 是否高于 decode，再决定候选动作是 FlashAttention、chunked prefill 还是 prefix cache——这三者处理的是**不同层面**的瓶颈，不能混成一个动作。

---

## 5. 踩坑与自测补充

1. **causal mask 必须用全局位置**：块内偏移（`0..Bq-1`）在 j>0 的 K 块上会算错。我实现时用 `arange(i, i+Bq)` 和 `arange(j, j+Bk)` 构造全局 query/key 位置再比较，测试里 `block_size=2` 的 6×6 causal 用例才对齐。
2. **重标定容易被漏掉**：第一次写的时候只更新了 `l`，忘了 `O_old` 也要按 `l_old·exp(m_old−m_new)/l_new` 缩放，结果在"最大值后移"的用例上误差很大；数值等价测试就是拿来抓这个的。
3. **把 scale 放在外层**：`q_block * scale` 只算一次，块内不再重复缩放（参考解也这么做）。
4. **dtype 要跟着输入走**：`out/m/l` 都显式用 `q.dtype`，否则 float64 用例的 `out64.dtype == q64.dtype` 断言会挂。
5. **工具链**：notebook 用 `python3 -m nbconvert --to notebook --execute --inplace` 在 vm-60 上跑（torch 2.9.1+cu128），比手工点单元格更可复查；生成证据图时发现远端 matplotlib 缺中文字形，改用显式注册 SimHei 解决。

## 6. 小结

- Prefill 慢的根因不是"算不动"，而是 **N² 中间矩阵的物化与 HBM 往返**；用这台机器的参数算，朴素路径强度约 32 FLOPs/Byte，低于约 150 FLOPs/Byte 的拐点，是 memory-bound。
- **tiling 决定数据怎么分批放进 SRAM，online softmax 决定分批之后 softmax 为什么仍然精确**，两者合起来才让 FA 在不物化完整矩阵的前提下得到与标准 attention 一致（1e-7 量级）的结果。
- 实测同 workload 下，SDPA（flash 后端）在 S=16384 时延迟低 23×、峰值显存低 82×，与 O(S²) → O(S) 的工作集模型一致。
- 落地时还要分清三层优化：**FA 改访存路径、chunked prefill 改长输入执行粒度、prefix cache 改跨请求复用**；判断依据是 prompt length、TTFT、prefill_share 这组口径，而不是单一 FLOPs。
