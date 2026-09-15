### llm-algo-leetcode 推理优化 | 202609 Task1 · Prefill 与 Attention Kernel

微信群昵称：（填写）
GitHub ID：wsm000
打卡选择：**(3) 4.1 + 4.2 + 4.3**
运行环境：vm-60 · NVIDIA RTX 4090 D（Ada sm89，114 SM，23.53 GB，L2 72 MB）· torch 2.9.1+cu128 · CUDA 12.8 · Python 3.10.12

学习笔记：https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/notes/task01-prefill-and-attention-kernel.md
补全并跑通的 notebook：[Part 01 · 14](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/14_solved.ipynb) · [Part 02 · 20](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/20_solved.ipynb) · [Part 02 · 34](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/34_solved.ipynb)
额外实验：prefill 规模扫描（naive vs SDPA，S=512→16384）[脚本](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/task1_prefill_scaling.py) · [数据](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/evidence/task1/task1_prefill_scaling.json)

---

#### (1) flash attention 的思想是什么，是如何解决 attention 问题的？

**思想**：在保持注意力数学结果**完全不变**的前提下，用分块把中间结果留在片上，减少最贵的 HBM 搬运。它不是近似算法。

**它解决的是"中间矩阵"问题**：Prefill 一次性处理 S 个 token，朴素实现对每个 query 和全部 key 算匹配度，得到 S×S 的 score 矩阵，再"写 score → 读回来 softmax → 写概率 → 读回来乘 V"，**每个元素至少四次 HBM 往返**。序列变长时这张表按 S² 增长（S=4096、8 头、bf16 就有 256 MB，S=16384 时 4 GB），而 GPU 算力涨得比带宽快，于是 Prefill 的 attention 从"矩阵乘法问题"变成"访存组织问题"。

我在 4090D 上做了固定 workload 扫描（B=1 H=8 D=64 bf16 causal，warmup 5 / iters 20）：

| seq_len | naive 延迟 | naive 峰值显存 | SDPA 延迟 | SDPA 峰值显存 | 加速比 | 显存比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.176 ms | 46.6 MB | 0.044 ms | 22.3 MB | 4.0× | 2.1× |
| 4096 | 3.378 ms | 558.1 MB | 0.288 ms | 32.3 MB | 11.7× | 17.3× |
| 16384 | 55.13 ms | 8544.1 MB | 2.385 ms | 104.6 MB | 23.1× | 81.7× |

用算术强度解释同一件事：实测本机 bf16 大矩阵乘 141.6 TFLOP/s、显存拷贝带宽 941 GB/s，roofline 拐点约 150 FLOPs/Byte。朴素路径每个 score 元素约 8 字节 HBM 流量、只有 2·d 的计算 → 强度约 32 FLOPs/Byte，**远低于拐点，memory-bound**；FA 让中间结果不落 HBM，强度被抬到拐点以上，**变成 compute-bound**。所以 FA 改变的不是计算量（QKᵀ 仍是 O(S²·d)），而是中间结果住在哪一层、要搬几次。

#### (2) tiling 与 online softmax 具体指什么？HBM 和 SRAM 指什么？在 flash-attention 中起到了什么作用？

- **tiling**：把 Q/K/V 沿序列维切成能放进片上存储的块，外层遍历 Q 块、内层遍历 K/V 块，**任何时刻只 hold 一个 score tile**。Part 01 · 14 的工作集模型（seq=4096、head_dim=128、bf16）：tile=64 → 单块工作集 72 KB、物化/分块比 ≈4096×；tile=128 → 160 KB、≈1024×；tile=256 → ≈288 KB、≈256×。tile size 是取舍：越小越省片上空间但调度次数多，越大调度少但可能放不进 shared memory、挤压 occupancy。
- **online softmax**：分块后一个 Q 行看不到全部 score，不能直接 softmax。于是为每行维护三个状态 `m`（已见最大 score）、`l`（以 m 为基准的指数和）、`O`（累积输出），新 K/V 块到来时 **m_new = max(m_old, m_block)**，并把旧状态按 `exp(m_old − m_new)` 重标定后再合并：
  `l_new = l_old·exp(m_old−m_new) + Σexp(S_block−m_new)`，
  `O_new = O_old·(l_old·exp(m_old−m_new)/l_new) + (exp(S_block−m_new)@V_block)/l_new`。
  **基准一变就把旧状态换到新基准**，是它和标准 softmax 精确等价的关键。
- **HBM**：片外高带宽显存（本机 23.53 GB，实测拷贝带宽 941 GB/s），容量大但往返最贵；在 FA 里只负责放全量 Q/K/V、权重和**最终输出**。
- **SRAM**：片上高速存储（本机 100 KB/SM，单 block 默认 48 KB、opt-in 99 KB；再加寄存器和 L1），快但小；在 FA 里是**战场**——Q/K/V tile、score tile、mask、m/l/O 状态都在片上产生、消费、丢弃，不落 HBM。tile size 的上限不是算法偏好，而是 shared memory 的物理容量。

**Part 02 · 20 CPU 运行截图**（补全 TODO 1–6 + causal 扩展后跑通全部测试）：

![Part 02 · 20 CPU 运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task1/task1_20_flashattention_runshot.png)

```
[seq=8, dim=4, block=2] 最大误差: 1.192093e-07
[seq=5, dim=3, block=3] 最大误差: 8.940697e-08
[seq=3, dim=2, block=1] 最大误差: 2.980232e-08
✅ Online Softmax 与分块计算逻辑正确！
```

causal mask、float64（atol 1e-10）、大 score 数值稳定性、`block_size<=0` 拒绝校验亦全部通过。误差在 1e-7 量级说明**是精确等价，不是近似**。

#### 4.2 增项 1

**GPU 架构与内存架构**：SM 是车间（本机 114 个，每 SM 1536 线程 / 65536 寄存器 / 100 KB shared），CUDA Core 与 Tensor Core 是工位（QKᵀ、PV 落在 Tensor Core），warp（32 线程）是班组；内存层级由近到远是 寄存器 → shared memory/L1（100 KB/SM）→ L2（72 MB）→ HBM（23.53 GB，941 GB/s）。核心矛盾是**算力涨得比带宽快**：attention 的 score 计算本身强度很高（d=128 时 256 FLOPs/Byte），把它拖成 memory-bound 的是被物化的中间矩阵——Part 01 · 03 的工作集实验里，seq_len 从 512 涨到 8192，attention 矩阵占总工作集比例从 50% 涨到 94%。

**FlashAttention 1-2-3-4 的创新点与硬件要求**：

| 版本 | 核心创新 | 面向硬件 |
|---|---|---|
| FA1 (2022) | tiling + online softmax + 重计算，不物化完整注意力矩阵；精确而非近似 | 通用 CUDA GPU（A100 验证，sm80+ 可用） |
| FA2 (2023) | 更好的 work partitioning：减少非矩阵乘 FLOPs、单 head 也能跨 thread block 并行提 occupancy、warp 分工减少 shared memory 通信；比 FA1 约 2×，A100 上达峰值 50–73% | Ampere 及更新，不依赖 Hopper 专属特性 |
| FA3 (2024) | 面向 Hopper 的异步与低精度：TMA + WGMMA 异步、warp-specialization 重叠搬运与计算、block quantization + incoherent processing 用 FP8；H100 上 1.5–2.0×，FP16 达 740 TFLOP/s（75%），FP8 近 1.2 PFLOP/s | **H100/H800**，CUDA ≥ 12.3（推荐 12.8） |
| FA4 (2025) | 用 **CuTeDSL** 重写，做算法与 kernel 流水线协同设计（针对算力/带宽非对称增长） | Hopper + **Blackwell**（H100/B200），`pip install flash-attn-4`，CUDA 13 建议 `[cu13]` |

主线：FA1 解决"要不要物化中间矩阵"，FA2 解决"算力有没有用满（并行分工）"，FA3 解决"Hopper 新硬件单元用没用上"，FA4 解决"硬件非对称增长下算法与流水线一起设计 + 换用 CuTeDSL"。硬件要求逐代收紧，消费级 sm89（本机 4090D）落地的是 FA1/FA2 风格 kernel；本次的 GPU 对照里 SDPA 走的就是 PyTorch 的 flash 后端，**不能当作 FA3/FA4 的性能结论**。

**SRAM 指什么、有什么作用**：芯片内部紧挨计算单元的片上存储，可编程层面主要是 shared memory（外加 L1、寄存器），容量小、延迟低、带宽远高于 HBM，但需 kernel 显式管理；FA 里它承载全部中间计算。Part 01 · 24 的实验把代价讲得很实：bank conflict 上 stride=1 无冲突、stride=2 就 2 路串行、stride=32 变 32 路串行（所以 tile 的**布局**和大小一样重要）；片上复用 1 次不划算（net_gain 0）、2 次以上才有正收益；寄存器 spill 时收益从 +8.2 直接掉到 −51.0，片上复用会被 occupancy 下降吃光。

#### 4.3 增项 2

**基线（标准 Attention + Prefill）**：每个请求把整段 prompt 一次性送进模型做 prefill，完整建立 KV Cache 并产出首 token。痛点是中间 score/KV 峰值随序列增长；**请求之间零复用**（系统提示词、RAG 模板、多轮历史每轮重算）；长 prompt 独占一段算力把 TTFT 抬高。

| | Prefix Cache | Chunked Prefill |
|---|---|---|
| 解决 | **跨请求的重复前缀**：缓存已完成 prefill 的公共前缀 KV，命中部分不再重算 | **单次请求内的长输入**：把未命中部分切成固定大小块逐块 prefill |
| 命中规则 | 必须**从开头连续命中**，中间偶然相同不算 | 不涉及命中，只是执行粒度 |
| 收益 | 省重复算力与 KV 写入，多轮/模板场景直接降 TTFT | 降单次显存峰值、便于与 decode 混排，改善长 prompt 调度 |
| 代价 | 额外 KV 显存、淘汰策略、前缀匹配开销 | 块数变多带来调度开销，块边界可能影响 kernel 效率 |

与标准基线相比，**Prefix Cache 改的是"跨请求复用"、Chunked Prefill 改的是"长输入的执行粒度"**，两者解决不同问题、可以叠加（先前缀匹配，再对未命中 suffix 分块）。

我用 Part 02 · 34 做了验证：补全并通过 `PrefixCacheManager`（TODO 1–8），`match_prefix([1,2,3,9])=3`、`match_prefix([1,2,0])=0`（不是从开头连续命中），`cache_stats([1,2,3,9])={hit:3, uncached:1, reuse_ratio:0.75}`，`chunked_suffix_prefill_plan([1,2,3,9,10])=[(9,10)]`（只对未命中 suffix 切块）；34 节可选 GPU 探针（合成张量，suffix 32768 / hidden 2048 / fp16）显示**一次性申请峰值 256 MB vs chunk_size=512 分块后 4 MB**。

![34 节 Prefix Cache 与 Chunked Prefill 运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task1/task1_34_prefix_cache_runshot.png)

证据边界：34 节的探针是合成张量的容量对比，不代表真实 prefill kernel 或端到端 TTFT；20 节的 CPU 模拟只验证数值等价。真实收益仍要回到统一口径测量——固定 prompt length 与 generated tokens，看 prompt length 增长时 TTFT 如何变化、prefill_share 是否高于 decode，再决定候选动作是 FlashAttention、chunked prefill 还是 prefix cache。
