# Task1 学习笔记：硬件与显存账本

> 课程：[DataWhale · llm-algo-leetcode 显存优化 | 202609](https://github.com/datawhalechina/llm-algo-leetcode) · [Task1 Issue #149](https://github.com/datawhalechina/llm-algo-leetcode/issues/149)
> 笔记基于官方教程整理与个人思考，未直接抄教程内容；数字均为本人在 vm-60 上的实测或按教程公式复算。
> 运行环境：vm-60 · NVIDIA RTX 4090D (24GB, 114 SM) · torch 2.9.1+cu128 · CUDA 12.8

**核心问题：当前显存压力来自哪个对象，理论容量和实际峰值应如何估算？**

对应学习材料（按「dtype → 参数规模 → 硬件条件 → 显存账本」顺序）：

1. [Part 01 · 01 数据格式与混合精度](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)
2. [Part 01 · 02 参数量与算力推导](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb)
3. [Part 01 · 06 显存计算与 ZeRO 优化](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)
4. （可选）[Part 01 · 03 GPU 物理架构与内存层级](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) · [Part 01 · 12 Tensor Core 与混合精度](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)
5. （扩展）[topic 01 显存账本与指标](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/memory_performance_tuning/01_vram_ledger_and_metrics.md) · [Part 02 · 04 多头注意力](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) · [Part 01 · 14 FlashAttention 显存模型](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)

---

## 必做问答 1：FP16、BF16（A100）、FP8（H100）这些数据精度具体指什么？

### 1. 浮点数只有三个字段，但决定了两件不同的事

一个浮点数的位宽被切成 **符号位 + 指数位 + 尾数位**，其中 **指数位管"能表示多大"（范围），尾数位管"能分多细"（精度）**。所谓"精度"从来不是单一维度：位数相同不代表能力相同，位数更少也不代表一定更差。

| 格式 | 符号/指数/尾数 | 最大有限值 | 最小正规数 | 备注 |
|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | ~3.4e38 | ~1.18e-38 | 基准与累加口径 |
| FP16 | 1 / 5 / 10 | 65504 | 6.10e-5 | 尾数细、范围窄 |
| BF16 | 1 / 8 / 7 | ~3.39e38 | ~1.18e-38 | 直接截断 FP32 尾数 |
| FP8 E4M3 | 1 / 4 / 3 | 448 | — | 精度优先，前向/激活 |
| FP8 E5M2 | 1 / 5 / 2 | 57344 | — | 范围优先，反向/梯度 |
| INT8 / INT4 | 整型 | — | — | 1 B / 0.5 B 每参数 |

### 2. A100 与 H100 各自改了什么

- **A100（Ampere）**：第一次把 **BF16 的 Tensor Core 路径**变成一等公民，并引入 **TF32**——TF32 用 FP32 的 8 位指数 + FP16 的 10 位尾数（19 bit 有效信息），但在显存里仍按 32 bit 存。它的意义是"不改代码就能让 FP32 矩阵乘走上 Tensor Core"，而不是节省显存。
- **H100（Hopper）**：加入 **原生 FP8**，并把它拆成两个变体：**E4M3 用在前向/激活**（精度优先），**E5M2 用在反向/梯度**（范围优先）。同时用 Transformer Engine 自动管理缩放，FP8 才算能真正用于训练。

### 3. 真机实测：位宽如何变成字节，以及范围差异有多真实

在 RTX 4090D 上真实分配 4096×4096 张量，读 `torch.cuda.memory_allocated` 增量：

| dtype | 实测占用 | 每元素 |
|---|---|---|
| FP32 | 64.0 MiB | 4 B |
| BF16 | 32.0 MiB | 2 B |
| FP16 | 32.0 MiB | 2 B |
| FP8 E4M3 | 16.0 MiB | 1 B |

范围差异同样实测得到：`float16(70000) = inf`（超过 65504 直接上溢），而 `bfloat16(70000) = 70144`（保持量级，末位误差 0.206%）。这解释了为什么大模型训练更愿意用 BF16：**上溢是灾难性的、不可恢复的，而末位精度损失会被后续更新平滑掉。**

**一句话**：dtype 决定的是账本里"每个参数、每个 KV cache 元素占几个字节"，它是所有容量估算的第一层乘数；但它单独无法解释实际峰值——峰值还要看对象种类、生命周期和框架行为。

## 必做问答 2：估算 LLaMA-7B 的参数量；MFU 是什么？

### 1. 参数量推导（忽略 bias 与 norm 细节）

$$
P \approx \underbrace{2Vd}_{\text{embedding} + \text{LM head}} + L\underbrace{\left(4d^2\right)}_{\text{Attention}} + L\underbrace{\left(3\,d\,d_{ff}\right)}_{\text{SwiGLU FFN}} + 2Ld
$$

- Attention：Q/K/V/O 四个 $d\times d$ 矩阵 = $4d^2$；
- FFN：SwiGLU 升维到 $\tfrac{8}{3}d$ 但有 3 个矩阵，仍是 $3 d d_{ff}$；
- 单 Block ≈ $12d^2$，所以参数量几乎由 $L \cdot 12d^2$ 决定，词表项 $2Vd$ 是常数级。

按教程函数复算 $V=32000, d=4096, L=32, d_{ff}=11008$：

| 模块 | 参数量 |
|---|---|
| embedding | 131.07 M |
| attention ($4d^2\times L$) | 2147.48 M |
| FFN ($3dd_{ff}\times L$) | 4328.52 M |
| LayerNorm ($2dL$) | 0.26 M |
| LM head | 131.07 M |
| **合计** | **6.738 B** |

这就是"7B 模型"的真实构成：**FFN 一项就占了 64%**，"7B"是量级称呼，不是精确值；换词表大小、是否共享 embedding、FFN 比例，结果都会漂。这也直接决定了显存账本里的 $4\Phi/8\Phi$ 项有多大。

### 2. MFU（Model FLOPs Utilization）

$$
\text{MFU}=\frac{\text{实际完成的模型 FLOPs}}{\text{硬件理论峰值 FLOPs}}
=\frac{6PT / t_{\text{实测}}}{\text{峰值}}
$$

分子用训练近似 $C_{train}\approx 6PT$（前向 $2PT$ + 反向 $4PT$），分母是规格书上的 Tensor Core 峰值。它衡量的是"这段代码把硬件算力用起来了多少"，而不是"训练质量好不好"。

**真机实测（本机，确定性口径：固定 shape、warmup 后取平均、CUDA Event 同步）**：

| 路径 | 实测吞吐 | 相对规格峰值 |
|---|---|---|
| FP32（关 TF32，4096³） | 47.8 TFLOPS | 57.9% / 82.6 |
| TF32（4096³） | 66.9 TFLOPS | 81.0% / 82.6 |
| BF16 Tensor Core（8192³） | 141.9 TFLOPS | **85.9% / 165.2** |

**MFU 的三个边界**（比数值本身更重要）：

1. $6PT$ 是 dense 近似，不含 attention 的 $S^2$ 项、MoE 路由和 checkpoint 重算；
2. 它是**端到端时间**算出来的，若数据加载、通信、优化器 step 阻塞，MFU 会掉到 20% 以下——这时的结论是"有阻塞"，不是"卡不行"；
3. 大 shape 才容易接近峰值：工业界大集群端到端 MFU 通常在 40%–60%。单算子 86% 与端到端 MFU 完全不是一回事。

## 必做问答 3：训练显存分成哪几部分？每一部分是怎么产生的？

### 1. 账本骨架与产生机制

| 对象 | 由什么产生 | 量级（每参数字节） | 生命周期 |
|---|---|---|---|
| **参数** | 模型权重本身 | FP32 4Φ / BF16·FP16 2Φ | 常驻 |
| **梯度** | 反向中为每个参数新建同形状张量 | 与参数 dtype 相同 | backward 写入 → 下一次 `zero_grad` |
| **优化器状态** | Adam/AdamW 首次 step 时创建 exp_avg / exp_avg_sq（+ FP32 master weights） | 8Φ（AdamW m+v，FP32）；含 master weights 还会更高 | 首次 step → 训练结束 |
| **activation** | 前向为反向保存的中间张量（saved tensors） | $\approx$ 层数 × micro-batch × seq_len × hidden × dtype；attention 的 $P$ 是 $B\cdot H\cdot N^2$ 项 | 产生 → 被对应 backward 消费完 |
| **框架 / 后端 buffer** | cuBLAS workspace、通信缓冲、allocator 碎片 | 不可由公式预测，必须实测 | 依实现而定 |
| **KV Cache（推理侧）** | 自回归解码缓存历史 K/V | $2 \times n_{kv} \times d_{head} \times$ 层数 × batch × 上下文 | 请求存活期 |

关键机制：**反向公式消费前向值**（$\text{mask}=(x>0)$、attention 的 $dV=P^T dO$ 都要用前向结果），所以 activation 必须从前向一直驻留到被消费。**backward 是唯一"三者同时在场"的阶段**：saved tensors 还没释放，反向又在创建同量级的梯度临时张量，于是峰值出现在 backward。

### 2. 真机验证：理论与实测对得上吗

在 RTX 4090D 上训练一个 52.03 M 参数的 TinyGPT（$V$=32000, $d$=512, $L$=6, $B$=8, $S$=256，FP32 参数 + bf16 autocast + AdamW），一段完整 step 的显存账本：

| 段位 | 实测 | 理论（16Φ 口径） |
|---|---|---|
| 参数 | 200.4 MiB | 198.5 MiB（4Φ） |
| 梯度（张量字节合计） | 198.5 MiB | 198.5 MiB（4Φ） |
| 优化器状态 | 400.1 MiB | 396.9 MiB（8Φ，m+v） |
| **常驻合计** | **815.1 MiB** | **793.9 MiB（16Φ）** |
| 框架 workspace | 16.1 MiB | 公式预测不到 |
| forward 后峰值 | 1065.3 MiB | — |
| **backward 后峰值** | **1315.3 MiB** | 整步峰值 |
| activation 峰值增量 | **≈ 500.2 MiB** | backward 峰值 − 常驻 |

（梯度项实测 0.0% 偏差、优化器状态 0.8% 偏差。）

**这次测量里最有价值的一条**：如果直接用 `memory_allocated` 的增量当"梯度"，会读到 214.6 MiB 而不是 198.5 MiB——多出来的 **16.1 MiB 是 cuBLAS workspace**，它是框架在第一次遇到某个 GEMM shape 时按需分配的。也就是说：

> **理论账本能算准"张量对象"（参数/梯度/优化器状态），但算不准框架侧的 workspace 与碎片；实际峰值 = 可预测的对象账本 + 必须实测的框架开销。**

这正是 topic 01 强调"账本先于技巧"的原因：只有把峰值拆到对象级，才知道该动哪一个——checkpointing 动 activation，ZeRO/sharding 动参数+梯度+优化器状态，offload 动驻留位置，量化动表示精度。

## 必做问答 4：ZeRO 概念检点

ZeRO 的核心不是"省计算"，而是**把训练状态从"每卡一份完整副本"改成"分片持有、用时临时聚合"**，用通信换显存。

| 策略 | 切分对象 | 每参数字节（FP16/BF16 + Adam） | 8×80GB 下 7B 单卡占用 | 该配置下最大可训练规模* |
|---|---|---|---|---|
| DDP | 不切 | $2+2+12 = 16\Phi$ | 112.00 GB | 4.0 B |
| ZeRO-1 | optimizer state | $4 + 12/N$ | 38.50 GB | 11.6 B |
| ZeRO-2 | + gradients | $2 + 14/N$ | 26.25 GB | 17.1 B |
| ZeRO-3 | + parameters | $16/N$ | 14.00 GB | 32.0 B |

\* 教程口径：单卡可用显存按 80GB × (1−20%) 预留 activation 与通信后反推，$N=8$。

**四个必须记住的判断**：

1. **stage 越高，省的是"常驻对象"**——ZeRO-1 只切优化器状态，却是性价比最高的一步，因为 Adam 状态本来是大头（16Φ 里占 12Φ）；
2. **activation 不在 ZeRO 的管辖范围**：它只由 micro-batch × seq_len × hidden 决定，要动它得用 checkpointing / offload / 序列并行；
3. **代价转移**：显存 ↓ 换来 all-gather / reduce-scatter 通信量 ↑、调度复杂度 ↑、切分粒度与碎片问题 ↑；单卡场景 ZeRO 收益为零；
4. **数值不代表 OOM 安全**：账本不含 workspace 与碎片（本次实测就多出 16.1 MiB 级别），所以预算必须留余量并用真机验证。

---

## 增项 1（4.2）：GPU 架构与内存架构、CUDA Core vs Tensor Core、混合精度

### 1. 内存层级与"算术强度"才是瓶颈判据

| 层级 | 容量 | 带宽量级 | 归属 |
|---|---|---|---|
| Register | 每线程几十个 | 最快 | 线程私有 |
| Shared Memory / SRAM | 每 SM 几百 KB | ~19 TB/s（教学量级） | Block 内共享 |
| L2 Cache | 几十 MB | ~1.5 TB/s（教学量级） | 全 GPU 共享 |
| HBM | 24–80 GB | 规格 1008 GB/s（本机） | 全局显存 |

**真机实测 HBM 带宽（读+写 copy）940.0 GB/s = 规格的 93.2%**，说明本机带宽确实接近理论上限，那么"算得不够快"就更容易是搬运问题而不是算力问题。判据是 **算术强度 = FLOPs / Bytes**：强度低 → 搬运先到顶（Memory Bound）；强度高 → 计算先到顶（Compute Bound）。大模型推理的 decode 阶段就是典型的低算术强度场景。

### 2. 代际演进（面向 LLM 的关键变化）

| 代际 | 关键变化 | 代表指标 |
|---|---|---|
| V100 / Volta | 首次引入 Tensor Core（只支持 FP16 MMA） | 开启混合精度训练时代 |
| A100 / Ampere | TF32 + BF16 原生、HBM2e、L2 40MB、MIG | FP16 TC ~312 TFLOPS，HBM ~1.5 TB/s |
| H100 / Hopper | 原生 FP8 + Transformer Engine、TMA、Thread Block Cluster | FP8 TC ~1979 TFLOPS，HBM3 3.35 TB/s |
| B200 / Blackwell | 第二代 TE、FP4、NVLink 5 | NVLink ~1.8 TB/s |

多卡装不下模型时，互连决定通信是否在关键路径上（PCIe Gen4 ≈ 64 Gbps vs NVLink ≈ 900 Gbps），ZeRO/FSDP 的所有收益都建立在这条互连之上。

### 3. CUDA Core 与 Tensor Core 的本质差别（含真机对照）

- **CUDA Core**：执行标量 FMA（$d = a\cdot b + c$），通用、逐元素，适合任意算子；
- **Tensor Core**：把一小块矩阵乘加打包成一次 MMA 指令，专门吃 GEMM。它不是"更快的标量单元"，而是**换了数据组织方式**——所以收益高度依赖 shape（大块、密集）与 dtype。

真机 GEMM 对照（同一张 4090D）：

| 路径 | 实测 | 占各自规格峰值 |
|---|---|---|
| FP32（无 TF32） | 47.8–48.7 TFLOPS | ~59% / 82.6 |
| TF32（FP32 截断上 TC） | 64.6–66.9 TFLOPS | ~78–81% / 82.6 |
| FP16 Tensor Core | 142.7 TFLOPS | **86.4%** / 165.2 |
| BF16 Tensor Core | 141.5–141.9 TFLOPS | **85.7–85.9%** / 165.2 |

**FP16/BF16 相对 FP32 约 2.9×**，TF32 相对 FP32 约 1.3×：Tensor Core 的收益真实存在，但需要低精度输入才吃得满。

### 4. 什么叫混合精度

混合精度的定义不是"全部降精度"，而是**沿计算图给不同环节分配不同 dtype**：

- 矩阵乘**输入**用 FP16/BF16/FP8（省存储、提吞吐、走 Tensor Core）；
- **累加器**保持 FP32（保数值稳定）；
- **优化器主权重**保留 FP32 master weights（否则 $10^{-5}$ 量级的学习率 × 小梯度会被舍入吃掉）；
- 数值敏感的算子（norm、softmax、loss）留在 FP32。

它同时改写三笔成本：**内存**（元素字节数）、**带宽**（搬运量）、**计算路径**（是否走 TC）。本机实测的范围差异（fp16 上溢为 inf、bf16 保持 70144）正是"混合"而非"全降"的直接理由。

## 增项 2（4.3）：FlashAttention、Attention 与 Transformer、MHA / GQA / MLA

### 1. Attention 与 Transformer 的关系

Transformer 是整体架构（Embedding → N × Block → LM head），**Attention 是 Block 内部的"跨 token 信息聚合算子"**；Block = Attention（token 间混合）+ FFN（逐 token 变换）。Attention 决定了模型"如何回看上下文"，也决定了长上下文推理的显存与带宽账单。Q/K/V 的含义是：每个 token 投影出 Query（我要找什么）、Key（我有什么）、Value（我能给出什么），再以 $\mathrm{softmax}(QK^T/\sqrt{d})V$ 完成加权聚合。

### 2. MHA / MQA / GQA / MLA：差别在"Q 头与 KV 头的配比"

| 结构 | 配比 | KV Cache 缩减 | 代表 |
|---|---|---|---|
| MHA | $n_q : n_q$ | 1×（最大） | 原始 Transformer |
| MQA | $n_q : 1$ | $1/n_q$ | 早期加速尝试 |
| GQA | $n_q : g$ | $g/n_q$ | LLaMA-2/3 |
| MLA | 缓存压缩潜向量，用时解压 | 大幅压缩 | DeepSeek-V2/V3 |

**真机账本（按 LLaMA-2-70B 的 64 Q 头 / 8 KV 头 / $d_{head}$=128 计算，fp16）**：

| 结构 | 单 token 单层 | 说明 |
|---|---|---|
| MHA（64 KV 头） | 32768 B = 32.00 KB | $2 \times 64 \times 128 \times 2$ |
| GQA（8 KV 头） | 4096 B = 4.00 KB | 直接省 8× |
| MQA（1 KV 头） | 512 B = 0.50 KB | 省 64×，但质量控制更难 |
| MLA（latent 576） | 1152 B = 1.12 KB | 只缓存压缩向量 |

**LLaMA-3-8B（GQA，8 KV 头，32 层）全模型 KV Cache**：4K=0.500 GiB，8K=1.000 GiB，32K=4.000 GiB，**128K=16.000 GiB**——已经逼近甚至超过中等显存卡的余量。真机按理论形状实际分配 8K 上下文全层 cache，读回的字节数 **1.000 GiB vs 理论 1.000 GiB**，账本完全对得上。这也说明长上下文推理 OOM 的主因往往不是权重，而是随 batch × 上下文线性增长的 KV Cache。

### 3. Part 02 · 04 节实现要点（本人实跑通过）

`GroupedQueryAttention.forward` 的四步：① `reshape(B,S,H,D).transpose(1,2)` 切多头；② KV Cache 在 **dim=2（seq 维）** 拼接，**必须在 `repeat_kv` 之前拼接**，否则会把已扩展的头缓存下来，丢掉 GQA 的全部收益；③ `scores = Q@Kᵀ/√d` 后加 mask、softmax、加权；④ `transpose(1,2).reshape(B,S,-1)` 合并多头。

- 教程测试结果：MHA / GQA 前向形状断言通过，KV Cache 自回归更新后形状 `(2, 4, 6, 32)` 正确；
- **额外做了一致性校验**：同一组权重下，手写实现与 `torch.nn.functional.scaled_dot_product_attention(is_causal=True)` 的输出 **最大误差 0.0e+00**（float64），确认 mask 语义与缩放因子都没写错；
- `repeat_kv` 是"延迟扩展"：缓存只存 $n_{kv}$ 个头，计算前临时广播到 $n_q$——这正是 GQA 省显存的落点。

### 4. FlashAttention：思想、tiling 与 online softmax

**要解决的问题**：标准 attention 的中间矩阵 $S=QK^T$、softmax 结果都是 $B\times H\times N\times N$，要么常驻显存，要么反复往返 HBM。它把"算不动"变成了"搬不动"。

**两条手段**：

- **Tiling（分块）**：把 Q/K/V 切成能放进 SRAM 的小块，$S_{block}$ 在片上算完就消费掉，不落 HBM；
- **Online softmax（在线归约）**：块间维护 running max $m$ 与指数和 $l$，新块到来时用 $e^{m_{old}-m_{new}}$ 修正历史累积，从而"边算边归约"，不需要先把整行 $S$ 物化再统一 softmax。

分块账本（$N$=4096，fp16 教学模型）：

| tile | 一维 tile 数 | score tile 数 | 单 tile | 相对完整 score 矩阵的物化倍率 |
|---|---|---|---|---|
| 64 | 64 | 4096 | 8.0 KB | 4096× |
| 128 | 32 | 1024 | 32.0 KB | 1024× |
| 256 | 16 | 256 | 128.0 KB | 256× |

**真机峰值实测（B=1, H=32, D=128, bf16）——最有说服力的一条证据**：

| seq_len | 标准实现峰值 | SDPA(FlashAttention) 峰值 | 倍数 | 输出最大误差 |
|---|---|---|---|---|
| 512 | 36.0 MiB | 4.1 MiB | 8.9× | 7.8e-3 |
| 1024 | 136.0 MiB | 8.1 MiB | 16.7× | 4.9e-3 |
| 2048 | 528.0 MiB | 16.3 MiB | 32.5× | 3.9e-3 |
| 4096 | 2080.0 MiB | 32.5 MiB | **64.0×** | 5.9e-3 |

标准实现的峰值随 $N$ **二次**增长（36→136→528→2080，每翻倍约 4×），FlashAttention 路径只随 $N$ **线性**增长（4.1→8.1→16.3→32.5，每翻倍约 2×），而两者输出在 bf16 下数值等价（误差 < 7.8e-3）。**FlashAttention 没有减少主要矩阵乘法，它减少的是落在 HBM 上的中间结果。**

---

## 实验证据

| 文件 | 内容 |
|---|---|
| `evidence/task1/task1_41_runshot.png` | 4.1 打卡截图：01/02/06 节测试通过 + dtype/参数量/MFU/真机账本/ZeRO |
| `evidence/task1/task1_42_runshot.png` | 4.2 截图：内存层级、真机 HBM 带宽、CUDA Core vs Tensor Core 吞吐、混合精度数值范围 |
| `evidence/task1/task1_43_runshot.png` | 4.3 截图：Part02 04 节实跑通过、KV Cache 账本、FlashAttention 分块与真机峰值曲线 |
| `evidence/task1/task1_4*.log` | 三段脚本在 vm-60 上的完整 stdout（ALL_TESTS_PASS） |
| `code/task1_shot_41.py` / `42` / `43` | 可复跑脚本（教程函数复跑 + 真机实测 + 自动出图） |
| `code/task1_common.py` | 出图公共组件（中文字体、页眉、卡片） |

复跑方式（vm-60 上已装 torch 2.9.1+cu128）：

```bash
python3 task1_shot_41.py   # 4.1 最小打卡：01 / 02 / 06 节测试 + 真机账本
python3 task1_shot_42.py   # 4.2 增项1：03 / 12 节测试 + 带宽与 Tensor Core 实测
python3 task1_shot_43.py   # 4.3 增项2：Part02 04 节测试 + KV Cache / FlashAttention 实测
```

## 一句话总结

**dtype 定了每个对象的"单价"，参数量定了"数量"，硬件定了"上限"，而账本把三者接成一条可核对的预算线**——参数/梯度/优化器状态是常驻项（16Φ，ZeRO 才能分摊），activation 是从前向驻留到被反向消费的中间项（峰值就出在 backward），KV Cache 随上下文线性增长（长上下文推理的真实瓶颈），框架 workspace 则只能实测不能预测；每一次"省显存"都要回答代价转移到了哪里（重算、通信、精度还是时间）。

## 参考与署名

- 教程：[datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)（Part 01 · 01/02/03/06/12/14、Part 02 · 04、topic_discussion 显存优化专题）
- 打卡 Issue：[#149](https://github.com/datawhalechina/llm-algo-leetcode/issues/149)
- 社区：[DataWhale](https://github.com/datawhalechina)
- 本机实测与出图脚本均由本人编写并运行，数字来自 vm-60（RTX 4090D）当次运行日志。
