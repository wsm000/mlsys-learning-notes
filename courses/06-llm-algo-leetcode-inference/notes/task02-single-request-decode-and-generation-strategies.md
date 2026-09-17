# Task 2 · 单请求 Decode 与生成策略

> **课程**：llm-algo-leetcode 推理优化（202609）
> **教程地址**：<https://github.com/datawhalechina/llm-algo-leetcode>（DataWhale 社区）
> **打卡 Issue**：[#165 Task2 · 单请求 Decode 与生成策略](https://github.com/datawhalechina/llm-algo-leetcode/issues/165)
> **本机环境**：vm-60 · NVIDIA RTX 4090 D（Ada sm89，24 GB）· torch 2.9.1+cu128 · transformers 5.16.1 · Python 3.10.12
> **打卡选择**：(3) 4.1 + 4.2 + 4.3（3 个增项实验全部实测）

单条请求的生成可以拆成两个阶段：**Prefill** 一次读完 prompt 并建立 KV Cache，**Decode** 每轮只推进一个 token，循环「读已有 KV → 算 logits → 选 token → 追加状态」。Task2 关心的就是后半段：这一步里哪些东西是可改的，改了之后显存、延迟、质量和稳定性分别怎么变。

| 可改点 | 代表手段 | 本文对应问题 |
|---|---|---|
| 候选生成 | draft 模型先行草拟、单轮多候选 | 4.1(4)、4.2 |
| 选择规则 | greedy / temperature / top-k / top-p | 4.1(3)、4.3(2) |
| 目标验证 | min(1, p/q) 接受-拒绝、residual 修正、bonus | 4.1(4)、4.2(2) |
| 停止与回退 | EOS、首次拒绝即回退 | 4.2(3) |
| 历史状态成本 | KV Cache 的表示 / 组织 / 复用 / 精度 | 4.1(1)(2)、4.3(1)(3) |

**本文所有数字都来自本机真实运行**：6 个 notebook 的解答版在 vm-60 上执行通过，外加 4 个自加实验脚本（脚本、JSON、日志、运行截图都在 [evidence/task2](../evidence/task2/)）。

---

## 4.1 最小打卡

### （1）KV Cache 增长与哪些因素有关？近似账本公式是什么？为什么它是线性而不是平方？

**账本公式**（单个请求、推理态、每层都存 K 和 V）：

`KV Cache Bytes \approx \underbrace{2}_{K,V} \times L \times B \times H_{kv} \times D \times S \times \text{dtype_bytes}`

- `L`：层数——每一层都有自己的一份历史状态，成本在层维度上重复
- `B`：batch size / 并发请求数——每个请求各自持有独立缓存
- `H_{kv}`：KV 头数（注意不是 query 头数）
- `D`：head_dim
- `S`：已缓存的历史 token 数（上下文长度）
- `2`：K 和 V 各一份；dtype_bytes 是每个元素的字节数（fp16=2）

**为什么是线性而不是平方**：每个新 token 对缓存的贡献是一个**常数** `2 \times L \times H_{kv} \times D \times \text{dtype_bytes}` 字节，与 `S` 无关；`S` 在这个公式里只出现一次（一阶），所以对 `S` 求导是常数。

平方项 `S^2` 属于**另一个对象**：attention 的 score 矩阵 `QK^T`（`S \times S`）。它是 prefill 阶段的**中间结果**，每处理一个请求就产生、用完即弃，不会被长期保留——这正是 Task1 里 FlashAttention 要消掉的东西。把这两件事分开记：**KV Cache 是「历史状态成本」（线性、常驻），score 矩阵是「中间计算成本」（平方、瞬时）**。所以在长上下文里最先爆的通常是显存容量而不是算力。

**本机实测验证**（task2_exp1_kv_ledger.py，合成 LLaMA：12 层 / hidden 1024 / 8 query heads / head_dim 128 / GQA 2 KV heads / fp16）：

| 变化项 | 实测 | 账本 | 结论 |
|---|---:|---:|---|
| S=512, B=1 | 6.0 MiB | 6.0 MiB | 每 token 12288 B |
| S=2048, B=1 | 24.0 MiB | 24.0 MiB | 每 token 12288 B（不变） |
| S=4096, B=1 | 48.0 MiB | 48.0 MiB | 每 token 12288 B（不变） |
| S=8192, B=1 | 96.0 MiB | 96.0 MiB | 每 token 12288 B（不变） |
| S=2048, B=8 | 192.0 MiB | 192.0 MiB | 随 batch 线性 |

序列翻倍缓存就翻倍、batch 翻倍缓存也翻倍，而且**每 token 字节数恒定**——这就是「线性」的实测形态。实测字节数与 `2 \cdot L \cdot B \cdot H_{kv} \cdot D \cdot S \cdot 2` 完全一致（误差 0），说明这部分显存就是这些张量本身，没有隐藏放大。

**放到真实模型尺度上的账本**（80 层、head_dim 128、fp16，纯计算，不实测）：

| 配置 | 每 token（全层） | 32K 上下文 B=1 | 32K 上下文 B=32 |
|---|---:|---:|---:|
| MHA（64 KV 头） | 2560 KiB | 80 GiB | 2560 GiB |
| GQA（8 KV 头） | 320 KiB | 10 GiB | 320 GiB |
| MQA（1 KV 头） | 40 KiB | 1.25 GiB | 40 GiB |
| MLA（latent 512 近似） | 80 KiB | 2.5 GiB | 80 GiB |

**公式的边界**：它只算 K/V 张量本身，不含 allocator 碎片、页对齐浪费、workspace 和临时张量；实测里 S=4096/B=16 的峰值显存远大于缓存本身。要判断真实占用必须实测，本文里的做法就是两者都给。

### （2）针对 KV Cache 压力，可以从哪些层面优化？各自减少的对象和新增代价是什么？

| 层面 | 代表机制 | 减少的对象 | 新增代价 |
|---|---|---|---|
| 表示压缩 | MQA / GQA / MLA | `H_{kv}`（每组 K/V 服务的 query 头变多），MLA 更激进地把 K/V 压到 latent 空间 | 表达能力与训练配方要改；MLA 需要额外的上投影/吸收计算 |
| 缓存组织 | PagedAttention | 连续大块预留和内部碎片（按页分配、按需增长） | 页表管理、非连续访存、kernel 复杂度 |
| 缓存复用 | Prefix Cache | 跨请求重复前缀的 prefill 计算与 KV 写入 | 额外缓存显存 + 淘汰策略 + 前缀匹配开销 |
| 数据精度 | KV Cache 量化（FP8/INT8） | 每元素字节数 dtype_bytes | 反量化开销与精度损失，需要校准 |
| 执行粒度 | Chunked Prefill / PD 分离 | 单次峰值与相互干扰（算力占用被切开） | 调度开销、块边界效率、额外传输 |
| 历史长度 | 滑窗 / Attention Sink / 稀疏注意力 | `S`（不再保留全部历史） | 长程信息丢失，质量需要重新评估 |

把这张表和 4.3(1) 的实验对上：**改的是「每 token 存多少」（表示/精度）、「存的形状怎么放」（组织）、「要不要重算」（复用）、「留着多少」（长度）**。同一份推理负载下，先看压力来自哪一项再选动作。

我实测的 MHA/GQA/MQA 对照（同一骨架只改 `num_key_value_heads`，S=4096、B=1、fp16）：

| 配置 | KV 头数 | 实测 cache | 账本 | 每 token | 20 GiB 预算下并发请求数 |
|---|---:|---:|---:|---:|---:|
| MHA | 8 | 192.0 MiB | 192.0 MiB | 48.0 KiB | 106 |
| GQA | 2 | 48.0 MiB | 48.0 MiB | 12.0 KiB | 426 |
| MQA | 1 | 24.0 MiB | 24.0 MiB | 6.0 KiB | 853 |

即：KV 头数按 8:2:1 下降，缓存也按 8:2:1 下降，**能同时服务的请求数按同样比例上升**。这就是长上下文模型普遍采用 GQA/MQA 的直接原因——它先买到的是容量与并发，而不是单步延迟（延迟什么时候才受影响，见 4.3(1) 的长上下文对照）。

### （3）Greedy、Temperature、Top-k、Top-p 分别改变了什么？为什么提高 Temperature 可能增加多样性、也可能降低稳定性？

四者都作用在**同一个位置**：模型算出 logits 之后、`multinomial` 采样之前。它们都不改变模型的前向计算，只改变「哪些 token 还有资格、以什么概率被选中」。

| 策略 | 改变的对象 | 数学动作 | 不改变的东西 |
|---|---|---|---|
| Greedy | 选择规则（不采样） | 取 argmax | 分布本身 |
| Temperature | 分布的**平坦程度** | `z \leftarrow z/T`（`T<1` 变尖，`T>1` 变平） | token 排序（单调变换） |
| Top-k | 候选**集合大小** | 保留第 K 大及其以上，其余置 `-\infty` 后再 softmax | 保留项的相对比例 |
| Top-p | 候选集合的**累计质量** | 按降序累计到 `p` 的最小集合，边界 token 保留，其余置 `-\infty` | 截断外的质量分布 |

**实测（task2_exp2_sampling.py，gpt2 124M，6 个 prompt × 3 个 seed，max_new=100）**：

| 策略 | 平均长度 | 重复 bigram 占比 rep2 | distinct-2 | 自困惑度 PPL（std） | 与 greedy 逐 token 一致率 | 解码 tok/s |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 100.0 | **0.684** | 0.095 | 2.23 (0.57) | 1.000 | 126.7 |
| T=0.3 | 100.0 | 0.580 | 0.328 | 2.50 (0.73) | 0.074 | 118.7 |
| T=0.7 | 100.0 | 0.166 | 0.703 | 6.28 (1.40) | 0.026 | 119.3 |
| T=1.0 | 99.5 | 0.070 | 0.838 | 12.47 (2.77) | 0.015 | 119.9 |
| T=1.5 | 100.0 | **0.022** | **0.918** | **33.88 (7.11)** | 0.007 | 119.1 |
| top_k=10 (T=1) | 100.0 | 0.141 | 0.760 | 7.30 (1.18) | 0.024 | 119.7 |
| top_p=0.9 (T=1) | 100.0 | 0.113 | 0.795 | 8.96 (3.03) | 0.014 | 116.9 |
| top_k10+top_p0.9 | 99.9 | 0.175 | 0.702 | 6.33 (1.51) | 0.033 | 117.0 |

三条单调规律：温度越高，**重复率越低**（0.684 → 0.022）、**distinct-2 越高**（0.095 → 0.918）、**与 greedy 的一致率越低**（1.000 → 0.007）。

**为什么多样性会上升**：除以 `T>1` 把 logits 的差距压平，softmax 之后的分布熵变大，原本概率极小、排在后面的长尾 token 也获得了可采样的概率质量；不同 seed 因此更容易走到不同分支上，跨样本的 distinct-n 自然上升。

**为什么稳定性会下降**：同一条 prompt 在不同 seed 下的输出开始分岔——跨 seed 的自困惑度标准差从 0.57 涨到 7.11，与 greedy 的逐 token 一致率从 1.0 掉到 0.007。也就是说「同一次调用还像不像上一次」这件事被牺牲掉了；对需要确定性复现或格式严格的场景（工具调用、代码补全）这是直接风险。

**这里有一个必须点破的指标陷阱**：表里 greedy 的自困惑度最低（2.23），但它的重复率最高（0.684）——因为自回归模型对自己刚刚重复过的那句话打分当然很高。**PPL 低不等于质量高**，它必须和重复率一起读。这也是我在 4.3(2) 里同时记录「长度 / 重复率 / 多样性 / 质量 / 一致性」的原因。

**Top-k 与 Top-p 的定位**：它们是把温度那种「整体压平」换成「直接砍掉尾部」的粗粒度手段，因此比纯温度更可控——表里 top-k=10、top-p=0.9 在 T=1 的多样性水平（distinct-2 0.76/0.80）明显低于 T=1.5，但 PPL 好得多（7.3/9.0 vs 33.9）。工业做法通常是「低温度 + 截断」组合，本表最后一行的 top_k10+top_p0.9 就是这种组合：质量回到 6.33、重复率被压到 0.175。

**一个容易忽略的观察**：8 种策略的解码速度几乎一样（117–127 tok/s），差异在 5% 以内。**采样策略的开销可以忽略，代价全部体现在输出分布上**——所以对它们的选择本质是质量/稳定性取舍，而不是性能取舍。

### （4）如何理解投机解码？草稿模型和目标模型分别承担什么职责？

**一句话**：用一个小而便宜的 draft 模型先连续猜出 `K` 个 token，再让大而准的 target 模型**一次性并行验证**这 `K` 个位置——把 target 的 `K` 次串行前向压缩成 1 次前向。它不改变最终分布，只改变「谁来推进」。

| 参与者 | 职责 | 产出 | 成本特征 |
|---|---|---|---|
| 草稿模型（draft） | 便宜地提出连续候选，负责**推进长度** | `draft_tokens` + `draft_probs`（`[K, vocab]`） | 每 token 都算一次，但模型小 → 摊薄 |
| 目标模型（target） | 集中验证候选并提供兜底，负责**正确性** | `target_probs`（`[K+1, vocab]`，多一行给 bonus） | 每轮只算一次前向（`K+1` 个位置并行） |

**单轮的四个动作**（23 节解答版）：

1. **提议**：draft 给出 `K` 个 token 及其概率 `q(x)`；
2. **验证**：逐个位置比较 `p(x)` 与 `q(x)`，接受概率 `\alpha(x)=\min(1, p(x)/q(x))`；`p \ge q` 时必然接受；
3. **拒绝修正**：一旦第一次拒绝就终止本轮后续验证，并从 residual 分布 `p'(x) \propto \max(p(x)-q(x), 0)` 归一化后采样一个**修正 token**；
4. **全部接受**：追加一个 **bonus token**，取自 target 多出来的第 `K` 行分布。

第 3、4 步是「无损」的关键：被接受的部分按 `\min(1,p/q)` 校正，被拒绝时用 residual 补齐剩余概率质量，因此**最终样本的分布仍等于 target 自己的分布**——投机解码是等价加速，不是近似。题目区的测试正好把四条路径都钉住了：必接受、按 `p/q` 掷硬币后接受、拒绝后 residual 采样得到 `3`、全接受后 bonus 得到 `4`；另外两个边界也很重要：`q=0` 且 `p>0` 时应直接接受（否则除零会误拒），residual 质量为 0 时必须显式报错而不是静默采样。

**收益的条件**：平均一轮推进 `1+\bar{a}` 个 token（`\bar a` 为平均接受长度），耗时 ≈ draft 的 `K` 步 + target 的 1 次前向。所以账要这么算：**接受率要高、verify 不能太贵、draft 要足够便宜**，三者缺一不可。这正是 68 节的判定顺序（`quality_ok → acceptance_rate → throughput_gain → verify_cost_delta`），也是我把它作为增项的原因：投机解码的收益不是「能不能跑」的问题，而是「这条链路在这个 workload 下值不值得留」的问题。

---

## 4.2 增项 1：多 Token 解码

### （1）什么叫多 Token 解码？

多 token 解码（Multi-Token Decoding）指的是**在一次解码步里尝试推进多个 token**：先批量生成一段候选前缀，再由目标侧验证，**连续接受的候选前缀被一次性确认**，遇到第一个拒绝就停止并把后面的候选作为回退后缀丢弃/重生成。它的动机和投机解码一样——减少「每次只推进 1 个 token」带来的往返次数（KV 追加、kernel 启动、调度开销），区别在于强调点是「单轮产出多个」而不是「由谁提议」。

35 节解答版用 `MultiTokenDecoderSim` 把控制流跑通：`propose`（截取本轮候选前缀）→ `verify`（从左到右验证，首次拒绝即停）→ `decode`（输出接受前缀、拒绝位置、回退后缀、推进比例）。测试给出的账是：提议 3 个、接受 `[10, 20]`、`rejected_at=2`、`rejected_suffix=[30]`、`progress_per_round = 2/3`；全部接受时 `progress_per_round = 1.0` 且回退后缀为空。

### （2）与投机解码在「候选生成」和「目标验证」上的主要区别

| 维度 | 投机解码（Speculative Decoding） | 多 Token 解码（Multi-Token Decoding） |
|---|---|---|
| 候选生成 | **独立的 draft 模型**采样出候选，候选分布 `q` 与 target 分布 `p` 不同 | 在同一个解码步内尝试推进多个候选，候选可来自近似/并行头/同模型的受控提议 |
| 目标验证 | 用 `\alpha=\min(1, p/q)` 做接受-拒绝采样，**保证最终分布等价于 target** | 常用阈值/一致性规则（如本节 `target_prob >= draft_prob * min_accept_ratio`），**只保证控制流正确，不保证分布等价** |
| 增量成本 | 多一个 draft 模型（显存 + 每步 `K` 次前向） | 多一次候选提议与验证，通常不引入第二个模型 |
| 失败处理 | 拒绝 → residual 采样得到修正 token，本轮结束 | 拒绝 → 该位置起成为回退后缀，保守路径重生成 |
| 本节实现 | 23 节：`speculative_decode_step`（接受/修正/bonus） | 35 节：`MultiTokenDecoderSim`（提议/验证/回退/推进比例） |

一句话概括差别：**投机解码的核心是「分布校正」（无损），多 token 解码的核心是「单轮吞吐」（控制流）**。两者共享同一个难点：接受率、验证成本、首个拒绝之后的回退。

### （3）一轮中，候选 token、验证结果和状态更新之间是什么关系？

| 顺序 | 输入 | 产生的状态 | 对下一步的影响 |
|---|---|---|---|
| ① 提议 propose | `draft_tokens` | `proposed = draft_tokens[:max_proposal_len]` | 决定本轮**最多**能验证几个位置 |
| ② 验证 verify | `proposed` + 对应行的 `draft_probs` / `target_probs` | 逐位置 `accepted` 布尔值；首次为假时写入 `rejected_at` | 一旦拒绝，**后面所有候选不再验证**（它们依赖被拒前缀） |
| ③ 状态汇总 decode | `proposed` / `accepted_tokens` / `rejected_at` | `accepted_len`、`progress_per_round`、`rejected_suffix` | 接受前缀 = 本轮真正推进的 token；后缀 = 回退/重生成对象 |
| ④ 全部接受 | 所有位置通过 | `rejected_at=None`、后缀为空、`progress_per_round=1.0` | 本轮推进长度取到上限 |

关键点：**候选是「批量提出」，验证是「串行短路」**。验证之所以必须从左到右并在首次拒绝处停止，是因为第 `i+1` 个候选的合法性建立在前 `i` 个被接受的前提上——前缀被否，后面的验证结果就没有意义。测试里 `max_proposal_len` 从 3 变成 1/2/4，接受的 token 数随之变化，正好把这条依赖关系量化了。而 `progress_per_round` 只是「本轮接受比例」，**不是吞吐收益**：真实吞吐还要除以 draft 提议成本和 verify 成本。

---

## 4.3 增项 2：最小实验设计（4 选 2，本文做了 3 个）

### （1）如何设计一个最小实验，比较不同 KV head 配置下的显存占用、缓存利用率和生成性能？

**设计**（task2_exp1_kv_ledger.py）：固定模型骨架（12 层 / hidden 1024 / 8 个 query head / head_dim 128 / fp16），**只把 `num_key_value_heads` 从 8（MHA）改成 2（GQA）改成 1（MQA）**；用 `use_cache=True` 前向一次，把返回的 cache 对象里的张量按 `data_ptr` 去重求和得到「实测 cache 字节」；再与 `2 \cdot L \cdot B \cdot H_{kv} \cdot D \cdot S \cdot 2` 对账。缓存利用率用**固定显存预算（20 GiB）下能同时容纳的请求数**表达；生成性能测 prefill 延迟、decode 每步延迟与吞吐。

| 观察量 | 口径 | 为什么这样定 |
|---|---|---|
| 显存占用 | 求和 cache 张量字节 + `max_memory_allocated` | 前者是账本，后者含激活值和池，两者一起看才知道「理论 vs 实际」 |
| 缓存利用率 | `\lfloor` 预算 / 单请求 cache `\rfloor`，以及 cache / 预算 | 直接对应「这张卡能同时服务多少请求」，比百分比更可操作 |
| 生成性能 | 固定 B/S 的 prefill ms、decode ms/step、tok/s | 必须固定 workload，否则比的是不同上下文长度 |

**结果**：

| 配置 | KV 头 | cache(S=4096,B=1) | 每 token | 20 GiB 并发数 | prefill(B=8,S=1024) | decode ms/step | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| MHA | 8 | 192.0 MiB | 48.0 KiB | 106 | 35.34 ms | 11.15 | 717.2 |
| GQA | 2 | 48.0 MiB | 12.0 KiB | 426 | 31.28 ms | 11.38 | 703.3 |
| MQA | 1 | 24.0 MiB | 6.0 KiB | 853 | 30.56 ms | 11.25 | 711.3 |

**判读（这张表最重要的一句话）**：在**短上下文、小 batch** 下，KV 头数从 8 降到 1 让容量翻 8 倍，但 decode 每步耗时几乎不动（11.15 → 11.25 ms）——这个规模下每步由权重读取与 kernel 启动主导，KV 读取还没成为瓶颈。**容量差异先变现，延迟差异要等到长上下文才出现**：

| 配置 | S=2048, B=16 | S=8192, B=16 |
|---|---|---|
| MHA | 11.10 ms/step，峰值 4841 MiB | **22.58 ms/step**，峰值 **15642 MiB** |
| GQA | 11.30 ms/step，峰值 3690 MiB | 11.30 ms/step，峰值 11034 MiB |
| MQA | 11.26 ms/step，峰值 3498 MiB | 11.29 ms/step，峰值 10266 MiB |

S=8192 时 MHA 的 cache 是 MQA 的 8 倍，每步耗时也变成了 GQA/MQA 的 2 倍（22.6 vs 11.3 ms）：此时每步要读的 KV 达到 GB 量级，**decode 从「算不动」变成「读不动」**。而 MHA 的峰值 15.6 GB 已经吃掉 24 GB 卡的 2/3，直接限制了并发。结论：**KV head 配置既是容量旋钮，也是长上下文下的延迟旋钮**，判断该不该动它，取决于你的瓶颈在「能放多少请求」还是「每步读多少字节」。

### （2）如何设计一个最小实验，比较不同采样策略对质量、重复率和生成长度的影响？

**设计**（task2_exp2_sampling.py）：固定模型（gpt2，124M）、固定 prompt 集（6 条）、固定 `max_new_tokens=100`、固定 seed 集合（3 个），**只改解码策略**；每个策略在每个 prompt × seed 上生成一次，保证可比。

| 指标 | 定义 | 它回答的问题 |
|---|---|---|
| 生成长度 | 生成的 token 数（遇到 EOS 提前结束） | 输出是否被提前掐断/是否啰嗦 |
| 重复率 rep2 | 重复 bigram 占比 `1-\frac{\text{unique bigram}}{\text{全部 bigram}}` | 是否陷入复读 |
| distinct-1/2 | 跨样本 n-gram 去重比例 | 采样多样性（不是单词多样性） |
| 自困惑度 PPL | 用同一模型给「prompt+生成」打分，取生成段 | 流畅度代理，**必须与重复率一起读** |
| 与 greedy 一致率 | 逐 token 相同的比例 | 偏离确定路径多远 |
| 解码 tok/s | 端到端生成速度 | 采样本身的开销 |

结果表见 4.1(3)。**主要结论**：① 温度升高 → 重复率单调下降（0.684 → 0.022）、distinct-2 单调上升（0.095 → 0.918）、PPL 单调变差（2.23 → 33.88）、跨 seed 波动单调变大（std 0.57 → 7.11）；② top-k/top-p 是「砍尾部」版本的温度，多样性与质量都停在中间档；③ **生成长度在这个设置下几乎不变**（都跑满 100），只有 T=1.0 与 top_k10+top_p0.9 出现 5.6% 的提前 EOS——说明想用采样参数控制「长度」，不如直接设 `max_new_tokens` 或长度惩罚；④ 速度几乎不受影响（117–127 tok/s）。

### （3）推理性能比较时，为什么要同时记录 TTFT、TPOT、吞吐、峰值显存和端到端延迟？这些指标分别反映什么？

| 指标 | 测量边界 | 反映的问题 | 只看它会漏掉什么 |
|---|---|---|---|
| TTFT | 请求发出 → 第一个 token | **交互体验**（用户等多久看到字） | 稳态成本，长输出场景会误判 |
| TPOT | 首 token 之后的平均每 token 时间 | **稳态生成速度 / 单请求解码效率** | 首字延迟，短输出场景会误判 |
| 吞吐 | 单位时间产出的 token 数（含 batch） | **服务容量 / 单位成本** | 单请求体验：高吞吐常靠大 batch |
| 峰值显存 | 全流程 `max_memory_allocated` | **能塞下多大 batch / 多长上下文**（决定并发上限） | 延迟与质量 |
| 端到端延迟 | 请求发出 → 最后一个 token | **用户实际感知**，`\approx` TTFT + TPOT×输出长度 | 内部构成，无法定位瓶颈 |

**实测（task2_exp3_metrics.py，gpt2，gen=32，每个组合 warmup=2 + 取 3 次中位数）**：

prompt 长度扫描（B=1）：

| prompt | TTFT | TPOT | e2e | 吞吐 | 峰值显存 |
|---:|---:|---:|---:|---:|---:|
| 32 | 8.60 ms | 7.37 ms | 237.1 ms | 134.9 tok/s | 259 MiB |
| 128 | 9.34 ms | 7.36 ms | 237.6 ms | 134.7 tok/s | 275 MiB |
| 512 | 11.29 ms | 7.48 ms | 243.2 ms | 131.6 tok/s | 340 MiB |
| 768 | 9.96 ms | 7.45 ms | 240.9 ms | 132.9 tok/s | 386 MiB |

batch 扫描（prompt=256）：

| batch | TTFT | TPOT | e2e | 吞吐 | 峰值显存 |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.71 ms | 7.50 ms | 242.1 ms | 132.2 tok/s | 297 MiB |
| 4 | 9.93 ms | 7.92 ms | 255.6 ms | 500.9 tok/s | 430 MiB |
| 16 | 18.80 ms | 7.75 ms | 259.1 ms | 1976.0 tok/s | 959 MiB |
| 32 | 37.91 ms | 7.83 ms | 280.7 ms | 3648.2 tok/s | 1662 MiB |

用这两张表回答「为什么要一起记」：

- **prompt 变长**：TTFT 上升（prefill 要算更多 token），TPOT 基本不动（decode 每步只处理 1 个新 token）→ 只看吞吐会以为「没变化」，而用户的等待时间其实变长了；
- **batch 变大**：吞吐近线性上涨（132 → 3648 tok/s）、峰值显存同步上涨（297 → 1662 MiB），但 TTFT 从 9.7 ms 涨到 37.9 ms，而 TPOT 几乎不动（7.5 → 7.8 ms）→ **吞吐的收益是用 TTFT 和显存换来的**，这正是 66 节里「candidate 只提升吞吐但拉高 TTFT 时要说明适合离线还是在线」的实测形态；
- **端到端** `\approx` TTFT + TPOT × 31 在这张表里成立（9.7 + 7.5×31 ≈ 242 ms），说明这套口径自洽，同时也说明**短输出场景里 TTFT 占比只有 4%，长输出场景里 TPOT 才是大头**——两个指标缺一个就会得出相反结论。

**补一个把 4.1(1) 和性能指标接起来的对照**（task2_exp3b_kv_reuse.py，合成 LLaMA，只切换是否复用 KV）：

| B | S | 带缓存 TPOT | 每步重算 TPOT | 倍数 | 峰值显存（重算 → 带缓存） |
|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 11.2 ms | 17.0 ms | 1.5× | 932 → 765 MiB |
| 8 | 2048 | 11.3 ms | 75.1 ms | 6.6× | 2457 → 1827 MiB |
| 16 | 4096 | 11.2 ms | 380.8 ms | **34×** | 8567 → 6077 MiB |

序列越长，不做缓存就要按 `O(S)` 重算，S=4096 时端到端从 0.50 s 变成 6.05 s。**KV Cache 用「线性增长的常驻显存」换掉了「每步随上下文线性增长的计算」**——这正是 4.1(1) 那个账本公式的收益面。

**口径边界（必须写清楚）**：这三张表都是 HuggingFace eager 单流实现，gpt2 每步 7.4 ms 主要由框架与 kernel 启动开销主导，不代表任何 serving 栈（vLLM/SGLang）的性能；它们的价值在于**指标定义与趋势**，不在绝对值。

### （4）如何设计 baseline 与 speculative candidate 的统一实验？

虽然按「4 选 2」我做了前三个，这里仍把我理解的设计写出来（68 节的 G0/G1/G2 框架）：

**固定项**：同一 target 模型与版本、同一 prompt 分布、同一 batch / max_new_tokens / temperature / top_p、同一 warmup 与重复次数、同一统计口径（P50/P99）。**变量**：只改「是否启用 draft + proposal 长度」。

**三组**：G0 = target 单独跑（baseline）；G1 = target + draft（speculative，proposal_len 固定）；G2a/G2b = 同一 speculative 链路只改 `proposal_length`（如 3 vs 8），用来找收益曲线的拐点。

**必须同时记录的字段**：TTFT、TPOT、吞吐、峰值显存、**acceptance rate**（逐轮接受分布，不只均值）、**draft cost / verify cost**，以及**质量门槛**（同一评测集上的通过率，或与 G0 输出的分布一致性）。判读顺序按 68 节：先看质量是否达标（`quality_ok=False` 直接 reject），再看接受率是否过门槛，再看吞吐增益，最后看 verify 成本是否可接受；输出统一为 `accept / tune / reject` 与下一步动作（如 `refine_draft_or_verify`）。

**一个关键纪律**：speculative 的「加速比」必须与 acceptance rate 一起报告。低接受率下的吞吐提升往往是特定 prompt 分布的偶然结果；而 verify 太贵时，即使接受率很高也不值得保留——这两句话正是 68 节 Step 3 的核心，我在 68 节解答版里把它们实现成了可执行的判定函数（`compare_speculative_to_baseline` + `recommend_speculative_run`）。

---

## 5. 证据清单

| 证据 | 文件 | 说明 |
|---|---|---|
| Part 01 · 11 KV Cache 与显存增长 | [11_solved.ipynb](../code/11_solved.ipynb) · [runshot](../evidence/task2/task2_11_kv_cache_runshot.png) · [log](../evidence/task2/run_11_KV_Cache_and_Memory_Growth.log) | 账本公式、序列/batch/头数三个小验证全部通过（该节题目区无 TODO，直接执行） |
| Part 02 · 21 解码策略 | [21_solved.ipynb](../code/21_solved.ipynb) · [runshot](../evidence/task2/task2_21_decoding_runshot.png) · [log](../evidence/task2/run_21_Decoding_Strategies.log) | 补全 temperature / top-k / top-p 三个 TODO；ties、非法参数、batch、随机种子、自回归循环断言全通过 |
| Part 02 · 23 投机解码 | [23_solved.ipynb](../code/23_solved.ipynb) · [runshot](../evidence/task2/task2_23_speculative_runshot.png) · [log](../evidence/task2/run_23_Speculative_Decoding.log) | 补全输入契约、接受、residual 修正、bonus 四个 TODO；`q=0` 与 residual=0 边界通过 |
| Part 02 · 35 多 Token 解码 | [35_solved.ipynb](../code/35_solved.ipynb) · [runshot](../evidence/task2/task2_35_multi_token_runshot.png) · [log](../evidence/task2/run_35_Multi_Token_Decoding.log) | 补全 propose / _accept_token / verify / decode 四个 TODO；推进比例与回退后缀断言通过 |
| Part 02 · 66 推理性能比较 | [66_solved.ipynb](../code/66_solved.ipynb) · [runshot](../evidence/task2/task2_66_inference_metrics_runshot.png) · [log](../evidence/task2/run_66_Inference_Performance_Comparison.log) | 补全 7 个函数（离散事件模拟 + 指标口径 + 瓶颈诊断 + 对照 + 选型）；真实 backend 段按节默认保持 CPU-first |
| Part 02 · 68 投机解码基准 | [68_solved.ipynb](../code/68_solved.ipynb) · [runshot](../evidence/task2/task2_68_speculative_benchmark_runshot.png) · [log](../evidence/task2/run_68_Speculative_Decoding_Benchmark.log) | 补全逐轮验证/汇总/对照/决策四个 TODO；G0/G1/G2 实验计划正常输出 |
| 自加实验 1：KV head 配置 | [task2_exp1_kv_ledger.py](../code/task2_exp1_kv_ledger.py) · [json](../evidence/task2/task2_exp1_kv_ledger.json) | 实测 cache 与账本完全一致；MHA/GQA/MQA 容量 8:2:1；长上下文下 MHA 步时 2× |
| 自加实验 2：采样策略 | [task2_exp2_sampling.py](../code/task2_exp2_sampling.py) · [json](../evidence/task2/task2_exp2_sampling.json) | 8 种策略 × 6 prompt × 3 seed；长度/重复率/多样性/PPL/一致性/速度 |
| 自加实验 3：指标口径 | [task2_exp3_metrics.py](../code/task2_exp3_metrics.py) · [json](../evidence/task2/task2_exp3_metrics.json) | prompt 与 batch 两个扫描；TTFT/TPOT/吞吐/峰值显存/e2e 同口径 |
| 自加实验 3b：KV 复用收益 | [task2_exp3b_kv_reuse.py](../code/task2_exp3b_kv_reuse.py) · [json](../evidence/task2/task2_exp3b_kv_reuse.json) | 带缓存 vs 每步重算：1.5× → 6.6× → 34× |
| 实验汇总图 | [task2_exp_kv_sampling_runshot.png](../evidence/task2/task2_exp_kv_sampling_runshot.png) | 三个实验的结果合图（含结论判读） |
| 作图脚本 | [task2_runshot_figures.py](../code/task2_runshot_figures.py) · [task2_common.py](../code/task2_common.py) | 从已执行 notebook 与实验 JSON 直接渲染，数字不做手工编排 |

---

## 6. 记录：这次踩到的坑

1. **上游 21 节题目区有一处语法错误**：`decode_next_token` 里多了一行孤立的三引号（题目区第 78 行），使该 cell 无法解析。参考实现里没有这一行，解答版删除了它、函数体与参考实现一致——记录在这里，避免后来者以为是自己补错。
2. **gpt2 的 `max_position_embeddings=1024`**：做 prompt 长度扫描时，「prompt=1024 再 decode」会让位置索引越界（1024 > 1023），报的是 `device-side assert` 而不是友好的越界提示。改成 prompt+decode ≤ 1024（最大 768+32）后正常。
3. **HF eager 单流的 7.4 ms/step 不是模型算不动**：gpt2 124M fp16 权重只有约 250 MB，这个数字主要来自框架与 kernel 启动开销。所以实验 3 的结论只用于**指标口径与趋势**，不能当成 serving 性能；需要绝对值时应换 vLLM/SGLang 并单独固定口径（66 节的可选 backend 入口就是为此准备的，本机未安装 vLLM，故按节默认保持 CPU-first）。
4. **vm-60 上的 CJK 字体是 .ttc/.otf**：matplotlib 自动扫描 `fontext="ttf"` 会漏掉，导致截图里中文变成方框。显式 `addfont` 指定 SimHei 后才正常——作图脚本已经把这一步放进模块导入时执行。

## 7. 参考

- 教程：[llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)（DataWhale 社区）
  - [Part 01 · 11 KV Cache 与显存增长](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb)
  - [Part 02 · 21 解码策略](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)｜[23 投机解码](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)｜[35 多 Token 解码](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb)
  - [Part 02 · 66 推理性能比较](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)｜[68 投机解码基准](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)
  - [专题讨论 · 03 解码策略](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/inference_optimization/03_decoding_strategies.md)
- 打卡 Issue：[#165](https://github.com/datawhalechina/llm-algo-leetcode/issues/165)
- 论文：Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*（<https://arxiv.org/abs/2211.17192>）
