### llm-algo-leetcode 推理优化 | 202609 Task2 · 单请求 Decode 与生成策略

微信群昵称：（填写）
GitHub ID：wsm000
打卡选择：**(3) 4.1 + 4.2 + 4.3**
运行环境：vm-60 · NVIDIA RTX 4090 D（Ada sm89，24 GB）· torch 2.9.1+cu128 · transformers 5.16.1 · Python 3.10.12

学习笔记：https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/notes/task02-single-request-decode-and-generation-strategies.md
补全并跑通的 notebook：[Part 01 · 11](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/11_solved.ipynb) · [Part 02 · 21](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/21_solved.ipynb) · [Part 02 · 23](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/23_solved.ipynb) · [Part 02 · 35](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/35_solved.ipynb) · [Part 02 · 66](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/66_solved.ipynb) · [Part 02 · 68](https://github.com/wsm000/mlsys-learning-notes/blob/main/courses/06-llm-algo-leetcode-inference/code/68_solved.ipynb)
额外实验：KV 账本与 head 配置 · 采样策略 · 指标口径 · KV 复用（脚本与 JSON 见 [evidence/task2](https://github.com/wsm000/mlsys-learning-notes/tree/main/courses/06-llm-algo-leetcode-inference/evidence/task2)）

---

#### 4.1 最小打卡

#### (1) KV Cache 增长与哪些因素有关？近似账本公式，以及为什么是线性不是平方

**账本公式**：`KV Cache Bytes \approx 2 \times L \times B \times H_{kv} \times D \times S \times \text{dtype_bytes}`

`L` 层数、`B` batch/并发数、`H_{kv}` KV 头数（不是 query 头数）、`D` head_dim、`S` 已缓存 token 数、前导 `2` 表示 K 和 V 各一份。

**为什么线性**：每个新 token 对缓存的贡献是常数 `2 \times L \times H_{kv} \times D \times \text{dtype_bytes}`（与 `S` 无关），`S` 在公式里只出现一次，所以对 `S` 是常数导数。平方项 `S^2` 属于**另一个对象**——attention 的 score 矩阵 `QK^T`，它是 prefill 阶段的中间结果，用完即弃、不常驻。**KV Cache 是「历史状态成本」（线性、常驻），score 矩阵是「中间计算成本」（平方、瞬时）**，Task1 的 FlashAttention 消的是后者。

**实测验证**（合成 LLaMA 12 层 / hidden 1024 / GQA 2 KV 头 / fp16，只改 S 与 B）：

| 变化项 | 实测 | 账本 | 每 token |
|---|---:|---:|---:|
| S=512, B=1 | 6.0 MiB | 6.0 MiB | 12288 B |
| S=2048, B=1 | 24.0 MiB | 24.0 MiB | 12288 B |
| S=4096, B=1 | 48.0 MiB | 48.0 MiB | 12288 B |
| S=8192, B=1 | 96.0 MiB | 96.0 MiB | 12288 B |
| S=2048, B=8 | 192.0 MiB | 192.0 MiB | — |

序列翻倍缓存翻倍、batch 翻倍缓存翻倍，**每 token 字节恒定**——线性的实测形态；实测与账本完全一致（误差 0）。

#### (2) 针对 KV Cache 压力可以从哪些层面优化？各自减少什么、新增什么代价

| 层面 | 代表机制 | 减少的对象 | 新增代价 |
|---|---|---|---|
| 表示压缩 | MQA / GQA / MLA | `H_{kv}`（每份 K/V 服务更多 query 头），MLA 压到 latent | 训练配方/表达能力调整，MLA 有额外投影 |
| 缓存组织 | PagedAttention | 连续大块预留与内部碎片 | 页表管理、非连续访存、kernel 复杂度 |
| 缓存复用 | Prefix Cache | 跨请求重复前缀的 prefill 计算与 KV 写入 | 额外缓存显存、淘汰策略、匹配开销 |
| 数据精度 | KV 量化（FP8/INT8） | dtype_bytes | 反量化开销与精度损失 |
| 执行粒度 | Chunked Prefill / PD 分离 | 单次峰值与相互干扰 | 调度开销、块边界效率 |
| 历史长度 | 滑窗 / Sink / 稀疏注意力 | `S` | 长程信息丢失 |

实测 MHA/GQA/MQA（同骨架只改 `num_key_value_heads`，S=4096、B=1）：192.0 / 48.0 / 24.0 MiB，即 8:2:1，20 GiB 预算下并发数 106 / 426 / 853。**先买到的是容量与并发**，不是单步延迟。

![Part 01 · 11 KV Cache 与显存增长运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task2/task2_11_kv_cache_runshot.png)

#### (3) Greedy、Temperature、Top-k、Top-p 各改变了什么？为什么温度高既增多样性又降稳定性

四者都作用在 logits 之后、采样之前：greedy 改**选择规则**（argmax），temperature 改**分布平坦程度**（`z/T`，单调变换不改排序），top-k 改**候选集合大小**，top-p 改**候选集合的累计质量**（边界 token 保留）。它们都不改模型前向计算。

真实生成实测（gpt2 124M，6 prompt × 3 seed，max_new=100）：

| 策略 | rep2（重复率） | distinct-2 | 自困惑度 PPL（std） | 与 greedy 一致率 |
|---|---:|---:|---:|---:|
| greedy | 0.684 | 0.095 | 2.23 (0.57) | 1.000 |
| T=0.7 | 0.166 | 0.703 | 6.28 (1.40) | 0.026 |
| T=1.0 | 0.070 | 0.838 | 12.47 (2.77) | 0.015 |
| T=1.5 | 0.022 | 0.918 | 33.88 (7.11) | 0.007 |
| top_k=10 | 0.141 | 0.760 | 7.30 (1.18) | 0.024 |
| top_p=0.9 | 0.113 | 0.795 | 8.96 (3.03) | 0.014 |

**多样性上升**：除以 `T>1` 压平 logits 差距 → softmax 后熵变大 → 长尾 token 也获得可采样概率 → 不同 seed 更容易走到不同分支。**稳定性下降**：同一 prompt 在不同 seed 下的输出分岔，跨 seed PPL 标准差 0.57 → 7.11，与 greedy 逐 token 一致率 1.000 → 0.007。

顺带一个必须点破的陷阱：greedy 的 PPL 最低（2.23）**恰恰因为它在重复自己**（rep2 0.684）——**PPL 低 ≠ 质量高，必须和重复率一起读**。另外 8 种策略的解码速度几乎一样（117–127 tok/s），**采样开销可忽略，代价全在输出分布上**。

#### (4) 如何理解投机解码？draft 与 target 的职责

用便宜的小 draft 模型先连猜 `K` 个 token，再让大 target 模型**一次并行验证**这 `K` 个位置，把 `K` 次串行前向压成 1 次。draft 负责**推进长度**（产出 `draft_tokens` 与 `draft_probs`），target 负责**正确性**（产出 `target_probs`，前 `K` 行验证、第 `K` 行给 bonus）。

单轮四步：提议 → 按 `\alpha(x)=\min(1,p(x)/q(x))` 验证 → 首次拒绝时从在 residual `p'(x) \propto \max(p(x)-q(x),0)` 上采样**修正 token** 并结束本轮 → 全部接受时追加 **bonus token**。接受-拒绝采样 + residual 补齐保证了**最终分布仍等于 target 分布**，所以是无损加速。题目区测试覆盖了四条路径，还钉住了两个边界：`q=0` 且 `p>0` 直接接受（否则除零误拒），residual 质量为 0 必须显式报错。

收益条件是三件事同时成立：**接受率高、verify 不贵、draft 足够便宜**——这正是 68 节的判定顺序（`quality_ok → acceptance_rate → throughput_gain → verify_cost_delta`）。

![Part 02 · 21 解码策略运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task2/task2_21_decoding_runshot.png)
![Part 02 · 23 投机解码运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task2/task2_23_speculative_runshot.png)

---

#### 4.2 增项 1

**(1) 什么叫多 token 解码？** 在一次解码步里尝试推进多个 token：先批量提议候选前缀，再由目标侧验证，连续接受的候选前缀被一次性确认，遇到第一个拒绝就停止并把后面的候选当作回退后缀丢弃/重生成。动机与投机解码相同（减少逐 token 往返的 KV 追加、kernel 启动与调度开销），强调点是「单轮产出多个」而非「由谁提议」。

**(2) 与投机解码在候选生成与目标验证上的区别**

| 维度 | 投机解码 | 多 token 解码 |
|---|---|---|
| 候选生成 | 独立 draft 模型采样，候选分布 `q \ne p` | 同一解码步内推进多个候选，可不引入第二个模型 |
| 目标验证 | `\min(1,p/q)` 接受-拒绝，**保证分布等价于 target** | 阈值/一致性规则（本节为 `target \ge draft \times ratio`），**只保证控制流正确** |
| 增量成本 | 多一个 draft 模型 + 每步 `K` 次前向 | 多一次提议与验证 |
| 失败处理 | residual 采样修正 token | 该位置起成为回退后缀 |

一句话：**投机解码的核心是分布校正（无损），多 token 解码的核心是单轮吞吐（控制流）**；共同难点都是接受率、验证成本与首次拒绝后的回退。

**(3) 一轮中候选、验证结果与状态更新的关系**：① `propose` 截取候选前缀（决定本轮最多验证几个位置）→ ② `verify` 从左到右短路验证，首个拒绝写入 `rejected_at`，**其后候选不再验证**（它们的合法性依赖被拒前缀）→ ③ `decode` 汇总接受前缀、`accepted_len`、`progress_per_round` 与回退后缀 → ④ 全接受时 `rejected_at=None`、后缀为空、推进比例 1.0。测试里 `max_proposal_len` 由 3 改为 1/2/4，接受数随之变化，把这条依赖量化了；`progress_per_round` 只是本轮接受比例，**不是吞吐收益**。

![Part 02 · 35 多 Token 解码运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task2/task2_35_multi_token_runshot.png)

---

#### 4.3 增项 2（4 选 2，本次做了 3 个）

**(1) KV head 配置的最小实验**：固定骨架（12 层 / hidden 1024 / 8 query head / head_dim 128 / fp16），只改 `num_key_value_heads` ∈ {8, 2, 1}；用 `use_cache=True` 前向后按 `data_ptr` 去重求和得到实测 cache 字节，与账本对账；利用率用「20 GiB 预算下并发请求数」表达；性能固定 B/S 测 prefill、decode ms/step 与 tok/s。

| 配置 | KV 头 | cache(S=4096,B=1) | 每 token | 20 GiB 并发 | decode ms/step(B=8,S=1024) |
|---|---:|---:|---:|---:|---:|
| MHA | 8 | 192.0 MiB | 48.0 KiB | 106 | 11.15 |
| GQA | 2 | 48.0 MiB | 12.0 KiB | 426 | 11.38 |
| MQA | 1 | 24.0 MiB | 6.0 KiB | 853 | 11.25 |

**判读**：短上下文下容量差 8 倍但步时几乎不变（权重读取与 kernel 启动主导）；**到 S=8192、B=16 时 MHA 步时 22.58 ms vs GQA/MQA 11.3 ms（2×）**，峰值显存 15642 vs 11034 vs 10266 MiB——此时每步要读的 KV 达到 GB 量级，decode 从「算不动」变成「读不动」。KV head 配置既是容量旋钮，也是长上下文下的延迟旋钮。

**(2) 采样策略的最小实验**：固定模型（gpt2）、prompt 集（6 条）、`max_new_tokens=100`、seed 集合（3 个），只改解码策略；记录生成长度、重复率、distinct-n、自困惑度、与 greedy 一致率、速度（结果表见 4.1(3)）。结论：温度升高 → 重复率单调降、多样性单调升、质量单调差、跨 seed 波动单调变大；长度在此设置下几乎不变（只有 5.6% 提前 EOS），想控长度应直接设 `max_new_tokens`；速度几乎不受影响。

**(3) 为什么同时记录 TTFT、TPOT、吞吐、峰值显存、端到端延迟**（gpt2，gen=32，warmup=2 取 3 次中位数）：

| 场景 | TTFT | TPOT | e2e | 吞吐 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| prompt 32, B=1 | 8.60 ms | 7.37 ms | 237.1 ms | 134.9 tok/s | 259 MiB |
| prompt 768, B=1 | 9.96 ms | 7.45 ms | 240.9 ms | 132.9 tok/s | 386 MiB |
| prompt 256, B=16 | 18.80 ms | 7.75 ms | 259.1 ms | 1976.0 tok/s | 959 MiB |
| prompt 256, B=32 | 37.91 ms | 7.83 ms | 280.7 ms | 3648.2 tok/s | 1662 MiB |

- **prompt 变长**：TTFT 上升、TPOT 不动 → 只看吞吐会误判「没变化」；
- **batch 变大**：吞吐 132 → 3648 tok/s、显存 297 → 1662 MiB，代价是 TTFT 从 9.7 → 37.9 ms，而 TPOT 几乎不动 → **吞吐收益是用 TTFT 与显存换来的**；
- **端到端** `\approx` TTFT + TPOT × 31 成立，且短输出时 TTFT 只占 4%，长输出时 TPOT 才是大头——少一个指标就会得出相反结论。

补充「为什么必须有 KV Cache」（合成 LLaMA，只切换是否复用 KV）：B=8/S=512 → TPOT 17.0 → 11.2 ms（1.5×）；B=8/S=2048 → 75.1 → 11.3 ms（6.6×）；B=16/S=4096 → 380.8 → 11.2 ms（**34×**，e2e 6.05 s → 0.50 s）。**用线性增长的常驻显存，换掉每步随上下文线性增长的计算。**

**(4) baseline 与 speculative candidate 的统一实验**（68 节框架）：固定 target 模型与版本、prompt 分布、batch / max_new_tokens / temperature / top_p、warmup 与重复次数、统计口径；变量只改「是否启用 draft + proposal 长度」。分组 G0 = target 单独跑，G1 = target + draft，G2a/G2b = 同一链路只改 `proposal_length`（3 vs 8）找拐点。同时记录 TTFT、TPOT、吞吐、峰值显存、acceptance rate（逐轮分布）、draft/verify cost 与质量门槛；判定顺序为先质量、再接受率、再吞吐、最后 verify 成本，输出 `accept / tune / reject` 与下一步动作。**纪律：加速比必须与 acceptance rate 一起报告。**

![Part 02 · 66 推理性能比较运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task2/task2_66_inference_metrics_runshot.png)
![Part 02 · 68 投机解码基准运行结果](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task2/task2_68_speculative_benchmark_runshot.png)
![自加实验汇总：KV 账本 · 生成策略 · 指标口径](https://raw.githubusercontent.com/wsm000/mlsys-learning-notes/main/courses/06-llm-algo-leetcode-inference/evidence/task2/task2_exp_kv_sampling_runshot.png)

---

**证据与脚本**：6 个解答版 notebook（已在 vm-60 实机执行，测试全部通过）、4 个自加实验脚本与 JSON、7 张运行截图、全部运行日志，均在 [evidence/task2](https://github.com/wsm000/mlsys-learning-notes/tree/main/courses/06-llm-algo-leetcode-inference/evidence/task2)。

**顺带记录的一处上游问题**：Part 02 · 21 题目区的 `decode_next_token` 里多了一行孤立的三引号，导致该 cell 无法解析（参考实现里没有这一行）；解答版删除了它，函数体与参考实现一致。
