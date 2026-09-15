# Task0 学习笔记：请求结构与指标（推理优化 | 202609）

> 课程：[Datawhale · llm-algo-leetcode 推理优化 | 202609](https://github.com/datawhalechina/llm-algo-leetcode) · [Task0 Issue #146](https://github.com/datawhalechina/llm-algo-leetcode/issues/146)
> DataWhale 社区：https://github.com/datawhalechina · 教程地址：https://github.com/datawhalechina/llm-algo-leetcode
> 打卡选项：**(3) 4.1 + 4.2 + 4.3**
> 运行环境：vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128 · Python 3.10.12 · CPU 推理测试
> 截图证据：`evidence/task0/task0_04_attention_runshot.png` · 代码：`code/04_solved.ipynb`

---

## 4.1 Attention 与 Transformer 的关系；MHA、GQA、MLA

### Attention 在 Transformer 中的位置

Attention 不是独立于 Transformer 的东西，而是 Transformer Block 内部的核心层。一个标准 Block 可以粗略写成：

```
x → Multi-Head Attention → 残差连接 → LayerNorm → FFN(MLP) → 残差连接 → LayerNorm → 输出
```

它的作用是**信息路由**：序列中每个 token 用自己的 Query 去匹配上下文里所有位置的 Key，再按匹配度加权聚合对应的 Value。Transformer 相比 RNN 的关键优势就在这里——任意两个位置之间的信息传递路径长度是 O(1)，且整段序列可以并行计算。

我自己补全了 [Part 02 · 04 Attention（MHA/GQA）](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) 的 4 个 TODO 并跑通全部测试（运行截图见 issue 打卡 / evidence 目录）：

- TODO 1：`[B,S,H×D]` reshape + transpose → `[B,H,S,D]`（K/V 用 `H_kv`）
- TODO 2：历史 KV Cache 在 `dim=2`（seq 维）拼接，且**只缓存未扩展的 KV**
- TODO 3：`scores = Q@Kᵀ/√D` → causal mask → softmax → `@V`
- TODO 4：`[B,H,S,D] → [B,S,H×D]` → `o_proj`

测试验证了三件事：MHA/GQA 输出形状与数值有限；非法头配置（如 `hidden_dim=130, num_heads=4`）被拒绝；**完整序列最后一个 token 的输出 == 前缀 Cache + 单步 Decode 的输出**（allclose 通过），说明 KV Cache 保持计算语义、只是省掉重复计算。

### MHA / GQA / MLA：三种 KV 组织方式

| 机制 | 结构 | KV Cache 开销 | 代表模型 |
|---|---|---|---|
| MHA | 每个 Query head 独占一组 K/V head | 最大（`H` 组） | 原始 Transformer、LLaMA-2 7B |
| GQA | 多个 Query head 分组共享一组 K/V head | 居中（`H_kv` 组，LLaMA-3 8B 为 8/32=1/4） | LLaMA-3、Qwen2 |
| MQA | 所有 Query head 共享 1 组 KV | 最小，但质量损失明显 | 早期多查询模型 |
| MLA | K/V 压缩成低维潜向量缓存，用时解压 | 最小且质量好，但实现复杂 | DeepSeek-V2/V3 |

核心权衡：**KV Cache 显存/带宽 vs 模型表达能力**。实测里我们模型 `H=4, H_kv=2` 时每 token KV 参数量从 2048 降到 1024（1/2）；真实 LLaMA-3 8B 是 1/4。GQA 的工程要点是**延迟扩展（lazy expansion）**——缓存里只存 `H_kv` 组未扩展 KV，计算时才 `repeat_kv` 扩到 Query 头数。如果在缓存前就扩展，显存优势完全丢失，这也是 notebook 里 TODO 2 必须在 `repeat_kv` 之前拼接的原因。

MLA 走的是另一条路：不减少"头数"，而是把 K/V 联合投影到一个低维潜在空间，缓存只存这个潜向量，注意力计算时再通过上投影还原各头的 K/V。相当于把"组共享"换成"连续压缩"，压缩率更高，但对 kernel 实现要求也更讲究。

## 4.2 一个请求的过程：Prefill 与 Decode；TTFT 与 TPOT

### 请求生命周期

```
用户 prompt → [排队] → Prefill（整段 prompt 一次性并行前向，建立 KV Cache，产出第一个 token）
           → Decode（自回归循环：读 KV Cache → 算下一个 token → 追加进 Cache）×N
           → 输出完成（长度上限 / EOS）
```

两个阶段本质完全不同：

| | Prefill | Decode |
|---|---|---|
| 输入 | 整段 prompt（S 个 token） | 每步 1 个 token |
| 并行度 | 高（所有 prompt 位置并行算） | 无（自回归，一步一个） |
| KV Cache | 建立并写入 | 持续读取 + 每步追加 1 个位置 |
| 瓶颈 | 算力（大量 GEMM） | 显存带宽（每步都要读全部权重 + 全部 KV） |
| 对应指标 | TTFT | TPOT |

### 两个指标

- **TTFT（Time To First Token）**：从发出请求到收到第一个 token 的时间。主要由排队等待 + Prefill 时长构成，prompt 越长、当前 batch 越满，TTFT 越差。它决定用户的"响应感"。
- **TPOT（Time Per Output Token）**：第一个 token 之后平均每个 token 的生成间隔。由 Decode 单步耗时决定，近似线性累计——输出 512 token 的总延迟 ≈ TTFT + 512 × TPOT。

按教程 01 节的判断框架：**长 prompt 推高 TTFT → prefill-bound，去看 Attention kernel 和 prefill 优化；TPOT 高、生成占比大 → decode-bound，去看 KV Cache 和调度**。这正是后续 Task 的路线图。

## 4.3 用 Roofline 模型理解 Prefill 与 Decode

Roofline：算术强度 `I = FLOPs / 内存字节数` 决定性能上限——`I` 低于机器平衡点（峰值算力/峰值带宽）时是 **memory-bound**，高于时是 **compute-bound**。

- **Prefill 是 compute-bound**：一次前向处理 S 个 token，权重从显存读一次就被 S 个 token 复用，算术强度随 S 线性上升，轻松越过平衡点，GPU 算力跑满。所以 Prefill 的优化方向是**计算效率**：FlashAttention 类 kernel（减少 S² 中间矩阵的显存读写）、GEMM 调优、chunked prefill。
- **Decode 是 memory-bound**：每步只有 1 个 token，却要把**全部权重**和**全部历史 KV Cache**从显存读一遍。算术强度极低（约 1 FLOP/字节量级），GPU 大部分时间在等数据。所以 Decode 的优化方向是**访存量**：GQA/MLA 压 KV、KV 量化、PagedAttention 减碎片、连续 batching 提高权重复用。

这也解释了 4.1 与 4.2 的联系：GQA 削减的正是 Decode 阶段每步必须读取的 KV 字节数，直接改善 TPOT；而 Prefill 阶段 KV 头数影响较小，主要是 Attention 矩阵计算量。同一份权重，两个阶段的瓶颈类型不同，优化手段也必须分开评估——这正好呼应显存方向课上强调的"优化把代价转移到了哪里"：GQA 把代价转移到了 Q 头信息分辨率的轻微损失和 kernel 复杂度上。

## 实测记录

- 填全 TODO 后 `test_mha_mqa_gqa()` 输出：`Testing MHA... / Testing GQA... / Testing KV Cache Autoregressive Decoding... / ✅ All Tests Passed!`
- Cache 语义验证（Prefill 末位 vs Cache Decode）：`allclose = True`
- KV Cache 每 token 参数量（B=2, S=16, D_model=128, H=4）：MHA 2048 vs GQA(H_kv=2) 1024

## 复现方式

```bash
# vm-60: ~/task0/ 下执行
jupyter nbconvert --to notebook --execute --inplace 04_solved.ipynb
python3 make_shot.py   # 生成 task0_04_attention_runshot.png
```

## 参考

- 教程：https://github.com/datawhalechina/llm-algo-leetcode （Part 02 · 04；topic_discussion/inference_optimization/01）
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- GQA 论文: https://arxiv.org/abs/2305.13245
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- vLLM Metrics: https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md
