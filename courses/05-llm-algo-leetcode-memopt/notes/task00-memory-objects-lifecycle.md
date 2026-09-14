# Task0 学习笔记：显存对象与生命周期

> 课程：[DataWhale · llm-algo-leetcode 显存优化 | 202609](https://github.com/datawhalechina/llm-algo-leetcode) · [Task0 Issue #148](https://github.com/datawhalechina/llm-algo-leetcode/issues/148)
> 笔记基于官方教程整理与个人思考，未直接抄教程内容。
> 运行环境：vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128 · CUDA 12.8

对应学习材料（按「计算图 → backward → 状态驻留与释放」顺序）：

1. [00 · 07 PyTorch 自动求导与反向传播](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/00_Prerequisites/07_PyTorch_Autograd_and_Backward.ipynb)
2. [02 · 18 激活与损失反向](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb)
3. [02 · 17 注意力反向传播与自定义自动求导](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)
4. [训练侧显存压力（topic_discussion/02）](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/memory_performance_tuning/02_training_memory_pressure.md)

---

## 必做问答 1：为什么 forward 中间结果必须保留到 backward？峰值为什么在 backward 附近？

### 1. 计算图只记录“怎么算”，不复制“算过什么”

autograd 在前向时记录的是 Tensor 与算子的**依赖关系**（`grad_fn` 链），反向时沿这条链应用链式法则。但很多算子的局部导数依赖**前向的输入值本身**，光有“怎么算”不够，还需要“当时的输入”。例如：

- ReLU 的反向门控 `mask = (x > 0)` 依赖前向输入 `x`；
- Softmax 的反向 `dS = P ⊙ (dP − rowsum(P ⊙ dP))` 依赖前向输出 `P`；
- Attention 反向的 `dV = Pᵀ·dO`、`dQ = dS·K·scale` 依赖前向保存的 `P`、`K`、`Q`。

所以 `autograd.Function.forward` 里要 `ctx.save_for_backward(...)`，PyTorch 原生算子也一样内部保存 saved tensors。**这是问答 1 的第一层：反向公式依赖前向中间值，图记录代替不了值本身。**

### 2. 峰值为什么在 backward 附近

一次训练 step 的显存由三类对象叠加：

| 对象 | 生命周期 | 规模 |
|:---|:---|:---|
| activation（前向中间结果/saved tensors） | 从产生驻留到 **backward 消费完** | 随 batch × seq_len × hidden × 层数增长；attention 的 P 还含 N×N 项 |
| 梯度临时张量 | backward 期间不断创建 | 与参数同量级，backward 完成后写回 `.grad` 驻留 |
| 参数 + 梯度 + 优化器状态 | 跨 step 常驻 | 固定，与可训练参数量和优化器种类有关 |

backward 阶段是三者**唯一同时在场**的阶段：saved tensors 还没释放（正在被逐个消费），同时 backward 又在创建同量级的梯度临时张量，两者叠加形成整步的**显存峰值**。`opt.step()` 之后，计算图销毁、saved tensors 释放，显存回落到「参数 + 梯度 + 优化器状态」的常驻水平。

我在 18_solved.ipynb 里补了一个生命周期演示（RTX 4090D 实测）：一个完整 step（forward+backward+step）结束后，`ctx.saved_tensors` 与计算图均已释放，常驻的只有参数与梯度（约 2.1M 参数的 Linear 层，实测参数/梯度各 2099712 个，驻留到下一次 `zero_grad`）。

### 3. 哪些状态必须暂时驻留，什么时候可以释放

| 状态 | 驻留窗口 | 释放时机 |
|:---|:---|:---|
| activation / saved tensors | 前向产生 → backward 消费 | 该算子的反向执行完（整图 backward 完成即全释放） |
| `.grad` | backward 写入 → 下一次 `zero_grad` | 显式 `zero_grad(set_to_none=True)` |
| 优化器状态（如 AdamW 的 m、v） | 首次 step 创建 → 训练结束 | 训练结束 / 删除优化器 |
| 计算图（grad_fn 链） | 前向开始 → backward 完成 | `backward()` 自动销毁（retain_graph=True 例外） |

释放时机由**最后一个消费者**决定：activation 的最后一个消费者是对应算子的 backward，`.grad` 的消费者是优化器更新，更新完即可 `zero_grad`。优化器状态则是真正的“跨 step 常驻”，任何换显存策略（offload/量化）动它之前，要先确认它确实是大头——这正是训练侧显存压力一节说的「先确认主因，再选策略」。

### 4. Attention backward 为什么可能需要保存 Q、K、V 和 softmax 概率

沿依赖链 $dO \rightarrow dV、dP \rightarrow dS \rightarrow dQ、dK$ 逐个看反向公式的消费者：

- $dV = P^T \cdot dO$ —— 消费 **P**；
- $dP = dO \cdot V^T$ —— 消费 **V**；
- $dS = P \odot (dP - \text{rowsum}(P \odot dP))$ —— 再次消费 **P**；
- $dQ = dS \cdot K \cdot \frac{1}{\sqrt d}$，$dK = dS^T \cdot Q \cdot \frac{1}{\sqrt d}$ —— 消费 **K、Q**。

每一步反向公式都直接消费前向值，所以显式 Attention 路径的 `ctx.save_for_backward(q, k, v, p)` 一个都省不掉。其中 **P 的形状是 B×N×N**（含两个序列维），batch 或序列变长时它的保存容量和 HBM 读写都被二次放大，是 attention 反向显存压力的放大器。我在 17_solved.ipynb 补全了四个 TODO 并通过 gradcheck 与 PyTorch 自动求导对照（B=2, N=8, d=16, float64）。

**进阶：能不能不存 P？** 可以——把 softmax 反向化成逐行修正项后，只需要重算时能恢复 P 或其统计量，这正是 FlashAttention 分块计算 + online softmax 的动机：用重算换保存，把 B×N×N 的中间状态从显存里消掉。

## 必做问答 2：为什么增大 batch 或 seq_len 可能提高 activation 峰值？

activation 显存 ≈ Σ(每层保存的中间结果) ≈ 层数 × batch × seq_len × hidden × dtype 字节（线性层的输入/门控类中间值），所以：

- **batch ↑**：每个 micro-batch 的所有 activation 等比放大。注意梯度累积（accumulation）只改变多少个 micro-step 汇总成一次 update，**不会降低单个 micro-batch 的 activation 峰值**，参数/梯度/优化器状态也分毫不变——这是训练侧显存压力一节的关键提醒。
- **seq_len ↑**：线性部分随 S 线性放大；attention 的 scores/P 是 B×N×N 项，**随 S 二次增长**，长序列下成为大头。

我在 RTX 4090D 上实测了 Linear+GELU 网络（无 attention 项，纯线性部分）：

| 配置 | 峰值 | 观察 |
|:---|:---|:---|
| B=4, S=128 | 41.3 MiB | 基准 |
| B=8, S=128 | 59.3 MiB | batch 翻倍，峰值 +18 MiB（线性部分等比放大） |
| B=16, S=128 | 87.3 MiB | batch ×4，峰值 +46 MiB |
| B=4, S=256 | 59.3 MiB | S 翻倍，峰值与 B=8,S=128 完全相同 |
| B=4, S=512 | 87.3 MiB | S ×4，峰值与 B=16,S=128 完全相同 |

**实测印证**：在这个无 attention 的网络里，activation 只含线性项，B 和 S 在乘积 B×S 上的地位完全等价（B×8 与 S×4 翻倍都让峰值 +18 MiB）。若换成带显式 attention 的网络，S 增大还会额外引入 B×N×N 的二次项，seq_len 的放大效应会更陡。

## 训练侧显存压力有哪些（对应 02 训练侧显存压力）

训练时占显存的对象分四类，每类的"旋钮"和"代价转移"不同：

| 压力源 | 由什么决定 | 什么时候成为大头 | 能动它的策略 |
|:---|:---|:---|:---|
| **activation 峰值** | micro-batch × seq_len × hidden × 层数；attention 额外含 B×N×N 项 | 大 batch / 长序列 / 深网络的前向-反向期间 | 缩小 micro-batch、activation checkpointing（重算换保存）、offload |
| **梯度** | 参数量 × dtype | 参数量大时，与参数同量级 | 梯度分片（ZeRO-1 类）、低精度梯度 |
| **optimizer state** | 优化器种类（AdamW 的 m+v 是参数的 2 倍）× dtype | AdamW + 大参数量时，常超过参数本身 | 换 SGD/8bit 优化器、offload、参数分片（ZeRO） |
| **输入与临时张量** | 输入规模、实现的临时分配 | tokenization/embedding 输入、大张量搬运 | 融合算子减少临时、pin memory / 分块搬运 |

两个关键判断（教程原文的观察动作表）：
- **加 accumulation 不减显存**：它只减少"单次 update 的样本数含义"，单步 activation 峰值、参数、梯度、optimizer state 都不变，代价是微步数变多、时间变长；
- **OOM 出现阶段指向压力源**：forward 中 OOM → activation/临时张量；backward 中 OOM → saved tensors + 梯度临时；step 后 OOM → optimizer state 首次分配。先固定 workload 确认主因，再选 checkpoint / offload / sharding / 量化。

## 实验证据

- `evidence/task0/task0_18_runshot_final.png` —— 第18节运行结果 + batch/seq_len 扫描（4.2 截图）
- `evidence/task0/task0_17_runshot.png` —— 第17节运行结果 + Q/K/V/P saved tensors 账本与 P 随 N 二次增长曲线（4.1 截图）
- `code/17_solved.ipynb` —— 17 节 Attention backward 解答版（gradcheck + 自动求导对照通过，vm-60 实跑输出）
- `code/18_solved.ipynb` —— 18 节解答版 + 附加「生命周期演示 & batch/seq_len 扫描」cell（vm-60 实跑输出）

## 一句话总结

**前向图记录依赖，反向公式消费前向值**——所以 activation 必须从产生驻留到 backward 消费完；backward 阶段 saved tensors 与梯度临时张量同时在场形成峰值；peak 结束后一切回落到参数+梯度+优化器状态的常驻水平，而任何换显存的策略都要先确认压力主因，再看它动的是哪一类对象。
