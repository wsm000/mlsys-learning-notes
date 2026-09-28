# Task4：推理侧 KV Cache 与显存容量 —— 学习笔记

- GitHub ID：wsm000 ｜ 微信群昵称：（填写）
- 打卡选择：**(4) 4.1 + 4.2 + 4.3**（全量）
- 运行环境：vm-60 · NVIDIA RTX 4090 D（24 GB）· torch 2.9.1+cu128 · transformers 5.16.1 · Python 3.10.12
- 教程与社区：[datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) ｜ [DataWhale 社区](https://github.com/datawhalechina) ｜ [打卡 Issue #177](https://github.com/datawhalechina/llm-algo-leetcode/issues/177)
- 对应节：Part 01 · 11（KV Cache 与显存增长）、Part 02 · 22（vLLM PagedAttention）、Part 02 · 34（Prefix Cache 匹配与复用）、topic 04（推理 Cache 与显存预算）、Part 02 · 24（RadixAttention）、Part 02 · 66（推理性能对比）、Part 02 · 69（Prefix Cache 基准）

> 本篇为个人学习笔记，全部实验代码与实测数据在 [../evidence/task4](../evidence/task4)；观点为原创整理，非教程原文照抄。Issue 截止时间为 09-26 12:00，本篇为截止后补交（issue 仍处于 open 状态，同日程仍有其他同学在评论打卡）。

---

## 0. 一句话总览

按教程推荐的"增长 → 分页 → 复用 → 容量验证"顺序，我把这个 Task 理解为四个递进的问题：

1. **增长**：KV Cache 字节数 = `2 × layers × kv_heads × head_dim × tokens × dtype_bytes × batch`，六个维度全部严格线性（实验1）；
2. **分页**：连续预分配把"最多可能用多少"锁进显存，分页只锁"现在用多少"——同一请求分布下实测节省 89.3% token（实验2）；
3. **复用**：共享前缀跳过 prefill 计算、共享 KV block 减少重复驻留，但必须从第 0 个 token 连续匹配（实验3）；
4. **容量验证**：把权重、运行缓冲、安全余量一起扣掉得到 KV 池，反推各表示变体的最大并发；用准入压力实测验证规划（实验4）。

一句话：**KV Cache 是容量问题，不是算子问题**。四个机制分别攻击容量边界的不同侧面：分页减少浪费、复用减少重复、表示变化减少单价、预算与准入决定边界画在哪。

---

## 1. 4.1 最小打卡

### 1.1 KV Cache 的增长关系：哪些维度线性？（对应 Part 01 · 11）

账本公式（与教程 11 节一致）：

```
KV bytes = 2(K,V) × layers × kv_heads × head_dim × tokens × dtype_bytes × batch
```

我用两种方式验证（实验1，[task4_exp1_growth.json](../evidence/task4/task4_exp1_growth.json)）：

- **账本 vs 真实分配**：4 组配置（不同 B/S/H_kv/head_dim/dtype）在 GPU 上真实分配 K/V 张量，`numel × element_size` 与公式**逐字节相等**；
- **倍增实验**：六个维度各自翻倍，KV 字节数比率**恰好都是 2.0**（log2-log2 斜率 = 1）。

所以对 batch size、上下文长度（tokens）、层数、KV heads、dtype 字节数（head_dim 同理）**全部严格一阶线性**，没有任何平方项——attention 计算量随 S 平方增长，但 KV 显存随 S 线性增长，这正是长上下文下显存先于算力成为瓶颈的原因。

真实模型实测（12 层 / hidden 1024 / 8 heads，B=4，S=512，fp16，prefill 后）：

| 配置 | KV cache 实测 | 账本 | 每 token |
|---|---:|---:|---:|
| MHA（kv=8） | 96.0 MiB | 96.0 MiB | 192 KiB |
| GQA-4 | 48.0 MiB | 48.0 MiB | 96 KiB |
| GQA-2 | 24.0 MiB | 24.0 MiB | 48 KiB |
| MQA（kv=1） | 12.0 MiB | 12.0 MiB | 24 KiB |

decode 每步新增字节 = `per_token × B`（实测逐步核验）。**KV heads 是唯一"改结构就能线性压缩"的维度**——这就是 GQA/MQA/MLA 的显存侧动机。

### 1.2 连续预分配：为什么"用得少"也 OOM？PagedAttention 如何缓解？（对应 Part 02 · 22）

连续预分配按 `max_seq_len` 为每个请求预留一段**连续**显存，产生三重浪费：预留尾部（实际 300 token 按 4096 留）、外部碎片（总量够但凑不出大连续块）、不可共享（相同前缀各存一份）。

我在 4090D 上做了一个**构造性外部碎片**实验（实验2 B）：把显存填到接近上限后释放交错的 8 MiB 洞，空闲总计 **9.35 GiB**，此时申请 **4.63 GiB 的单个连续块失败（CUDA OOM）**，而按 8 MiB 粒度逐块申请同样的总量**成功**。这就是"总空闲量充足仍分配失败"的真实复现——碎片让"够用"变成"不能用"。

同一请求分布（200 个请求，长度 16…4095，seed 固定，每条含短/中/长混合）下两种策略对比（全模型每 token 48 KiB 换算）：

| 策略 | 分配 token | KV 显存 |
|---|---:|---:|
| 连续预留（按 S_max=4095/请求） | 819,000 | 38,390.6 MiB |
| 按需分页（block=16） | 87,488 | 4,101.0 MiB |

**节省 89.3% token**，尾块浪费仅 1,448 token（1.7%）。PagedAttention 把"预留策略"换成"分配策略"：逻辑连续、物理离散、按需增长，还有块表引用计数带来的前缀共享能力（实验2 A 验证了 refcount 共享共享块只付一份物理存储、最后一个引用释放才归还）。

用块表顺序 gather 回逻辑连续序列后，注意力输出与连续存放**逐位一致**（max_diff = 0.0，实验2 E）——物理离散不改变计算结果，这是分页能成立的前提。

### 1.3 block size 的三重影响（对应 Part 02 · 22）

block size 在"元数据开销 / 内部碎片 / 调度灵活性"之间做交易，我用同一 trace 做了扫描（实验2 D，全模型每 token 48 KiB）：

| block size | 块表条目 | 元数据（8B/条） | 内部碎片 token | 碎片占比 |
|---:|---:|---:|---:|---:|
| 1 | 86,040 | 688 KB | 0 | 0% |
| 16 | 5,468 | 43.7 KB | 1,448 | 1.7% |
| 32 | 2,777 | 22.2 KB | 2,824 | 3.3% |
| 128 | 792 | 6.3 KB | 15,336 | 17.8% |
| 256 | 466 | 3.7 KB | 33,256 | 38.7% |

- **不能无限小**：bs=1 时块表条目 = token 数，元数据与"软件 MMU"查表的间接开销压倒碎片节省，kernel 按块寻址的效率也崩塌；
- **不能无限大**：bs=256 时尾块浪费 38.7% token（本例 1.6 GB KV），退化回连续预留，分页失去意义；
- **调度粒度**：块越小，分配/回收/共享/驱逐的粒度越细，前缀匹配也能按更小块命中（实验3 中 64 token 块使半个 prompt 的偏移仍可复用）。

工程上 16~128 是常见折中，本质是让"尾块浪费期望（约半个块）"远小于"块表与寻址开销"。

### 1.4 Prefix Cache 复用了什么？降计算还是降显存？为什么必须连续？（对应 Part 02 · 34）

我实现了块哈希链前缀缓存（key = (parent_hash, block)，与 vLLM/SGLang 一致）并做了三层验证（实验3）：

**复用了什么**：命中前缀在**所有层**已经算好的 K/V 状态及其物理块映射；新请求跳过这些 token 的 QKV 投影、attention 打分、MLP 与 logits，只 prefill 未命中后缀。Q 不复用（Q 属于当前 token）。

**降 Prefill 计算是确定的**：prompt=4096 + suffix=128 的真实模型对照（实验3 B1）：整段 prefill 22.8 ms → 只算 suffix 12.0 ms（**TTFT -47.6%**），跳过了 96.97% 的 prefill token。logits 数值等价：复用路径与整段 prefill 的末位 logits 最大差 0.005（fp16 噪声量级）。

**也可能降 KV 显存，取决于是否共享物理块**：6 个请求共享 512 token system prompt 的账本 + 真实分配探针（实验3 C）：重复存储 180 MiB → 共享后 60 MiB（**-66.7%**）。但缓存本身占显存，所以**总显存不一定降**——这是 4.2(3) 的引子。

**为什么必须匹配连续前缀**：因果注意力下第 i 个 token 的 KV 依赖从 x₁ 到 xᵢ 的全部历史与 RoPE 位置。块哈希链要求从第 0 块起父子链完整匹配：首段不同 → 命中 0（TTFT 退回整段 13.7 ms）；中段不同 → 只命中分叉前的共享段（256/512 token）。同样的 token 串出现在不同位置也会因父哈希不同而不命中（链式哈希天然编码了位置）。

---

## 2. 4.2 增项 1（对应 topic 04 与 Part 02 · 24）

### 2.1 容量规划：权重 + 运行缓冲 + KV 池 + 安全余量（对应 topic 04）

规划式：

```
KV 池 = 总显存 × gpu_memory_utilization − CUDA 上下文 − 权重 − 运行缓冲峰值 − 安全余量
最大 token 容量 T = KV 池 ÷ 每 token KV 字节
固定上下文求最大并发 B_max = T ÷ S；固定并发求最大上下文 S_max = T
```

在 4090D（24 GB）上用**真实 1.5B 级 LLaMA 形状模型**测量前三项（实验4 A）：权重 3.09 GiB（param 计数字节实测）、prefill+decode 运行缓冲峰值 189 MiB、上下文约 0.6 GiB、util 0.9、安全余量 256 MiB → **KV 池 17.65 GiB**。各表示变体：

| 变体 | 每 token | 最大 token | S=4096 最大并发 |
|---|---:|---:|---:|
| MHA（kv=16，24L 实测） | 192 KiB | 96,397 | 23 |
| GQA-8 | 96 KiB | 192,794 | 47 |
| MQA（kv=1） | 12 KiB | 1,542,357 | 376 |
| MLA（512+64 latent） | 27 KiB | 685,492 | 167 |
| GQA-8 + FP8 KV | 48 KiB | 385,589 | 94 |

8B 级账本对照（LLaMA-3-8B 形状，32 层，fp16）：MHA 512 KiB/token → 36,149 token（S=4096 时 8 并发）；GQA-8 128 KiB → 144,596（35 并发）。与其他同学在 A100 80GB 上的同类计算口径一致（他们 60 GB KV 池 / 128 KiB = 491,520 token / 4096 ≈ 120 并发），差别全在池子大小，公式同源。

要点：**容量规划是四个数的减法，任何一个数估错，反推的并发上限就错**。权重和运行缓冲可以实测，CUDA 上下文和碎片只能靠余量兜底；MLA/量化行的 kernel 级收益要在真实 backend 复测，账本只给"表示变化"的上界。

### 2.2 RadixAttention  vs 固定 Prefix Cache（对应 Part 02 · 24）

我用压缩基数树（边存 token 段、terminal 标记完整路径、部分重叠边分裂）复现了 24 节机制（上游 notebook 补全执行，测试通过）。多分支场景是它相对"固定前缀列表"的核心优势：

- 请求 A=[stem,B]、B=[stem,C]、C=[stem,C,D]：树只存一条 stem 共享边，分支按需分裂，匹配沿最长公共前缀向下；
- 固定 Prefix Cache（34 节的 list 版）每条登记前缀独立存储，多分支各自重复登记公共部分。

代价也真实存在：树的分裂/合并、匹配时的逐边比较、LRU 驱逐要处理节点引用关系（实验3 A 的 LRU 测试钉住"被引用块不驱逐"）。24 节教学版只把 terminal 节点计入可复用长度，真实 SGLang RadixCache 每节点携 KV、不要求 terminal——读实现时不能混。

### 2.3 为什么命中率不够：TTFT、峰值显存、维护开销与 P95/P99（对应 Part 02 · 69）

命次率只说明"缓存有没有被用到"，不说明收益与代价。我的三组实测（实验3 + 实验4 C）：

1. **命中率 66.7% 但 TTFT 零收益**：prompt=512、6 请求，prefix cache 路径均值 12.18 ms vs 整段 prefill 11.96 ms（实验3 B2）——小 prompt 的 prefill 被 kernel 启动等固定开销主导，跳过的计算不变壁钟。同期 prompt=4096 时收益 -47.6%（B1）。**同一个机制，规模不同，服务结论不同**。
2. **P99 比均值更暴露问题**：同 workload（12 请求共享前缀 + 长尾）下，无界 prefix cache 命中率 1.0，但 TTFT P99 从 baseline 15.55 ms 恶化到 31.52 ms，峰值 KV 从 30 MiB 涨到 54 MiB（缓存常驻 + 深拷贝隔离开销）→ **reject**；LRU 封顶版命中率 0.917，TTFT 均值 12.4 ms、P99 12.89 ms 都是最优，但峰值同样 +24 MiB → **tune**（调容量/驱逐）。
3. **维护开销要单独计量**：deepcopy 隔离 2.9 ms/请求（我的实现为数值隔离付的代价；真实系统用 COW/块共享规避，但块表与驱逐逻辑仍在）。

---

## 3. 4.3 增项 2（对应 Part 02 · 66 与 Part 02 · 69）

### 3.1 66 的 baseline 如何为 69 提供共同口径（对应 Part 02 · 66、69）

66 节把一次推理测量收敛为同一 workload 契约：模型名/backend、batch、prompt/generated token 数、dtype、cache policy、并发、预热与重复次数、超时。69 节在此基础上只改 cache policy 一个变量。我按此口径固定了这些字段：模型（toy-12L-1024）、请求数 12、prompt 512、suffix 128、gen 24、dtype fp16、到达顺序 sequential、同一 seed 集；每组分别记录 TTFT（mean/P95/P99）、TPOT、吞吐、峰值 KV、命中率。

必须固定的字段缺一不可：模型 revision 与 dtype（决定权重与 KV 单价）、prompt/gen 分布（决定工作集）、并发与到达（决定排队与批大小）、预热与统计口径（决定尾延迟可信度）。**"同一 workload"不是口号，是可逐项核对的清单。**

### 3.2 TTFT 改善但 peak 上升、P99 变差的候选：accept / tune / reject？

这正是我实测到的场景（实验4 C）：无界 prefix cache 候选——命中率 1.0（最高），但相对 baseline：TTFT 均值 +0.93 ms（反而更差，维护开销吃掉收益）、P99 +15.97 ms、峰值 KV +24 MiB（+80%）。若只看命中率或只看 TTFT 均值会误判；把三者放一起，结论是 **reject**（缓存收益为负，还放大尾延迟与驻留）。同 workload 的 LRU 封顶候选：TTFT -0.71 ms、P99 最优，但峰值仍 +24 MiB → **tune**（缩小缓存容量上限或改驱逐策略，把峰值收益换回来）。

一般规则：**先看质量/正确性门槛，再看峰值显存是否在预算内，再比 TTFT/吞吐，最后看 P95/P99 是否被维护开销污染**；TTFT 与峰值同时恶化的候选没有调参价值，TTFT 改善但峰值/尾延迟恶化的候选进入容量-驱逐联调。

### 3.3 显存不足时：PagedAttention / Prefix Cache / KV 量化 / GQA 各自解决什么？

| 机制 | 解决的容量问题 | 我的实测证据 | 代价转移到 |
|---|---|---|---|
| PagedAttention | 预留浪费与外部碎片（分配方式） | 同 trace 省 89.3% token；构造性碎片下连续分配 OOM 而分页成功 | 块表元数据、非连续访存、尾块碎片 |
| Prefix Cache | 重复前缀的重复计算与重复驻留（复用） | TTFT -47.6%（prompt=4096）；KV 存储 -66.7% | 缓存驻留显存、失效管理、匹配/驱逐开销 |
| KV 量化（FP8/INT8） | 每 token KV 单价（表示宽度） | 真实 K/V：FP8 字节减半、max err 0.123；INT8 max err 0.012、rel L2 0.007 | 数值误差、反量化 kernel 开销 |
| GQA/MQA（及 MLA） | 每 token KV 单价（结构） | per-token 192→12 KiB（MHA→MQA），准入实测峰值 6.94→0.43 GiB | 需训练/结构改动，质量风险另评 |

**可组合**：分页是物理层（块表 + 池），前缀复用与量化是策略/表示层，真实系统都是"分页 block 做管理 + 前缀索引做复用 + 可选 INT8/FP8 存储"的组合；GQA/MLA 是训练期的结构选择，上线后与前三者正交叠加（GQA-8 + FP8 = 48 KiB/token，容量 ×4）。

**不可相互替代**：prefix cache 命不中重复流量就没收益（它治"重复"不治"总量"）；分页治"浪费"不治"单价"（省的是碎片不是每 token 字节）；量化与结构变化治"单价"但要付数值/质量代价。**先看容量瓶颈的类型，再选机制——这是 topic 04 判断顺序的实操版。**

---

## 4. 实测汇总与证据

全部数字在 [../evidence/task4](../evidence/task4) 的 JSON 与日志中可复核（vm-60，RTX 4090 D 24 GB，torch 2.9.1+cu128）。

| 实验 | 文件 | 关键实测 |
|---|---|---|
| 1 增长账本 | [task4_exp1_growth.json](../evidence/task4/task4_exp1_growth.json) | 公式=实测（4 组逐字节）；6 维线性；MHA 96/GQA-4 48/GQA-2 24/MQA 12 MiB；MLA S=4096 54 MiB vs MHA 384；FP8 err 0.123/INT8 err 0.012 |
| 2 分页 | [task4_exp2_paged.json](../evidence/task4/task4_exp2_paged.json) | 原子 OOM/边界扩容/refcount 共享；构造性碎片 9.35 GiB 空闲、4.63 GiB 连续块失败、分页成功；4101 vs 38390.6 MiB；bs 扫描 1→256；gather 注意力等价 max_diff=0 |
| 3 前缀复用 | [task4_exp3_prefix.json](../evidence/task4/task4_exp3_prefix.json) | 块哈希链 8 项机制测试；4096 prompt TTFT -47.6%；512 prompt 零收益（诚实对照）；存储 -66.7%；logits 差 0.005；连续前缀 512/0/256 |
| 4 容量验证 | [task4_exp4_capacity.json](../evidence/task4/task4_exp4_capacity.json) | KV 池 17.65 GiB（权重 3.09 实测）；六变体容量表；准入实测 127 请求、峰值 KV 6.94/3.47/0.43 GiB（MHA/GQA/MQA）；TPOT ~22 ms 平坦；策略决策 reject/tune |
| 上游 notebook | [task4_*_solved.ipynb](../evidence/task4/) | 11/22/34/24/66/69 六个 notebook 补全 TODO 后在 vm-60 全部执行通过（commit e518638） |

证据图：[01_growth_ledger.png](../evidence/task4/01_growth_ledger.png) · [02_paged_tradeoff.png](../evidence/task4/02_paged_tradeoff.png) · [03_prefix_cache.png](../evidence/task4/03_prefix_cache.png) · [04_capacity.png](../evidence/task4/04_capacity.png) · [task4_runshot.png](../evidence/task4/task4_runshot.png)（实测数据排版图，非终端截图）。

## 5. 复现

```bash
# vm-60（RTX 4090 D）执行；onnx 目录见 evidence/task4
python3 task4_common.py            # 无独立输出（公共模块）
python3 task4_exp1_growth.py       # ALL_TESTS_PASS
python3 task4_exp2_paged.py        # ALL_TESTS_PASS
python3 task4_exp3_prefix.py       # ALL_TESTS_PASS
python3 task4_exp4_capacity.py     # ALL_TESTS_PASS
python3 task4_render_evidence.py --evidence <dir>   # 出图
```

源码：[../code/](../code/)。上游 notebook 的补全方式：用各 notebook 自带参考实现单元格替换 TODO 单元格后整体 `nbconvert --execute`（69 节 5.5 单元格 `import json` 缺失为上游小瑕疵，已打补丁并在日志注明）。

## 6. 边界与诚实声明

- 真实模型均为**随机初始化**的教学形状（12 层/1024、24 层/2048、LLaMA 形状账本），结论是机制与容量关系，不是生成质量结论；
- FP8/INT8 KV 只测了存储字节与投影张量误差，未在 decode kernel 中做端到端质量评测；MLA 行为 latent 缓冲账本，非真实 MLA kernel 收益；
- 准入实测用整段 batch（请求同构 592 token），真实服务的异构混合、排队与驱逐会更复杂；波次上限 64 先于 KV 预算触发，报告中已注明；
- vm-60 单卡实测（RTX 4090 D 24 GB）；多卡/分布式显存（Task6 方向）不在本篇范围。
