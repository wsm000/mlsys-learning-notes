# Task5：量化与显存容量扩展 —— 学习笔记

- GitHub ID：wsm000 ｜ 微信群昵称：（填写）
- 打卡选择：**4.1 + 4.2 + 4.3（全量）**
- 运行环境：vm-60 · NVIDIA RTX 4090 D（24 GB，24564 MiB）· torch 2.9.1+cu128 · transformers 5.16.1 · torchao 0.15.0 · Python 3.10.12
- 教程与社区：[datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) ｜ [DataWhale 社区](https://github.com/datawhalechina) ｜ [打卡 Issue #178](https://github.com/datawhalechina/llm-algo-leetcode/issues/178)
- 对应节：Part 01 · 21（量化理论与 INT4/INT8）、Part 02 · 25（W8A16）、Part 02 · 40（GPTQ / AWQ）、Part 02 · 41（FP8 与 KV Cache 量化）、Part 02 · 66（推理性能对比）、Part 02 · 67（量化推理与部署）、topic 05（量化作为显存工具）
- 上游基准：commit `e518638`（与 Task4 相同）

> 本篇为个人学习笔记，实验代码与实测数据在 [../evidence/task5](../evidence/task5)；观点为原创整理，非教程原文照抄。Issue 截止时间为 09-28 12:00，本篇为截止后补交（issue 已被官方标记 completed，评论里已有其他同学打卡）。

---

## 0. 一句话总览与证据边界

Task5 的问题意识（issue 原文）：**压缩哪类对象、在什么时候压缩，才能真正换来更大的模型、上下文或并发？** 我把它拆成"对象 → 时机 → 证据链"三层，每层都用 vm-60 实测闭环：

1. **对象**：权重、激活、KV Cache 是三种生命周期完全不同的内存，量化时机、误差传播路径、验证指标都不同（实验1、实验2）；
2. **时机**：预测量化压缩"静态占用"，KV 量化压缩"随负载增长的单位成本"——前者省固定字节，后者改线性斜率（实验2C、实验3B）；
3. **证据链**：字节账本 → 数值误差 → 容量/并发 → 同 workload 候选决策，缺一层就把"候选收益"当"部署收益"（实验3、实验4 + 六份上游 notebook）。

证据边界（先讲清楚不说什么）：

| 证据等级 | 本笔记有哪些 | 因此不报告什么 |
|---|---|---|
| 账本/公式 | 权重三档、KV 每 token、容量池（exp1/exp3） | 不声称硬件一定支持对应计算路径 |
| GPU 真实状态测量 | 真实 Qwen2.5-0.5B 权重/KV/逐层前向（exp2/exp3 + notebook 25/40/41 的 real_gpu 步骤） | 不是 kernel 级收益，反量化副本是教学实现行为 |
| 构造性压力实测 | 准入至 OOM（exp3C）、24 层误差复合（exp2B） | 不代表真实 serving 的 P99/吞吐 |
| 教学模拟 | GPTQ/AWQ 模拟、KV 往返（exp4 KV-INT8）、toy-12L 候选对照 | 非真实 INT8/FP8 kernel；随机初始化模型上权重量化误差是下界 |
| 未运行 | vLLM/SGLang backend（未安装）、GPTQ/AWQ 真实 artifact 部署 | 67 的 `decision = pending`，如实标注 |

---

## 1. 4.1 最小打卡

### 1.1 哪类对象优先量化？为什么不能只看参数量？（对应 21 / 25 / 40）

三类对象的账本位置完全不同，我用同一台 4090D 实测了"静态账本"和"增长账本"两侧。

**权重侧（实验1A，真实 Qwen2.5-0.5B-Instruct 全部 169 个 2D+ 矩阵，493,961,216 参数）：**

| 表示 | 字节 | 相对 FP16 | scale 元数据 | 重建 MSE |
|---|---:|---:|---:|---:|
| FP16 | 987,922,432 B（942.16 MiB） | 1.000× | — | — |
| INT8 per-tensor | 493,961,892 B（471.08 MiB） | **2.000×** | 676 B（169 个标量） | 8.806e-07 |
| INT8 per-channel | 495,785,472 B（472.82 MiB） | 1.993× | +1.82 MiB | 4.805e-08 |
| INT4 per-group(g=128, 打包) | 262,416,896 B（250.26 MiB） | **3.765×** | 14.72 MiB | 1.899e-06 |

两个读数：① INT8 就是干净的 -50%，per-channel 只多 0.4% 元数据就把 MSE 压低 18 倍（粒度收益便宜）；② **INT4 名义 4× 实测只有 3.765×**——g=128 时 scale 占 5.9%，粒度越细元数据越稀释压缩收益（g 减半 scale 约翻倍）。这与 41 节 KV group size 的取舍是同一个权衡。

**增长侧（实验1B，32 层参考账本 + toy-12L 激活峰值实测）：**

| 变体 | FP16 | INT8/FP8 |
|---|---:|---:|
| MHA-32kv | 512 KiB/token | 256 KiB/token |
| GQA-8kv | 128 KiB/token | 64 KiB/token |
| MQA-1kv | 16 KiB/token | 8 KiB/token |

激活：toy-12L（B=4，S=512 prefill）权重 419.0 MiB、forward 峰值 664.2 MiB → 激活+临时 ≈ 245 MiB。

**"不能只看参数量"的实证（实验1C，同一个 7B 模型、两种 workload）：**

| workload | 权重 FP16 | KV FP16（GQA-8kv） | 账本构成 |
|---|---:|---:|---|
| 短上下文单并发（512 tok，B=1） | 14.19 GiB | 0.06 GiB | 权重 99.6% / KV 0.4% |
| 长上下文高并发（32K tok，B=32） | 14.19 GiB | 128.0 GiB | 权重 10.0% / KV 90.0% |

权重一模一样，瓶颈从"装不下"翻成"涨不完"。再把权重 INT8 化：权重降到 7.10 GiB，**KV 占比反而从 90.0% 升到 94.7%**——量化会转移瓶颈，不消除瓶颈。所以选什么对象取决于 workload 的账本构成，而账本构成取决于 batch × 上下文 × 并发 × 阶段（prefill/decode），这些参数量一个都回答不了。

### 1.2 W8A16 如何减少权重显存？INT8 / scale / FP16 激活各是什么角色？（对应 25）

W8A16 = Weight 8-bit、Activation 16-bit：权重换 INT8 存储（1 B/参数），激活保持 FP16/BF16 参与计算。三个角色（实验2A，真实 q_proj 896×896，输入 `[1,10,896]`）：

| 角色 | 承担什么 | 实测 |
|---|---|---|
| INT8 权重 | 存储本体，显存与加载流量减半的唯一来源 | 1,605,632 B → 802,820 B（**-50.0%**） |
| scale | 整数↔浮点的动态范围映射元数据 | per-tensor 仅 4 B；per-channel 每行一个也才 +1.82 MiB/0.4% |
| FP16 激活 | 计算路径，动态范围大、有离群值，不量化 | 输出 MSE 3.893e-06，余弦≈1 |

**反量化发生在进入 matmul 之前**：`w_fp = int8.to(fp16); w_dequant = w_fp / scale; F.linear(x, w_dequant, bias)`。所以 MACs 与 FP16 完全相同（不碰 Tensor Core 低比特路径），收益只在存储与权重读取带宽。代价也 measurable（实验2A）：前向 0.0591 → 0.1219 ms（0.07 ms 量级受 launch 开销影响，两次运行比率在 2.1–4.1× 间波动）、**峰值显存 11.46 → 14.84 MiB（+29%，稳定复现）**——反量化物化了一份 fp16 权重副本。上游 25 节 real_gpu 步骤同向（0.0408→0.094 ms、11.46→14.84 MB、MSE 3.89e-06），峰值与 MSE 两次测量一致，latency 因计时方法（中位数 vs 均值）不同。

### 1.3 为什么 W8A16 不一定解决长上下文/高并发的 KV OOM？（对应 25 → 41）

因为 W8A16 只动账本的"权重行"，且省下的是**固定字节**；KV Cache 是另一行**随 token 线性增长**的动态账。实测三条：

1. **全模型存储收益打折**（实验2B）：Qwen2.5-0.5B tied embedding（272 MiB）不量化 + bias/scale，参数从 988.1 MB 只降到 768.1 MB（**×1.29，不是 ×2**）。0.5B 这种小模型 embedding 占比高，压缩比天然被稀释；7B 才是接近 ×2。
2. **省下的固定字节很有限**（实验2C）：toy-12L 权重 419.0 → 209.5 MiB，固定省 209.5 MiB；每 token KV 48.0 KiB → 只够多 **4,470 token**（≈11 个 512-token 请求）。7B INT8 省 7.10 GiB，一个 32K 上下文 GQA-8kv fp16 请求 KV = 4 GiB——两个长请求就吃回去，而 decode 每步还在线性增长。
3. **教学实现的峰值反升**（实验2B）：逐层反量化物化副本，forward 峰值 976 → 1970 MiB（per-tensor）/ 2487 MiB（per-channel）；latency 28.96 → 40.18 / 41.69 ms。kernel 内反量化的真实 backend 不会有这个副本，但**"存储收益 ≠ 峰值收益"要用同一 workload 实测才能下结论**。

### 1.4 GPTQ 与 AWQ 的误差控制依赖什么信息？为什么需要校准数据？（对应 40）

两者都是 PTQ，回答"量化误差摊到哪"，但利用的信息不同：GPTQ 用**层输出的二阶信息**（Hessian 近似，逐列量化并把已量化列的残差补偿到后续列）；AWQ 用**激活幅度统计**找出显著通道并原值保护。上游 40 节 real_gpu 步骤（[40_gptq_awq_gpu.json](../evidence/task5/notebook_results/40_gptq_awq_gpu.json)，真实 q_proj + 3 条校准 prompt / 256 个激活样本）：

| 路径 | 权重重建 MSE | **校准输出 MSE** | 保护通道 | fit 耗时 | peak memory |
|---|---:|---:|---:|---:|---:|
| GPTQ | 5.195e-05 | 5.2352e-04 | 0 | 8,788 ms | 23.64 MB |
| AWQ | 3.747e-05 | **1.3635e-04（GPTQ 的 1/3.8）** | 56 | 12,322 ms | 27.58 MB |

**为什么需要校准数据**：量化误差对输出的影响 = 权重误差 × 实际流过的激活；权重分布是静态的，"哪条通道的误差最致命"只存在于激活分布里。注意两个读数：① 权重 MSE 只差 28%，**校准输出 MSE 差 3.8 倍**——不能只看权重 MSE；② 这份输出 MSE 是在校准数据自己身上算的（自评），分布外任务必须另测（见 1.5 与 3.2）。

**校准在权重量化里是"可选项"，在 KV 量化里是"默认不存在"**：KV 是运行时产生的，没有离线校准阶段，只能 per-tensor/per-row 地在线量化——这正是 2.1 里真实 KV 误差比权重差 5 个数量级的原因之一。

### 1.5 预算不足时，除压缩比还要比较什么质量与运行时证据？（对应 21/25/40 → 67）

我的选型清单 = 误差预算（硬门槛）→ 峰值显存 → TTFT/TPOT/吞吐 → 元数据与 kernel 路径。同 workload 实测（实验4，toy-12L GQA-8kv，B=4，prompt 512，gen 64，teacher-forced 固定续续序列，同 seed 同权重，只改量化一个变量族）：

| 候选 | TTFT ms | TPOT ms | 吞吐 tok/s | peak MiB | KV MiB | logits 余弦 | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| FP16 baseline | 15.54 | 12.260 | 324.9 | 705.3 | 108.00 | 1.00000 | baseline |
| W8A16 权重 | 19.21 | 16.517 | 241.6（-26%） | **648.7（-8.0%）** | 108.00 | 0.99881 | **tune** |
| INT4 权重 | 17.41 | 17.150 | 233.2 | **779.3（+10.5%）** | 108.00 | 0.96016 | **reject** |
| W8A16 + KV-INT8 | 17.03 | 21.486 | 186.8 | 648.7（-8.0%） | **54.00（-50%）** | 0.99877 | **tune** |

三个结论：① **误差门槛前置**——INT4 候选余弦 0.960 < 0.99 直接 reject，且它峰值显存反而 +10.5%（解包副本），"更激进的位宽"在这个教学实现里两头亏；② W8A16 只有显存一类收益（-8%）、吞吐 -26% → tune（kernel 内反量化才有可能同时收益）；③ KV-INT8 把 KV 存储对半切但 TPOT 恶化 +75%（每步全量往返反量化），同样是 tune。**压缩比只是入场券**：真实决策要看误差预算、峰值、时延、元数据成本和 backend 路径五类证据。

---

## 2. 4.2 增项 1（topic 05 + Part 02 · 41）

### 2.1 FP8 计算路径与 KV Cache 量化路径分别压缩什么？（对应 41）

| | FP8 计算路径 | KV Cache 量化路径 |
|---|---|---|
| 作用对象 | 前向计算中的权重/激活（GEMM 输入输出） | 跨 decode step 驻留的历史 K/V |
| 时机 | 训练/前向中，可离线标定部分 | 请求执行中在线量化，无校准阶段 |
| 显存影响 | 降中间张量与访存 | 降驻留容量 → 直接改并发/上下文上限 |
| 主要风险 | 动态范围有限，误差**当步**影响输出 | 误差被每个 decode step **重读累积**；scale/block 管理与 backend 支持 |

上游 41 节用同一份教学模拟器把两条路跑成对照：`hidden (1,128,4096)`（前向张量）与 `kv (1,16,512,128)`（驻留状态）同样过 INT8 容器，物化的误差对象完全不同。真实状态实测（实验3A，Qwen2.5-0.5B-Instruct 第 0 层真实 `past_key_values`，K/V `[1,2,13,64]`，FP16 基线 6,656 B）：

| 变体 | 字节 | 压缩比 | MSE_K | MSE_V | max_err_K | 余弦_K |
|---|---:|---:|---:|---:|---:|---:|
| INT8 per-tensor | 3,336 B | 1.995× | 7.585e-02 | 4.155e-07 | 0.513 | 0.999964 |
| INT8 per-row (g=64) | 3,536 B | 1.882× | 7.151e-02 | 6.944e-08 | 0.512 | 0.999966 |
| INT4 per-row (g=64) | 1,872 B | 3.556× | **4.216e+00** | 4.817e-06 | **4.344** | 0.998014 |
| FP8 E4M3 per-tensor | 3,328 B | 2.000× | 5.525e-01 | 6.032e-07 | **4.000** | 0.999740 |

三个观察：① **K 远比 V 难量化**（MSE 差 5 个数量级）——K 有大幅值离群 token；② **粒度不是越细越好**：per-row（按 token 行分组）对 K 几乎没有改善（7.59e-02 → 7.15e-02），因为离群值集中在个别行，该行 scale 被拉大、同行其余元素被压碎；③ INT4/FP8 的 max_err 到 4.3/4.0（e4m3 指数范围只有 ±448 且尾数 3 位），名义 4×/2× 压缩要用最大误差和余弦一起审，不能只看比率。

### 2.2 为什么 FP8 weight-only 探针成功 ≠ KV Cache 量化可用于 serving？（对应 41）

torchao FP8 weight-only 探针（[41_fp8_torchao_probe.json](../evidence/task5/notebook_results/41_fp8_torchao_probe.json)）在 4090D 上真实跑通、shape 正确，但它的 `decision` 字段自己写着 `weight_only_api_path_executed`——**它验证的是库 API 路径，没有碰过 KV Cache**。真实 KV 量化要成立，backend 必须在 attention kernel 内完成：读量化 KV → 按组反量化 → 支持动态增长与驱逐 → 与 PagedAttention/前缀缓存兼容。这些探针一个都没验证。上游 41 节 real_gpu 步骤也只是"真实 KV 状态 + INT8 容器往返"的教学测量（9,216 → 2,592 B，×3.556，MSE 3.04e-02，peak 962.8 MB），同样不构成 serving 证据。**看到"探针成功"先问它探的是 API 还是 kernel。**

### 2.3 KV 量化如何影响最大并发/上下文？还要预留什么？（对应 41 + Task4 容量方法）

容量账本（实验3B，4090D 24 GB；KV 池 = 总显存×0.9 − 权重（实测 419.0 MiB）− 运行缓冲（实测 75.1 MiB）− 256 MiB 余量 = **20.44 GiB**）：

| 变体 | 每 token | 最大 token 数 | 最大并发 @S=512 | @S=4096 |
|---|---:|---:|---:|---:|
| MHA-32kv FP16 | 512 KiB | 41,863 | 81 | 10 |
| MHA-32kv INT8/FP8 | 256 KiB | 83,726 | **163** | **20** |
| GQA-8kv FP16 | 128 KiB | 167,452 | 327 | 40 |
| GQA-8kv INT8/FP8 | 64 KiB | 334,904 | **654** | **81** |
| MQA-1kv FP16 | 16 KiB | 1,339,619 | 2,616 | 327 |
| MQA-1kv INT8/FP8 | 8 KiB | 2,679,238 | **5,232** | **654** |

每 token 单价减半 → 容量各项精确翻倍（容量与单价成反比）。**构造性准入实测**（实验3C，toy-12L 真实 KV 形状 48 KiB/token，S=4096 每请求 192 MiB，逐请求真实分配直到 OOM）：FP16 准入 **123** 请求（20.35 GiB）OOM，INT8（96 MiB/请求）准入 **246** 请求（同样 20.35 GiB）OOM，**并发 ×2.00**——账本翻倍在真机上兑现。

仍要预留的运行时空间（实验3D，实测）：权重 419.0 MiB、激活峰值 243.1 MiB（forward 峰值 662.2 MiB）、分配器 reserved−allocated gap 71.8 MiB，再加 256 MiB 余量与量化 scale 元数据。**量化后的理论容量不能全部转成用户 token 容量。**

### 2.4 权重已量化、长 prefill 高并发仍 OOM，先查什么？（对应 41 + topic 05）

按"OOM 随什么增长"分两条线：若随累计 token 增长，查 KV Cache（是否仍 FP16、每 token 单价、block/pool 实际分配、kv_heads×head_dim×层数、前缀缓存驻留、未释放的 cache）；若只在长 prefill 时炸，查激活峰值与 prefill 工作区、scheduler token budget——**不是所有长 prefill OOM 都是 KV 导致的**。我的账本侧佐证是 1.1 的瓶颈转移表：权重 INT8 后 KV 占比反而升到 94.7%，此时继续压权重（INT4）对 OOM 边界几乎无贡献，实验4 的 INT4 候选也实测"峰值 +10.5%、余弦 0.960"两头亏。

### 2.5 backend 不支持目标量化 kernel 时，正确加载的 artifact 有什么风险？（对应 67）

三类风险都在本轮量到：① **静默回退浮点**：artifact 能加载但 kernel 不认 → 只拿到存储收益，规划却按计算收益做（67 的上游契约测试把"没有 baseline 不算完成量化部署"写成机制：`absent/legacy/mismatch/compatible` 四态，且 `adapter_manifest.can_replace_quantization_artifact = False`）；② **反量化副本推高峰值**：实验2B 的教学实现 per-tensor 峰值 +102%（976→1970 MiB）就是"存储省了、峰值升了"的活例；③ **吞吐倒退**：实验4 W8A16 TPOT +34%、+KV-INT8 TPOT +75%（往返反量化）。67 的 Step 5 真实 backend（vLLM）本机未安装，`RUN_REAL_BACKEND=False` → `decision = pending`，不伪造任何 serving 数字。

---

## 3. 4.3 增项 2（Part 02 · 66 + Part 02 · 67）

### 3.1 为什么 66 的浮点 baseline 是必要对照？必须固定哪些字段？（对应 66）

baseline 是**唯一比较基准 + 契约来源**：模型、backend、workload、dtype、batch、并发、warmup、重复次数、评测口径。67 的 CPU 模型也把这点写成机制（`build_inference_config` 要求 baseline/candidate 共用同一 workload 契约，`summarize_quantized_result` 只做差不重算）。我的实验4 固定字段：toy-12L GQA-8kv（同 architecture/同 seed 73 保证同一份初始权重）、B=4、prompt 512、gen 64、fp16、teacher-forced 固定续续序列、同预热、同 kernel。上游 66 节波次/瓶颈/策略三组测试在 vm-60 全过（vLLM backend 未装，G0/G1 真实对照未跑，如实标注）。

### 3.2 peak memory / KV 占用 / TTFT / TPOT / 吞吐 / 质量各用于判断什么？（对应 66 + 67）

- **peak memory**：能不能装下、剩余余量多少（注意 2.5 的副本陷阱——要分 allocated 与 reserved 看）；
- **KV 占用**：并发/上下文上限的直接决定项（实验3B/C 的 ×2 翻倍）；
- **TTFT**：prefill 与权重加载路径的健康度（量化权重影响加载，KV 量化基本不影响 TTFT）；
- **TPOT**：decode 每步成本，KV 量化的真实代价出现在这里（读取路径反量化）；
- **吞吐**：端到端容量收益是否兑现的唯一综合指标；
- **质量**：logits 余弦/MSE 是门槛不是排序项（实验4 INT4 被 0.960 余弦直接 reject）。

### 3.3 峰值显存下降但可承载并发没提升，可能卡在哪？（对应 66 + 67）

实验4 的 W8A16 候选就是活例：peak -8.0%（705.3→648.7 MiB）、KV 存储不变（108 MiB）、吞吐 -26% → 判决 tune。原因链：**KV 单价没变 → 容量边界没动**；吞吐还被反量化开销拖累。若峰值降了、KV 也降了但并发仍不涨，剩下的限制因素在容量之外：准入策略（KV 池预算怎么分给请求）、调度（波次/并发上限、paged block 粒度）、峰值与驻留的差（副本/碎片）、以及 backend 是否真的执行了量化路径。**先判断容量瓶颈类型（静态权重 vs 动态 KV vs 临时激活），再选机制**——这正是 topic 05 的判断顺序，也是本 Task 四个实验的共同结论。

---

## 4. 上游 notebook 执行记录（commit e518638，vm-60）

六份 notebook 的题目单元格均换成上游自带"参考实现"单元格，整体 `nbconvert --execute`（kernel python3），执行产物与全部输出在 [../evidence/task5/](../evidence/task5/)：

| notebook | 对应节 | 结果 |
|---|---|---|
| [task5_21_..._solved_executed.ipynb](../evidence/task5/task5_21_Quantization_Theory_and_INT4_INT8_solved_executed.ipynb) | Part01·21 | 直接执行，3 组测试全过 |
| task5_25_..._solved_executed.ipynb | Part02·25 | TODO 1-2 补全；real_gpu + torchao Int8WeightOnlyConfig 探针，机制测试全过 |
| task5_40_..._solved_executed.ipynb | Part02·40 | TODO 1-5 补全；real_gpu 校准（32 prompt/256 激活），机制测试全过 |
| task5_41_..._solved_executed.ipynb | Part02·41 | TODO 1-3 补全；real_gpu 真实 KV + torchao Float8WeightOnlyConfig 探针，机制测试全过 |
| task5_66_..._solved_executed.ipynb | Part02·66 | TODO 1-3 补全，项目测试全过（vLLM backend 未装，G0/G1 未跑） |
| task5_67_..._solved_executed.ipynb | Part02·67 | TODO 1-3 补全，artifact 契约四态测试全过；`decision = pending`（无 vLLM） |

为跑通 real_gpu 路径做的 **5 处上游修复**（只碰执行路径，不改量化数学）：

| # | notebook | 现象 | 修法 |
|---|---|---|---|
| 1 | 25 | `nn.Module.float()` 是 in-place，污染 fp16 baseline | `q.from_float(copy.deepcopy(fp).float())` |
| 2 | 40 | real_gpu 分支末尾 `del batch` 后仍被引用 | 删除前记录 `calibration_batch_size` |
| 3 | 40 | 单次 `fit()` 约 8.8 s，24 次合计约 29 分钟 | `WARMUP/ITERS` 2/10 → 1/1，真实值写进 config JSON |
| 4 | 41 | transformers 5.x `DynamicCache` 不可下标 | 改 `past.layers[0].keys / .values` |
| 5 | 41 | torchao FP8 kernel 在 `inference_mode` 下报 version_counter | 改 `torch.no_grad()` |

另：67 依赖上游 `tools/` 包（`inference_project_runtime`），已随 notebook 一并放到 vm-60 执行目录。

---

## 5. 复现方式

```bash
# vm-60
cd ~/projects/memopt-task5/code
python3 task5_exp1_object_timing.py && python3 task5_exp2_w8a16_vs_kv.py \
  && python3 task5_exp3_kv_quant_capacity.py && python3 task5_exp4_candidates.py
python3 task5_render_evidence.py --evidence ../evidence/task5
# notebook：cd ../notebooks && python3 -m nbconvert --to notebook --execute \
#   --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=python3 task5_XX_*_solved.ipynb
```

代码：[../code/task5_common.py](../code/task5_common.py)（量化工具与账本）、四个实验脚本、[task5_render_evidence.py](../code/task5_render_evidence.py)（五张证据图）。
证据：[../evidence/task5/](../evidence/task5/)（4 个实验 JSON+日志、6 份已执行 notebook、notebook GPU 步骤 JSON、5 张图、运行总览）。

## 6. 口径声明

- 计时：CUDA 边界（synchronize + perf_counter），warmup 后取中位数（exp2A 单层 0.07 ms 量级受 launch 开销影响，latency 比率不稳定，峰值显存稳定复现）；
- 显存：`torch.cuda.max_memory_allocated` 为主，`memory_reserved` 在 exp3D 单独注明；
- 真实模型权重/KV 为 Qwen2.5-0.5B-Instruct 真实状态；候选对照用随机初始化 toy-12L（同 Task2/4 形状），其权重量化误差是下界（无真实模型的离群结构），质量侧结论以 exp2B 的真实模型 24 层复合为准；
- KV-INT8 候选为"存储 int8、读取时反量化"的往返路径，用于测量存储容量与反量化开销，**不代表真实 INT8 KV kernel**；
- 未安装 vLLM/SGLang/gptqmodel/awq，serving 级数字一律不报告。
