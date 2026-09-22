# Task3 增补：Colab 真实模型实测（Qwen2.5-0.5B · T4 · 全量增项）

社区与教程：[DataWhale](https://www.datawhale.cn/) · [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) · [Issue #169](https://github.com/datawhalechina/llm-algo-leetcode/issues/169)。本文是 [task03-training-measurement-budget.md](task03-training-measurement-budget.md)（vm-60 合成 MLP）的真实模型补充：同一套测量与判定方法跑真实 Qwen2.5-0.5B + WikiText-2，并把 4.2 增项全部实测。证据：[`../evidence/task3_colab/`](../evidence/task3_colab/)（24 个文件，含 trace 与图表）。

## 1. 环境与 workload

| 项 | 值 |
|---|---|
| GPU | Tesla T4（可见 14.56 GiB） |
| torch / CUDA | 2.11.0+cu128 / 12.8 |
| transformers / datasets | 4.48.3 / 3.2.0 |
| 模型 | Qwen2.5-0.5B，commit `060db649`，494,032,768 参数 |
| 数据 | wikitext-2-raw-v1，commit `b08601e`，train/validation 指纹记录在案 |
| 口径 | FP32 权重 + 优化器，autocast bf16；seq=128、EOS 分隔无 padding、effective batch=2；3 warmup + 5 measured；逐 batch sha256；单卡无通信 |

## 2. 主表 + 增项实测总览

| 策略 | peak reserved | peak allocated | tokens/s | 初始 loss | 最终 held-out loss | 显存节省 | 吞吐比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 9.457 | 8.424 | 408.6 | 3.0038 | 2.6046 | 0% | 1.000 |
| 梯度累积 1×2 | 9.740 | 9.080 | 355.9 | 3.0038 | 2.6038 | **−2.99%** | 0.871 |
| checkpoint | 9.271 | 8.410 | 395.0 | 3.0038 | 2.6046 | +1.97% | 0.967 |
| activation offload | 9.254 | 8.410 | 261.3 | 3.0038 | 2.6046 | +2.15% | 0.640 |
| **8-bit AdamW** | **5.932** | **5.170** | **429.6** | 3.0038 | 2.6033 | **+37.27%** | **1.052** |

前三行与 offload、8-bit 五行工作负载完全相同（seq128 / effective batch 2），可互比。8 步短跑，held-out loss 从 3.0038 到约 2.604 是真实学习，**不是收敛结论**。

4-bit 权重（NF4，仅静态/前向，权重冻结不训练）：加载后 allocated **0.444 GiB**、前向峰值 **0.736 GiB**、实存权重字节 0.420 GiB（FP32 权重 1.840 GiB，**省 76%**）、held-out loss **3.1022**（FP32 同一点 3.0038，**质量代价 +0.098，可实测**）。

## 3. 显存账本（对应 4.2(2)）

按张量精确统计 + 实测差值（`memory_ledger.json`）：

| 账本 | 数值 | 口径 |
|---|---:|---|
| 参数 | 1.8404 GiB | 494M × 4B |
| 梯度 | 1.8404 GiB | 与参数同形 |
| Adam m/v | 3.6808 GiB | fp32 两态 |
| **常驻模型状态** | **7.3617 GiB** | 占 baseline 峰值 allocated 8.424 的 **87.4%** |
| 激活 + 临时 | 0.0365 GiB | 实测「反向结束 allocated − 参数 − 梯度」 |
| 峰值 − 常驻 | 1.0156 GiB | 含 logits/输入/workspace 的混合残差 |
| 分配器缓存 | 1.501 GiB | 实测「reserved − allocated」 |

per-strategy 峰值残差：baseline 1.0623、累积 1.7188、checkpoint 1.0482 GiB——**激活类策略能动的只有这 1 GiB 量级**，天花板决定了下文的结论。

## 4. 为什么显存下降不一定值得采用（对应 4.2(1)）

本次实测同时给出了正反两面：

- **反面（省了也没用）**：累积显存峰值反而高 2.99%、吞吐 −12.9%；checkpoint 省 1.97% 但吞吐 −3.3%；offload 省 2.15% 却吞吐 −36%。三者的「收益」都在测量噪声与 allocator 行为量级，代价却是实打实的时间。原因是静态状态占 87.4%，激活技巧的天花板只有约 1 GiB。
- **正面（值得采用）**：8-bit AdamW 把 3.6808 GiB 的 fp32 优化器状态压掉，峰值 reserved 降 **37.27%**，吞吐反而快 5.2%，held-out loss 差 0.0013。它省的是常驻状态而不是激活，所以收益不被天花板限制。

结论：**先看要省的对象是不是大头，再谈值不值**。4.2(1) 的答案在这个 workload 上就是：省激活 = 不值得；省优化器状态 = 值得。

## 5. 预算敏感性（实测峰值 + 阈值扫描）

规则：`peak reserved + 1 GiB <= 预算`、吞吐 `>= baseline 40%`（163.4 tokens/s）、`|final loss − baseline| <= 0.15`。

| 预算 GiB | 可行 | 推荐 |
|---:|---|---|
| 6 | — | infeasible |
| 8 | 8-bit AdamW（需 6.932） | 8-bit AdamW |
| 10 | 8-bit AdamW | 8-bit AdamW |
| 12 | 全部五项（最大需求 10.74） | 8-bit AdamW（吞吐 429.6 最高） |
| 14 / 16 / 20 / 24 | 全部五项 | 8-bit AdamW |

每个预算档都是 8-bit AdamW——它不是「窄阈值下偶然成立」，而是在全预算区间占优，因此不标记为预算敏感。

> 说明：Notebook 内 `budget_sensitivity.json` 因按 `workload` 字符串过滤（基线三策略该字段为空）只保留了 offload/quant；上表是用同一批实测数据、同一规则重算的，输入全部来自 `summary.json` 的实测列。

## 6. profiler trace 四分类（对应 4.2(4)）

对 baseline 与 offload 各跑一次带 profiler 的 step，导出 chrome trace 后按 `kernel`/`gpu_memcpy`/`cuda_runtime` 事件类型聚合（`trace_summary.json`，原始 trace 同名 json，可在 Perfetto / chrome://tracing 打开）：

| 类别 | baseline | offload | 解读 |
|---|---:|---:|---|
| 计算候选（gemm kernel） | 652 次 / 359.7 ms | 652 次 / 292.8 ms | 两者的计算量一致（同输入同模型） |
| 访存候选（elementwise/reduce） | 7,288 次 / 283.3 ms | 7,481 次 / 366.4 ms | 量级相当 |
| **数据传输（memcpy）** | **54 次 / 0.42 ms** | **1,904 次 / 303.28 ms** | offload 的代价主体：PCIe 搬运放大约 720 倍 |
| **同步等待（synchronize）** | **7 次 / 104.6 ms** | **932 次 / 461.1 ms** | 等搬运完成，同步点爆炸 |
| 调度/启动（runtime） | 8,740 次 / 295.0 ms | 15,247 次 / 303.4 ms | launch 次数近乎翻倍 |
| CPU 算子 | 34,815 次 / 1,614 ms | 42,102 次 / 3,526 ms | CPU 侧也因搬运同步变重 |

方法论：**先按事件类型分桶，再看桶间差异**。kernel 时间长不等于 compute-bound——本 trace 里 gemm 只占 GPU 事件约 45%，且没有带宽计数器，不能断言饱和；memcpy/同步桶的暴涨才是 offload 的瓶颈证据。CPU 侧的 synchronize 时间包含对已提交 GPU 工作的等待，不能与 kernel 时间相加当成额外代价。

## 7. 五条路径的横向结论（对应 4.2(3)）

| 路径 | 实测收益 | 实测代价 | 判定 |
|---|---|---|---|
| 量化（优化器 8-bit） | reserved −37.27% | 吞吐 +5.2%，loss +0.0013 | **值得采用** |
| 量化（权重 NF4） | 权重 1.840 → 0.420 GiB（−76%，推理侧） | held-out loss +0.098；冻结不训练 | 推理侧可用，训练侧另论 |
| checkpoint | reserved −1.97% | 吞吐 −3.3% | 本 workload 不值得 |
| offload | reserved −2.15% | 吞吐 −36%，memcpy ×35 | 本 workload 不值得 |
| 缩小 batch | **未测**（独立 workload，改变了每步样本数与优化轨迹） | — | 待补 |
| 缩短 sequence | **未测**（独立 workload，丢失上下文） | — | 待补 |

「缩小 batch / 缩短序列」两个变体本次未运行（会话时间花在 4-bit 加载的运行时兼容排障），已在证据目录如实标注；两者都需要独立质量评估，不能与主表比 loss。

## 8. 复现与排障记录

`code/task3_colab_real_model_new.ipynb` → Colab 选 GPU → 从上到下运行；安装格含 bitsandbytes，装完按提示重启运行时再跑（否则 transformers 的 bnb 集成会在缺 `nn`/`bnb` 的状态下被 import，4-bit 加载连续报 NameError）。完整踩坑与修复过程见 git 提交记录（缓存刷新 → 命名空间注入 → 模块重载 → lazy 缓存 → gc 扫描五轮迭代）。

## 9. 局限

- 8 步短跑：loss 下降是真实学习，不是收敛结论；固定数据重复 8 步，指标只用于公平对照。
- T4 上 autocast 记录为 bf16（运行环境决定）；单卡，无通信测量。
- 4-bit 行为推理侧静态/前向测量，不参与训练侧吞吐比较。
- 预算规则中的 1 GiB 余量与 40% 吞吐门槛是人为设定，不是物理 OOM 测试。
- 合成残差 MLP 的机制验证见 [task03-training-measurement-budget.md](task03-training-measurement-budget.md)（vm-60）。
