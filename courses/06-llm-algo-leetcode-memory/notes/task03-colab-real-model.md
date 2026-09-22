# Task3 增补：Colab 真实模型实测（Qwen2.5-0.5B · T4）

社区与教程：[DataWhale](https://www.datawhale.cn/) · [llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode) · [Issue #169](https://github.com/datawhalechina/llm-algo-leetcode/issues/169)。本文是 [task03-training-measurement-budget.md](task03-training-measurement-budget.md)（vm-60 合成 MLP）的真实模型补充：同一套测量与判定脚本跑真实 Qwen2.5-0.5B + WikiText-2。证据：[`../evidence/task3_colab/`](../evidence/task3_colab/)。

## 1. 环境与 workload

| 项 | 值 |
|---|---|
| GPU | Tesla T4，可见 14.56 GiB |
| torch / transformers / datasets | 2.11.0+cu128 / 4.48.3 / 3.2.0 |
| 模型 | Qwen2.5-0.5B，commit `060db649…`，494,032,768 参数 |
| 数据 | wikitext-2-raw-v1，commit `b08601e…`，train/validation 指纹记录在案 |
| 口径 | FP32 权重 + AdamW(foreach=False)，autocast bf16，seq=128，EOS 分隔无 padding，effective batch=2，3 warmup + 5 measured，逐 batch sha256 |

模型与 `main` 首次运行即解析为不可变 commit，之后所有策略从同一本地快照加载；每批输入哈希固定，策略之间不存在数据漂移。

## 2. 主表实测（decisions.json：三策略全部 PASS）

| 策略 | peak allocated | peak reserved | tokens/s | step 中位 | 初始 loss | 最终 held-out loss |
|---|---:|---:|---:|---:|---:|---:|
| baseline（micro 2×1） | 8.42 GiB | 9.49 GiB | 467.6 | 0.542 s | 3.0038 | 2.6046 |
| 梯度累积（micro 1×2） | 9.08 GiB | 9.74 GiB | 367.6 | 0.690 s | 3.0038 | 2.6038 |
| checkpoint（micro 2×1） | 8.41 GiB | 9.27 GiB | 412.1 | 0.617 s | 3.0038 | 2.6046 |

held-out loss 从 3.0038 降到约 2.604，8 步内是真实学习；但这是短跑，**不是收敛结论**。三策略最终 loss 差异 ≤ 0.0008，质量无可辨差异。

## 3. 账本口径：静态状态吃掉 87%（对应 4.2(2)）

按实测：参数 494M × 4B = 1.84 GiB、梯度 1.84 GiB、Adam m/v 各 1.84 GiB，**常驻模型状态 7.36 GiB**，占 baseline 峰值 allocated 8.42 GiB 的 87.5%。峰值减常住的约 1.06 GiB 才是激活 + 输入 + logits/临时缓冲的混合残差——任何只省激活的策略，天花板都被压到个位数百分比。这正是本次实测最重要的观察：

- checkpoint 只把 reserved 峰值从 9.49 降到 9.27（−2.3%），吞吐 −11.9%；
- 累积的峰值反而略高（9.74 vs 9.49，allocator 行为，如实记录），吞吐 −21.4%。

## 4. 为什么显存下降不一定值得采用（对应 4.2(1)）

三策略质量无差异、都在 14 GiB 预算内时，**baseline 是唯一合理选择**：最快且放得下。激活类技巧在本 workload 的收益上限约 2%，代价是 12–21% 吞吐。只有预算被压到 9.5 GiB 以下（例如同机再跑第二个进程），才有理由为 checkpoint 付出的时间买单。这与 vm-60 合成实验方向一致：先看静态状态是不是大头，再决定要不要动激活。

## 5. 未覆盖部分（如实说明）

Notebook 已写好但本次导出未运行的实测单元格：显存账本的逐项精确分解（含 per-strategy 残差）、activation offload（`save_on_cpu`）、缩小 batch 与缩短序列（独立 workload）、bitsandbytes 8-bit 优化器与 4-bit NF4 权重静态实测、预算阈值 6→24 GiB 敏感性扫描、baseline/offload 的 profiler trace 四分类（计算 / 访存 / 传输 / 同步）。补跑并重新导出后，本文与证据目录会同步更新。

## 6. 复现

`code/task3_colab_real_model_new.ipynb` → Colab 选 GPU → 从上到下运行。第一格 `%pip` 安装固定版本后需重启运行时再跑一次；末尾单元格导出证据 ZIP。本地 vm-60 合成版见 [task03-training-measurement-budget.md](task03-training-measurement-budget.md)。