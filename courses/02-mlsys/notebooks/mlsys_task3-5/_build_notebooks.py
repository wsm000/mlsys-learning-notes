# -*- coding: utf-8 -*-
"""Generate mlsys task3/task4/task5 Colab notebooks."""
import json
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines]


def md(block):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(block)}


def code(block):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _src(block)}


def nb(name, cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "colab": {"provenance": [], "name": name},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def save(fname, notebook):
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print("wrote", path)


INSTALL = '''
# Colab 每次 new runtime 需要重新安装（约 1 分钟，纯 CPU 即可，不需要 GPU）
%pip install -q "git+https://github.com/harvard-edge/cs249r_book.git@dev#subdirectory=mlsysim"
'''

# ======================================================================
# Task 3
# ======================================================================
t3 = []

t3.append(md("""
# MLSys Task 3：KV Cache 与模型属性——显存与规模的权衡

对应打卡 issue：[datawhalechina/llm-algo-leetcode #133](https://github.com/datawhalechina/llm-algo-leetcode/issues/133)

理论材料：

- [KV-Cache: The Hidden Memory Consumer](https://mlsysbook.ai/mlsysim/blog/how-much-memory-llama3.html)
- [Quantization: Not a Free Lunch](https://harvard-edge.github.io/cs249r_book_dev/mlsysim/tutorials/05_quantization.html)

核心工具：`ServingModel`、`calc_kv_cache_size`、`CompressionModel`

使用说明：

- 所有实验都是**解析仿真**（Roofline / 物理公式），不是真实 GPU benchmark，Colab CPU runtime 就能跑
- 建议顺序执行；每个实验都有**预测区**，先写下猜测再运行验证（predict before you compute）
- 实验编号与 issue 一致：E1–E5 核心，O1–O4 选做。全部包含在本 notebook 中，可按打卡等级选做
"""))

t3.append(code(INSTALL))

t3.append(code('''
import mlsysim
from mlsysim import ureg
from mlsysim.core.units import Q_
from mlsysim.solvers import ServingModel, CompressionModel
from mlsysim.physics import calc_kv_cache_size
from mlsysim.show import table, info

print("mlsysim 版本:", mlsysim.__version__)

llama8b  = mlsysim.Models.Language.Llama3_8B
llama70b = mlsysim.Models.Language.Llama3_70B
h100     = mlsysim.Hardware.Cloud.H100
solver   = ServingModel()

head_dim_8b = llama8b.hidden_dim // llama8b.heads   # 4096 / 32 = 128

print(f"模型: {llama8b.name} | layers={llama8b.layers}, heads={llama8b.heads}, "
      f"kv_heads={llama8b.kv_heads} (GQA), hidden={llama8b.hidden_dim}")
print(f"硬件: {h100.name} | 显存={h100.memory.capacity.to('GB'):.1f}, "
      f"带宽={h100.memory.bandwidth.to('GB/s'):.0f}")
'''))

t3.append(md("""
## 0. 理论速览

**KV-Cache 公式**（autoregressive 推理的隐藏内存消耗者）：

```text
KV_bytes = 2 × L × H_kv × d_head × S × B × bytes_per_elem
           │   │      │       │      │   │       └ 精度字节(FP16=2)
           │   │      │       │      │   └ batch = 并发请求数
           │   │      │       │      └ 序列长度
           │   │      │       └ 每个 head 的维度
           │   │      └ KV 头数（注意 GQA：用 kv_heads，不是 heads！）
           │   └ 层数
           └ 开头的 2 = Key 和 Value 各存一份
```

> Llama3-8B 用了 GQA：32 个 attention head 但只有 **8 个 KV head**。
> 公式里误用 heads 会把 KV-Cache 高估 4 倍——最常见的踩坑点。

**量化的作用边界**：量化只减少“字节数”，不减少“FLOPs”

| 阶段 | 瓶颈类型 | 耗时公式 | INT4 效果 |
|---|---|---|---|
| Decode | memory-bound | (权重+KV 字节) / 显存带宽 | ≈ 4× 加速 |
| Prefill | compute-bound | FLOPs / 峰值算力 | ≈ 1×（除非硬件低精度算力更强） |
"""))

t3.append(md("""
---
## E1 权重显存：不同精度下模型有多大？

**预测区**（运行前先填）：
- Llama3-8B FP16 ≈ ____ GB，INT8 ≈ ____ GB，INT4 ≈ ____ GB
- Llama3-70B FP16 能放进单张 80GB H100 吗？____
"""))

t3.append(code('''
# E1: Model.size_in_bytes(precision) —— precision 是“每参数字节数”的 Quantity
PRECISIONS = {"FP16": "2 byte", "INT8": "1 byte", "INT4": "0.5 byte"}

rows = []
for name, bpp in PRECISIONS.items():
    size = llama8b.size_in_bytes(ureg(bpp))
    rows.append([name, bpp, f"{size.to('GB'):.1f} GB"])

table(["精度", "每参数字节", "Llama3-8B 权重"], rows)

# 验证“70B FP16 放不进单张 H100”
w70_fp16 = llama70b.size_in_bytes()                      # 默认 FP16
w70_int4 = llama70b.size_in_bytes(ureg("0.5 byte"))
cap_gb = h100.memory.capacity.to("GB").magnitude

print()
print(f"Llama3-70B FP16 权重 : {w70_fp16.to('GB'):.1f} GB")
print(f"Llama3-70B INT4 权重 : {w70_int4.to('GB'):.1f} GB")
print(f"H100 显存容量        : {cap_gb:.1f} GB")
print(f"70B FP16 单卡放得下? : {w70_fp16.to('GB').magnitude <= cap_gb}")
print(f"70B INT4 单卡放得下? : {w70_int4.to('GB').magnitude <= cap_gb}")
'''))

t3.append(md("""
**E1 观察要点**（对照你的预测）：

- 8B：FP16 ≈ 16 GB → INT8 减半 → INT4 再减半；权重大小 = 参数量 × 每参数字节，纯线性
- 70B FP16 ≈ 140 GB > 80 GB：**单卡物理放不下**。出路只有两条：量化到 INT4（≈35 GB），或张量并行 TP 切分到多卡
- 这解释了为什么 70B 级模型发布当天就配套了 INT4/GQA 等压缩技术
"""))

t3.append(md("""
---
## E2 单请求 KV-Cache：序列长度如何吃掉显存

**预测区**：Llama3-8B 在 4K 上下文时单个请求的 KV-Cache ≈ ____ MB？128K 时 ≈ ____ GB？

提示：先手算每 token 字节数 `2 × 32 × 8 × 128 × 2`。
"""))

t3.append(code('''
# E2: mlsysim.physics.calc_kv_cache_size —— 注意 n_heads 传 KV 头数（GQA）
per_tok = 2 * llama8b.layers * llama8b.kv_heads * head_dim_8b * 2
print(f"每 token 每请求 KV = 2x{llama8b.layers}x{llama8b.kv_heads}x{head_dim_8b}x2 "
      f"= {per_tok} B = {per_tok/1024:.0f} KB")
print()

rows = []
for seq in [2048, 4096, 32768, 131072]:
    kv = calc_kv_cache_size(
        n_layers=llama8b.layers, n_heads=llama8b.kv_heads, head_dim=head_dim_8b,
        seq_len=seq, batch_size=1, bytes_per_elem=2,
    )
    # 交叉验证：ServingModel 内部用的正是同一个公式
    r = solver.solve(llama8b, h100, seq_len=seq, batch_size=1, precision="fp16")
    ratio = kv.to("GB").magnitude / llama8b.size_in_bytes().to("GB").magnitude
    rows.append([f"{seq//1024}K", f"{kv.to('GB'):.3f} GB",
                 f"{r.kv_cache_size.to('GB'):.3f} GB", f"{ratio:.0%}"])

table(["序列长度", "手算 calc_kv_cache_size", "ServingModel 输出", "占 FP16 权重比例"], rows)

# 最常见的错误：把 attention 头数当成 KV 头数
kv_wrong = calc_kv_cache_size(n_layers=llama8b.layers, n_heads=llama8b.heads,
                              head_dim=head_dim_8b, seq_len=4096, batch_size=1)
kv_right = calc_kv_cache_size(n_layers=llama8b.layers, n_heads=llama8b.kv_heads,
                              head_dim=head_dim_8b, seq_len=4096, batch_size=1)
print()
print(f"正确 (kv_heads=8) : {kv_right.to('MB'):.0f} MB @4K")
print(f"错误 (heads=32)   : {kv_wrong.to('MB'):.0f} MB @4K  <- 高估 {(kv_wrong/kv_right).magnitude:.0f} 倍")
'''))

t3.append(md("""
**E2 观察要点**：

- KV-Cache 与序列长度**严格线性**：2K→128K 增长 64 倍，没有技巧能绕过这个量级
- 128K 时单请求 KV ≈ 16.8 GB，**已经和 8B 模型的 FP16 权重一样大**——“隐藏的显存杀手”名副其实
- GQA（8 个 KV 头）已经把缓存缩小到 1/4；没有 GQA 的老架构会更早撞墙
"""))

t3.append(md("""
---
## E3 显存预算 → 最大并发请求数（H100 80GB + Llama3-8B FP16）

**预测区**：4K 上下文时一张 H100 最多能同时服务几个请求？____

预算公式：`max_concurrent = (显存容量 − 模型权重 − 预留) ÷ 单请求KV(S)`
"""))

t3.append(code('''
# E3: KV-Cache 公式 + 显存预算
RESERVE = Q_("2 GB")   # 激活值 / CUDA 上下文 / 框架开销的教学近似
weights = llama8b.size_in_bytes()
available = h100.memory.capacity - weights - RESERVE
print(f"可用 KV 预算 = {h100.memory.capacity.to('GB'):.1f} - {weights.to('GB'):.1f} "
      f"- {RESERVE.to('GB'):.0f} = {available.to('GB'):.1f} GB")
print()

rows = []
for seq in [2048, 4096, 32768, 131072]:
    kv1 = calc_kv_cache_size(llama8b.layers, llama8b.kv_heads, head_dim_8b,
                             seq_len=seq, batch_size=1)
    max_b = int(available.to("GB").magnitude // kv1.to("GB").magnitude)
    rows.append([f"{seq//1024}K", f"{kv1.to('GB'):.2f} GB", max_b])

table(["上下文长度", "单请求 KV", "最大并发请求数"], rows)

# 交叉验证 4K 答案：ServingModel 的 feasible = 权重+KV <= 显存（不含预留项，所以略高）
b = 1
while solver.solve(llama8b, h100, seq_len=4096, batch_size=b).feasible:
    b += 1
print()
print(f"ServingModel 逐步试探的 4K 可行上界: batch = {b-1}")
'''))

t3.append(md("""
**E3 观察要点**：

- 2K 时能服务上百路并发，128K 时只剩个位数——**并发上限随上下文长度线性崩塌**
- 决定“能服务多少用户”的不是算力、不是权重，而是 **显存容量 − KV 占用**
- 所以模型卡片上的 “128K context” 不是功能而是**一张内存账单**：生产系统必须靠 PagedAttention、KV 量化、prefix 复用来省钱
"""))

t3.append(md("""
---
## E4 量化对 Decode 阶段（ITL）的影响

**预测区**：INT8 / INT4 相对 FP16 的 ITL 加速比 ≈ ____ / ____
"""))

t3.append(code('''
# E4: ServingModel.solve(precision=...) —— Decode 用 ITL 衡量
rows, base = [], None
for prec in ["fp16", "int8", "int4"]:
    r = solver.solve(llama8b, h100, seq_len=4096, batch_size=1, precision=prec)
    itl = r.itl.to("ms").magnitude
    base = base or itl
    rows.append([prec.upper(), f"{itl:.2f} ms",
                 f"{r.model_weights_size.to('GB'):.1f} GB",
                 f"{r.kv_cache_size.to('GB'):.2f} GB",
                 f"{base/itl:.2f}x"])

table(["精度", "ITL (seq=4K, batch=1)", "权重", "KV-Cache", "加速比 vs FP16"], rows)
'''))

t3.append(md("""
**E4 观察要点**：加速比几乎精确等于**字节缩减比**（INT8≈2x，INT4≈4x）。

原因：batch=1 的 decode 是教科书级 memory-bound——每生成一个 token 都要把全部权重从 HBM 重读一遍，
`ITL ≈ (权重字节 + KV 字节) / 显存带宽`。少搬字节 = 等比例提速。
"""))

t3.append(md("""
---
## E5 同样的量化对 Prefill 阶段（TTFT）呢？

**预测区**：TTFT 的 INT4 加速比 ≈ ____（提示：prefill 耗时由什么决定？）
"""))

t3.append(code('''
# E5: 验证“量化对计算受限阶段无效”
rows, base = [], None
for prec in ["fp16", "int8", "int4"]:
    r = solver.solve(llama8b, h100, seq_len=4096, batch_size=1, precision=prec)
    ttft = r.ttft.to("ms").magnitude
    base = base or ttft
    rows.append([prec.upper(), f"{ttft:.1f} ms", f"{base/ttft:.2f}x"])

table(["精度", "TTFT (seq=4K, batch=1)", "加速比 vs FP16"], rows)
'''))

t3.append(md("""
**E5 解读（本任务最重要的概念点）**：

- **INT4 ≈ 1.0x（“0 倍加速”）**：prefill 耗时 = FLOPs ÷ 有效算力。量化不改变 FLOPs，而 mlsysim 的 H100 数据表里没有 INT4 算力条目（回退 FP16 峰值 989 TFLOP/s），所以 TTFT 纹丝不动
- **INT8 可能 ≈ 2x**：H100 注册表里有 `int8: 1979 TOPS`（2× FP16 峰值）。compute-bound 的 prefill 读到这条更快的低精度路径就被加速了——这正是官方教程 “Nuance: INT8 Tensor Cores” 警告的**二阶效应**
- **结论**：量化首先省的是**字节**。能否转化为加速，取决于撞的是哪个屋顶——memory roof（decode，直接受益）还是 compute roof（prefill，默认不受益，除非硬件有更快低精度算力）。训练同理：大批量训练是 compute-bound，所以“INT4 训练 4 倍加速”是谎言
"""))

t3.append(md("""
---
## O1（选做）模型规模 8B → 70B：TTFT 和 ITL 各涨多少？

**预测区**：两个指标的涨幅都 ≈ 参数量比（8.8×）吗？还是会有差别？70B FP16 还可行吗？
"""))

t3.append(code('''
# O1: 先预测，再验证（对应 Quantization 教程 Exercise 1）
def probe(model):
    r = solver.solve(model, h100, seq_len=4096, batch_size=1, precision="fp16")
    return r.ttft.to("ms").magnitude, r.itl.to("ms").magnitude, r.feasible

ttft8, itl8, _ = probe(llama8b)
ttft70, itl70, ok70 = probe(llama70b)
param_ratio = (llama70b.parameters / llama8b.parameters).magnitude

print(f"参数量比 70B/8B : {param_ratio:.1f}x")
print(f"{'':8s}{'8B':>10s} {'70B':>10s} {'倍数':>8s}")
print(f"{'TTFT':8s}{ttft8:>8.1f}ms {ttft70:>8.1f}ms {ttft70/ttft8:>7.1f}x")
print(f"{'ITL':8s}{itl8:>8.2f}ms {itl70:>8.2f}ms {itl70/itl8:>7.1f}x")
print()
print(f"70B FP16 单卡可行: {ok70}   (E1 已解释: 140GB > 80GB)")
'''))

t3.append(md("""
**O1 解读**：

- TTFT ∝ FLOPs ∝ 参数量 → 约 8.8×；ITL ∝ 权重字节 → 也约 8.8×
- 规模放大**不会改变两阶段的瓶颈属性**（prefill 仍 compute-bound、decode 仍 memory-bound），只是把离墙的距离缩短了 8.8 倍——原本宽裕的显存预算瞬间见底（70B FP16 直接不可行）
- “模型变大”的系统后果是双重的：每一步更慢 + 能服务的并发更少（70B 有 80 层，每 token KV 是 8B 的 2.5 倍）
"""))

t3.append(md("""
---
## O2（选做）INT4 加速比存在临界 batch size 吗？

issue 预期：batch 变大后 decode 从 memory-bound 滑向 compute-bound，INT4 加速比会跌破 2x / 1.5x。

**预测区**：你认为临界 batch ≈ ____
"""))

t3.append(code('''
# O2: 扫 batch size 1->128，对比 FP16 vs INT4 的 ITL（对应 Quantization 教程 Exercise 2）
batches = [1, 2, 4, 8, 16, 32, 64, 96, 128]
rows, sp_hist = [], []
for b in batches:
    r16 = solver.solve(llama8b, h100, seq_len=4096, batch_size=b, precision="fp16")
    if not r16.feasible:
        rows.append([b, "OOM", "-", "-", "-"])
        continue
    r4 = solver.solve(llama8b, h100, seq_len=4096, batch_size=b, precision="int4")
    itl16 = r16.itl.to("ms").magnitude
    itl4  = r4.itl.to("ms").magnitude
    sp = itl16 / itl4
    sp_hist.append((b, sp))
    rows.append([b, f"{itl16:.2f} ms", f"{itl4:.2f} ms", f"{sp:.2f}x",
                 f"{r16.total_memory_required.to('GB'):.0f} / {cap_gb:.0f} GB"])

table(["Batch", "ITL FP16", "ITL INT4", "加速比", "FP16 显存占用/容量"], rows)

below = lambda th: next((bb for bb, s in sp_hist if s < th), None)
print()
print(f"加速比首次 < 2x 的 batch : {below(2)}")
print(f"加速比首次 < 1.5x 的 batch: {below(1.5)}")
'''))

t3.append(code('''
import matplotlib.pyplot as plt

if sp_hist:
    xs, ys = zip(*sp_hist)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(xs, ys, marker="o", color="#2563eb")
    ax.axhline(4.0, ls="--", c="gray", lw=1)
    ax.axhline(2.0, ls=":", c="red", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size (log2)")
    ax.set_ylabel("INT4 ITL speedup (x)")
    ax.set_title("INT4 vs FP16 decode speedup under memory budget")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.show()
'''))

t3.append(md("""
**O2 如实解读（一阶模型的边界，也是很好的学习素材）**：

- 在 decode 一阶公式 `ITL = (W + KV) / BW + 框架税` 里，W 和 KV **都按精度同比缩放**，所以加速比理论上恒等于字节比 ≈ 4x，只会被固定的框架税（32 层 × 0.01ms）略微拉低
- 实际发生的是：FP16 先撞 **OOM 边界**（看表中显存占用列，4K 上下文大约在 batch≈130 之后不可行），而 INT4 权重和 KV 都缩到 1/4，可行域大得多
- 真实系统中“加速比随 batch 跌破 2x”来自三个此模型未建模的因素：① 大 batch 后 decode 进入 compute-bound，受 Tensor Core 峰值限制；② TP>1 时通信不随精度缩减；③ 调度/采样开销固定。想看瓶颈迁移，用下面的 Engine Roofline 视角补充观察
"""))

t3.append(code('''
# 补充视角：Engine.solve 的 Roofline 判断（观察大 batch 下瓶颈从 Memory -> Compute）
for b in [1, 16, 64]:
    p16 = mlsysim.Engine.solve(llama8b, h100, batch_size=b, precision="fp16")
    p4  = mlsysim.Engine.solve(llama8b, h100, batch_size=b, precision="int4")
    sp = p16.latency.to("ms").magnitude / p4.latency.to("ms").magnitude
    print(f"batch={b:>3} | FP16: {p16.bottleneck:<8} {p16.latency.to('ms'):8.2f} | "
          f"INT4: {p4.bottleneck:<8} {p4.latency.to('ms'):8.2f} | speedup={sp:.2f}x")
'''))

t3.append(md("""
---
## O3（选做）CompressionModel：量化 vs 剪枝的压缩比-精度权衡

注意：`CompressionModel` 以 **FP32 (4 字节)** 为基线计算压缩比，所以 INT8 显示 4x、INT4 显示 8x。
"""))

t3.append(code('''
# O3: 量化(INT8/INT4) vs 剪枝(sparsity=0.5/0.75/0.9)（对应 Quantization 教程 Exercise 3）
comp = CompressionModel()
rows = []

for bits in [8, 4]:
    c = comp.solve(llama8b, h100, method="quantization", target_bitwidth=bits)
    rows.append([f"量化 INT{bits}", f"{c.compression_ratio:.0f}x",
                 f"{c.compressed_size_gb.to('GB'):.1f} GB",
                 f"{c.estimated_accuracy_delta:+.2%}",
                 f"{c.inference_speedup:.1f}x"])

for sp, stype in [(0.5, "unstructured"), (0.75, "unstructured"), (0.9, "unstructured"),
                  (0.5, "structured"), (0.75, "structured"), (0.9, "structured")]:
    c = comp.solve(llama8b, h100, method="pruning", sparsity=sp, sparsity_type=stype)
    rows.append([f"剪枝 {sp:.0%} ({stype})", f"{c.compression_ratio:.1f}x",
                 f"{c.compressed_size_gb.to('GB'):.1f} GB",
                 f"{c.estimated_accuracy_delta:+.2%}",
                 f"{c.inference_speedup:.1f}x"])

table(["方案", "压缩比(vs FP32)", "压缩后大小", "估计精度变化", "推理加速"], rows)
'''))

t3.append(md("""
**O3 解读框架**（结合表格数值作答）：

- **INT8 是性价比之王**：<1% 精度损失，换 2× 内存 + decode 2× 加速，几乎无条件值得
- **INT4**：8× 压缩但精度税 2–5%，适合对延迟/显存极度敏感、能接受质量回退的场景（或配合 QAT/LoRA 回收精度）
- **非结构化剪枝**：只有存储收益，`inference_speedup=1.0`——稀疏权重在没有专用硬件/内核时跑不出加速
- **结构化剪枝**：稀疏度越高加速越大，但经验规律是超过 ~50% 后精度断崖（对比 75%/90% 的 accuracy delta），风险显著高于量化
- 结论句式建议：“在我的约束（显存上限 / 精度下限）下，____ 是最优先手段，因为 ______”
"""))

t3.append(md("""
---
## O4（选做）70B 部署可行性报告素材（8×H100 节点）

思路：TP=8 把权重切到每卡；GQA 的 8 个 KV head 正好每卡分摊 1 个；剩下的显存全部用来买并发。
"""))

t3.append(code('''
# O4: FP16 vs INT4 在 8-GPU H100 节点上的并发容量（TP=8）
reserve_gb = 2.0

w70_fp16_gb = llama70b.size_in_bytes().to("GB").magnitude
w70_int4_gb = llama70b.size_in_bytes(ureg("0.5 byte")).to("GB").magnitude

# TP=8: kv_heads=8 -> 每卡 1 个 KV head；FP16 KV 每元素 2B
kv70_tok_card_fp16_gb = 2 * llama70b.layers * (llama70b.kv_heads // 8) * 128 * 2 / 1e9
kv70_tok_card_int4_gb = kv70_tok_card_fp16_gb / 4   # 权重与 KV 均按 4-bit 计

rows = []
for prec, w_pc, kv_tok in [("FP16", w70_fp16_gb / 8, kv70_tok_card_fp16_gb),
                           ("INT4", w70_int4_gb / 8, kv70_tok_card_int4_gb)]:
    for seq in [2048, 4096, 16384]:
        kv1 = kv_tok * seq
        free = cap_gb - w_pc - reserve_gb
        rows.append([prec, f"{seq//1024}K", f"{w_pc:.1f}", f"{kv1:.3f}",
                     int(free // kv1)])

table(["精度", "上下文", "每卡权重 GB", "每请求 KV GB (TP=8)", "每卡最大并发"], rows)
print()
print("节点总并发 = 每卡并发 x 8 卡")
'''))

t3.append(md("""
**O4 报告模板**（把上表数字填进去就是一份合格的部署建议）：

1. **用什么精度？** 建议 ____ 起步：每卡权重仅 ____ GB，给 KV 留出 ____ GB 预算；若精度敏感可退回 FP16 + 更短上下文
2. **最大支持多长上下文？** 16K 时 FP16 每卡并发跌至 ____，INT4 仍有 ____ → 生产上以 ____ 为宜
3. **最多服务多少并发？** 4K 目标下 8 卡节点合计约 ____ 路；要再往上需 PagedAttention（消除碎片 +20–40%）或 KV INT8（再翻倍）
4. **风险提示**：以上为解析仿真的一阶估算，未含激活值峰值、TP 通信对 ITL 的拖累与碎片化；上线前需真实流量压测
"""))

t3.append(md("""
---
## 打卡对照清单（issue #133）

**最小打卡（E1–E3）**
- [ ] E1：三种精度权重大小表 + “70B FP16 放不进单张 H100”的一句话证明
- [ ] E2：四档序列长度 KV 表 + 线性增长说明
- [ ] E3：各长度最大并发表 + “显存容量是核心约束”一段话

**学有余力 1（+E4 E5 O1 O2）**
- [ ] E4/E5：两张精度对比表 + 解释“为什么 Decode 4x 而 Prefill 0x”（记得提 INT8 二阶效应）
- [ ] O1：8B→70B 的 TTFT/ITL 倍数
- [ ] O2：加速比-batch 曲线 + 你找到的临界点/OOM 边界

**学有余力 2（+O3 O4）**
- [ ] O3：量化 vs 剪枝对比表 + 性价比结论
- [ ] O4：三问部署建议（精度/上下文/并发）

> 截图建议：保留每个 `table(...)` 输出与关键 `print` 结果，GitHub 打卡评论按 E1→O4 顺序贴图。
"""))

# ======================================================================
# Task 4
# ======================================================================
t4 = []

t4.append(md("""
# MLSys Task 4：数据、算法与系统优化——如何系统性地优化？

对应打卡 issue：[datawhalechina/llm-algo-leetcode #136](https://github.com/datawhalechina/llm-algo-leetcode/issues/136)

理论材料：

- [Starving the GPU](https://harvard-edge.github.io/cs249r_book_dev/mlsysim/tutorials/04_starving_the_gpu.html)（数据流水线 / Data Wall）
- [Design Space Exploration](https://harvard-edge.github.io/cs249r_book_dev/mlsysim/tutorials/12_design_space_exploration.html)（声明式搜索）
- Module 2: Advanced Single-Node Analysis（选读 PDF：投机解码 / Inference-Time Compute）

三个优化维度 ↔ 三个工具：

| 维度 | 工具 | 关键问题 |
|---|---|---|
| 数据流水线 | `DataModel` / `TransformationModel` | CPU 喂不饱 GPU？ |
| 算法优化 | `ServingModel(draft_model=...)` / `InferenceScalingModel` | 投机解码赚不赚？推理时计算多贵？ |
| 设计空间探索 | `DSE` 引擎 | 千种配置怎么自动找最优？ |

全部实验为解析仿真，Colab CPU runtime 可跑。
"""))

t4.append(code(INSTALL))

t4.append(code('''
import warnings, math
warnings.filterwarnings("ignore")

import mlsysim
from mlsysim import ureg
from mlsysim.core.units import Q_
from mlsysim.solvers import (SingleNodeModel, DataModel, TransformationModel,
                             ServingModel, DistributedModel, EconomicsModel,
                             InferenceScalingModel)
from mlsysim.show import table, info
from mlsysim.engine.dse import DSE
from mlsysim.engine.pipeline import Pipeline

print("mlsysim 版本:", mlsysim.__version__)

resnet50 = mlsysim.Models.Vision.ResNet50
a100, h100 = mlsysim.Hardware.Cloud.A100, mlsysim.Hardware.Cloud.H100
llama8b, llama70b = mlsysim.Models.Language.Llama3_8B, mlsysim.Models.Language.Llama3_70B
llama2_7b = mlsysim.Models.Language.Llama2_7B

gpu_solver   = SingleNodeModel()
data_solver  = DataModel()
xform_solver = TransformationModel()
serve        = ServingModel()


def mag(x, unit=None):
    """统一取数值：Quantity 可选转换单位；裸数字直接返回（兼容不同字段的类型差异）。"""
    if hasattr(x, "magnitude"):
        return x.to(unit).magnitude if unit else x.magnitude
    return x

'''))

t4.append(md("""
---
## E1 GPU 纯计算时间：你以为的天花板

训练一步有三个串联阶段：① 存储 I/O → ② CPU 预处理 → ③ GPU 计算。**最慢的一环决定吞吐**。
先单独测出 ③，作为后续对比的天花板。

**预测区**：ResNet-50 在 A100 上 batch=256 的单步训练耗时 ≈ ____ ms？吞吐 ≈ ____ img/s？
"""))

t4.append(code('''
# E1: SingleNodeModel = Engine.solve 的 resolver 封装；is_training=True 含反向传播
profile256 = gpu_solver.solve(resnet50, a100, batch_size=256, precision="fp16",
                              efficiency=0.5, is_training=True)

info("GPU 计算基线（天花板）",
     Model=resnet50.name,
     Hardware=a100.name,
     Step_latency=profile256.latency.to("ms"),
     Throughput=f"{profile256.throughput:.0f} img/s",
     Bottleneck=profile256.bottleneck)
'''))

t4.append(md("""
---
## E2 存储 I/O 检查：磁盘 / PCIe 供得上吗？

ImageNet JPEG 平均 ~500 KB。GPU 每步吞掉 `batch × sample` 字节，折算成速率后与硬件数据通路比较。

**预测区**：按 E1 步时间算，数据需求速率 ≈ ____ GB/s；A100 的 PCIe Gen4 x16（32 GB/s）会被打满吗？
"""))

t4.append(code('''
# E2: DataModel —— 比较“需求速率 vs 最慢数据通路”
SAMPLE = Q_("500 KB")
step_s = profile256.latency.to("s").magnitude
demand = Q_(256 * SAMPLE.to("GB").magnitude / step_s, "GB/s")

r = data_solver.solve(workload_data_rate=demand, hardware=a100)
info("存储 I/O 检查（按实际需求）",
     数据需求=f"{demand:.2f}",
     供给=f"{r.supply_bw:.2f}",
     瓶颈链路=r.bottleneck,
     利用率=f"{r.utilization:.1%}",
     是否停顿=r.is_stalled)

# issue 指定场景：给定 6 GB/s，PCIe 扛得住吗？（再顺手试一个超载值）
print()
for rate_str in ["6 GB/s", "40 GB/s"]:
    rr = data_solver.solve(workload_data_rate=Q_(rate_str), hardware=a100)
    print(f"需求 {rate_str:>8}: 供给 {rr.supply_bw:.0f} | 利用率 {rr.utilization:.1%} | "
          f"stalled={rr.is_stalled}")
'''))

t4.append(md("""
**E2 说明**：注册表里 `Hardware.Cloud.A100` 的数据通路是 **PCIe Gen4 x16 = 32 GB/s**（NVMe 属于 DGX 整机层，不在加速器对象里）。

所以 6 GB/s 的需求利用率只有约 19%，存储 I/O 不是瓶颈——真正的问题在下一个环节。
"""))

t4.append(md("""
---
## E3 CPU 预处理检查：8 个 worker vs 64 个 worker

JPEG 解码 / 裁剪 / 增广都在 CPU 上。典型单 worker 吞吐 ≈ 250 MB/s。

**预测区**：batch=512 时 8 worker（2 GB/s）会让 GPU 挨饿吗？消除瓶颈最少要几个 worker？____
"""))

t4.append(code('''
# E3: TransformationModel —— CPU 变换时间 vs GPU 步时间
prof512 = gpu_solver.solve(resnet50, a100, batch_size=512, precision="fp16",
                           efficiency=0.5, is_training=True)
BS = 512

rows = []
for n in [1, 2, 4, 8, 16, 32, 64]:
    t = xform_solver.solve(batch_size=BS, sample_size_bytes=SAMPLE,
                           cpu_throughput=Q_(f"{n*250} MB/s"),
                           accelerator_step_time=prof512.latency)
    rows.append([n,
                 f"{t.transform_time.to('ms').magnitude:.1f} ms",
                 f"{prof512.latency.to('ms').magnitude:.1f} ms",
                 "CPU 瓶颈" if t.is_bottleneck else "OK",
                 f"{t.accelerator_utilization:.1%}"])

table(["CPU workers", "CPU 变换耗时", "GPU 步时间", "判定", "GPU 利用率"], rows)

# 解析法求最少 worker 数：所需吞吐 = 每步数据量 / GPU 步时间
need_Bps = BS * SAMPLE.to("B").magnitude / prof512.latency.to("s").magnitude
min_workers = math.ceil(need_Bps / 250e6)
print()
print(f"解析解：需要 >= {need_Bps/1e9:.2f} GB/s -> 最少 {min_workers} 个 worker")

# 验证：刚好配足时 CPU 不再是瓶颈
t_min = xform_solver.solve(batch_size=BS, sample_size_bytes=SAMPLE,
                           cpu_throughput=Q_(f"{min_workers*250} MB/s"),
                           accelerator_step_time=prof512.latency)
print(f"验证：{min_workers} workers -> is_bottleneck={t_min.is_bottleneck}, "
      f"GPU 利用率={t_min.accelerator_utilization:.1%}")
'''))

t4.append(md("""
**E3 解读 —— 什么是 Data Wall**：

- GPU 利用率 = `GPU步时间 / max(CPU变换时间, GPU步时间)`。CPU 慢一截，GPU 就空转一截——**利用率只有 40% 时问题往往不在 GPU**
- 小批量时 GPU 步时间长，CPU 来得及准备；大批量时 GPU 越来越快、CPU 负载线性涨，瓶颈必然翻转（看上表的 crossover）
- 加 worker 是线性解药，但最终会顶到存储 I/O 或 PCIe（E2 的通路）——所以要**三段一起查**，而不是盯着 nvidia-smi 猜
"""))

t4.append(md("""
---
## E4 投机解码：Draft 8B 验证 70B

原理：小 draft 模型一次猜 K=4 个 token（便宜），大 target 一次并行验证（把 memory-bound 的权重读取摊到多个 token 上）。
期望接受 token 数 `E = 1 + α(1−α^K)/(1−α)`，α 为接受率。

**预测区**：α=0.75、K=4 时 ITL 加速比 ≈ ____ x？（先手算 E）
"""))

t4.append(code('''
# E4: ServingModel(draft_model=..., draft_acceptance_rate=...)
SEQ = 2048
base = serve.solve(llama70b, h100, seq_len=SEQ, batch_size=1, precision="fp16")
spec = serve.solve(llama70b, h100, seq_len=SEQ, batch_size=1, precision="fp16",
                   draft_model=llama8b, draft_acceptance_rate=0.75)

K = 4            # mlsysim 内置 SPECULATIVE_GAMMA=4
alpha = 0.75
expected = 1 + alpha * (1 - alpha**K) / (1 - alpha)

itl_base = base.itl.to("ms").magnitude
itl_spec = spec.itl.to("ms").magnitude
print(f"无投机解码 ITL : {itl_base:.2f} ms")
print(f"投机解码   ITL : {itl_spec:.2f} ms")
print(f"加速比          : {itl_base/itl_spec:.2f}x")
print(f"理论期望 token/步 E = {expected:.2f} （粗略上限 ≈ {expected:.1f}x，实际还要扣 draft 开销）")
extra_mem = spec.total_memory_required.to("GB").magnitude - base.total_memory_required.to("GB").magnitude
print(f"代价：显存里要多住一个 draft（+{extra_mem:.1f} GB）")
'''))

t4.append(md("""
---
## O1（选做）接受率的权衡：draft 太小 vs 太大

扫 α ∈ [0.5, 0.9]。两个极端都不好：

- **α 小**（draft 与 target 分布差太远，通常因为 draft 太小）：猜 4 个只中零星几个，draft 阶段白跑
- **α → 1 需要 draft 很大**：draft 阶段自身耗时逼近 target，即使全收也没赚头
"""))

t4.append(code('''
# O1: acceptance rate sweep
rows = []
for al in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
    s = serve.solve(llama70b, h100, seq_len=SEQ, batch_size=1, precision="fp16",
                    draft_model=llama8b, draft_acceptance_rate=al)
    itl = s.itl.to("ms").magnitude
    expected = 1 + al * (1 - al**K) / (1 - al)
    rows.append([al, f"{expected:.2f}", f"{itl:.2f} ms", f"{itl_base/itl:.2f}x"])

table(["acceptance rate", "E[token/步]", "有效 ITL", "加速比"], rows)
'''))

t4.append(md("""
---
## E5 DSE：三维网格自动搜最优并行配置

不用手写三层 for 循环：声明搜索空间 + 目标函数，交给 `DSE` 引擎穷举（解析仿真毫秒级完成）。
配置：Llama3-70B 训练，256×H100 集群（Research_256，Ethernet 100G），目标最大化吞吐。

**预测区**：最优配置会是“TP 拉满”吗？通信和气泡谁占大头？
"""))

t4.append(code('''
# E5: Design Space Exploration
research = mlsysim.Systems.Clusters.Research_256     # 256 x H100, Ethernet 100G

def eval_config(params):
    pipe = Pipeline([DistributedModel()])
    return pipe.run(model=llama70b, fleet=research,
                    batch_size=params["batch_size"],
                    tp_size=params["tp"], pp_size=params["pp"],
                    precision="fp16", efficiency=0.45, seq_len=2048)

space = {"batch_size": [1, 4, 16, 64], "tp": [1, 2, 4], "pp": [1, 2]}
dse = DSE(space=space, objective="maximize: DistributedModel.effective_throughput")
result = dse.search(eval_config)

bp = result["best_params"]
print(f"最优配置: TP={bp['tp']}, PP={bp['pp']}, Batch={bp['batch_size']} "
      f"-> {result['best_objective']:.1f}")
print()
print(f"{'TP':>3} | {'PP':>2} | {'Batch':>5} | {'吞吐(1/s)':>10} | {'步时(ms)':>9} | {'通信占比':>7} | {'气泡占比':>7}")
print("-" * 66)
for cand in result["top_candidates"][:3]:
    p = cand["params"]
    d = cand["result"]["DistributedModel"]
    step_ms = mag(d.step_latency_total, "ms")
    thr = mag(d.effective_throughput)
    comm_pct = mag(d.communication_latency, "ms") / step_ms
    print(f"{p['tp']:>3} | {p['pp']:>2} | {p['batch_size']:>5} | "
          f"{thr:>10.1f} | {step_ms:>9.1f} | "
          f"{comm_pct:>6.0%} | {d.bubble_fraction:>6.0%}")
'''))

t4.append(md("""
**E5 怎么解释最优配置**（对照输出作答）：

- 看 top 配置的三个分量谁在主导：DP AllReduce（跨节点梯度同步）、流水线气泡、还是本地 Roofline 计算；
- 直觉校验：TP 越大 → 每卡分片越小，但 Research_256 的 fabric 是 Ethernet 100G（不是 NVLink），TP AllReduce 的惩罚很重——这就是“TP 尽量限制在 NVLink 域内”的行业经验在数据里的体现；
- 注意 DSE 自动跳过了非法配置（如 TP×PP 不能整除 256 时抛异常即丢弃）——这就是“声明式搜索替代嵌套循环”的价值。
"""))

t4.append(md("""
---
## O2（选做）给 DSE 加 SLA 约束

吞吐最大的配置往往延迟爆炸。DSE 支持 `<metric> <op> <threshold>` 语法的硬约束，观察最优解如何移动。
"""))

t4.append(code('''
# O2: SLA-constrained DSE（把结果包装成扁平浮点字段，避免单位歧义）
class Cand:
    def __init__(self, params, d):
        self.params = params
        self.throughput = mag(d.effective_throughput)
        self.step_latency_ms = mag(d.step_latency_total, "ms")

def eval_cand(params):
    d = eval_config(params)["DistributedModel"]
    return Cand(params, d)

for sla_ms in [50, 200, 500, 2000, 10000]:
    dse_sla = DSE(space=space, objective="maximize: throughput",
                  constraints=[f"step_latency_ms < {sla_ms}"])
    try:
        res = dse_sla.search(eval_cand)
        p = res["best_params"]
        print(f"SLA < {sla_ms:>6} ms -> 最优 TP={p['tp']} PP={p['pp']} Batch={p['batch_size']} | "
              f"吞吐 {res['best_objective']:.1f}/s | 步时 {res['best_result'].step_latency_ms:.0f} ms")
    except ValueError:
        print(f"SLA < {sla_ms:>6} ms -> 无可行配置（整个搜索空间都被延迟约束杀死）")
'''))

t4.append(md("""
**O2 解读**：

- SLA 太紧 → 全军覆没：“吞吐最大化”和“延迟达标”在给定集群上是**对立目标**，只能靠扩集群或改算法调和
- SLA 逐级放宽时观察最优解迁移：约束紧时被迫选小 batch / 低并行度（步时短但吞吐低）；放宽后立刻跳到大 batch 高吞吐配置——**约束改变的是最优解的位置，而不只是过滤掉几个点**
"""))

t4.append(md("""
---
## O3（选做）InferenceScalingModel：o1 式“推理时计算”有多贵

o1-style 模型先生成 K 步隐藏推理（每步 ~50 token）再给答案：`T = TTFT + K × T_step`。
**算法选择直接变成基础设施账单**。
"""))

t4.append(code('''
# O3: K=1/8/32 的隐藏推理成本（K=1 即“无推理计算”的基线）
cot = InferenceScalingModel()
rows, base_t = [], None
for kk in [1, 8, 32]:
    rr = cot.solve(model=llama8b, hardware=h100, reasoning_steps=kk,
                   context_length=2048, precision="fp16")
    total_s = rr.total_reasoning_time.to("s").magnitude
    base_t = base_t or total_s
    rows.append([kk,
                 f"{rr.ttft.to('ms').magnitude:.0f} ms",
                 f"{total_s:.2f} s",
                 rr.tokens_generated,
                 f"{rr.energy_per_query.to('J'):.0f} J",
                 f"{total_s/base_t:.1f}x"])

table(["K 步隐藏推理", "TTFT", "总时延", "生成 tokens", "每查询能耗", "vs K=1"], rows)
'''))

t4.append(md("""
**O3 解读**：

- K 步成本 ≈ K × 50 tokens × ITL，TTFT 只是固定的一小块 → 总时延倍数略小于 K 并随 K 趋近 K
- 系统含义：**算力需求从训练转移到推理**。训练是一次性 CAPEX，推理时计算却按每条查询持续烧钱（QPS × 时延 → GPU 数量；详见 Task5 的 9M Question）
- 工程对策预告：路由（简单问题走小模型）、投机解码压 ITL、批处理摊带宽
"""))

t4.append(md("""
---
## O4（选做）端到端优化报告骨架：日活 100 万的 LLM 聊天应用

场景：LLaMA-3-8B，平均输入 2K tokens、输出 256 tokens，P99 ITL < 80 ms。
下面四个分析块可直接运行，跑完后把数字填进最后的结论模板。
"""))

t4.append(code('''
# O4-A 场景基线：负载估算 + 单卡服务能力
DAU = 1_000_000
queries_per_user = 10                      # 教学假设
qps_avg = DAU * queries_per_user / 86400
qps_peak = qps_avg * 3                     # 峰均比 3x
print(f"平均 QPS ~= {qps_avg:.0f}，峰值 QPS ~= {qps_peak:.0f}")

ctx = 2048 + 256
base8b = serve.solve(llama8b, h100, seq_len=ctx, batch_size=1, precision="fp16")
itl_ms = base8b.itl.to("ms").magnitude
print(f"单卡 batch=1: TTFT {base8b.ttft.to('ms').magnitude:.0f} ms, "
      f"ITL {itl_ms:.1f} ms -> 输出 256 token 需 {itl_ms*256/1000:.1f} s")
'''))

t4.append(code('''
# O4-B 算法优化：投机解码在该场景的收益
spec8b = serve.solve(llama8b, h100, seq_len=ctx, batch_size=1, precision="fp16",
                     draft_model=llama2_7b, draft_acceptance_rate=0.75)
itl_spec_ms = spec8b.itl.to("ms").magnitude
verdict = "满足" if itl_spec_ms < 80 else "仍不满足"
print(f"投机解码 ITL: {itl_spec_ms:.1f} ms (加速 {itl_ms/itl_spec_ms:.2f}x) "
      f"-> {verdict} P99<80ms 的原始 ITL 预算")
'''))

t4.append(code('''
# O4-C 系统优化：mini-DSE（硬件 x batch），挑满足 ITL 预算的最大吞吐配置
best = None
rows = []
for hw_name, hw in [("A100", a100), ("H100", h100)]:
    for bsz in [1, 8, 32, 64]:
        rr = serve.solve(llama8b, hw, seq_len=ctx, batch_size=bsz, precision="fp16")
        if not rr.feasible:
            continue
        itl_i = rr.itl.to("ms").magnitude
        tok_s = bsz / (itl_i / 1000)             # 每 replica 的 token 吞吐
        rows.append([hw_name, bsz, f"{itl_i:.1f} ms", f"{tok_s:.0f} t/s",
                     "ITL 达标" if itl_i < 80 else "超时"])
        if itl_i < 80 and (best is None or tok_s > best["tok_s"]):
            best = {"hw": hw_name, "bs": bsz, "tok_s": tok_s, "itl": itl_i}

table(["硬件", "batch", "ITL", "token 吞吐/replica", "判定"], rows)

if best:
    replicas = math.ceil(qps_peak * 256 / best["tok_s"])
    print()
    print(f"推荐: {best['hw']} x batch={best['bs']} (ITL {best['itl']:.1f} ms) "
          f"-> 峰值需 ~= {replicas} 张卡")
'''))

t4.append(md("""
**O4-D 结论模板**（填空即成文）：

1. **数据流水线**：本场景输入是已 tokenize 的文本，每查询数据量仅 KB 级，Data Wall 不是瓶颈（对照 E2/E3：图像训练才容易被 CPU 卡死）
2. **算法优化**：投机解码将 ITL 从 ____ ms 压到 ____ ms（____x），单查询 256-token 输出从 ____ s 降到 ____ s
3. **系统优化**：满足 P99 ITL<80ms 的最优配置为 ____ × batch=____，单卡 token 吞吐 ____ t/s，峰值需 ____ 卡
4. **综合建议**：硬件选 ____；开启投机解码 + continuous batching；预留 ____% 冗余应对峰谷。（四块运行的数字抄进来即可提交）
"""))

t4.append(md("""
---
## 打卡对照清单（issue #136）

**最小打卡（E1–E3）**
- [ ] E1：ResNet-50/A100 纯 GPU 步时间（天花板）
- [ ] E2+E3：存储 I/O 判定、CPU 预处理判定、最少 worker 数、“Data Wall”概念一段话

**学有余力 1（+E4 E5 O1）**
- [ ] E4+O1：基准 ITL、α=0.75 加速比、α 扫描趋势与“draft 太小/太大”的解释
- [ ] E5：Top-3 配置表 + 主导因素分析（算力/内存/通信）

**学有余力 2（+O2 O3 O4）**
- [ ] O2：SLA 加入前后最优配置的变化
- [ ] O3：K=0/8/32 的 TTFT 与总时延 + “算力从训练转向推理”
- [ ] O4：端到端系统优化报告（3–4 页）
"""))

# ======================================================================
# Task 5
# ======================================================================
t5 = []

t5.append(md("""
# MLSys Task 5：集群、成本与整合——大规模系统怎么建？

对应打卡 issue：[datawhalechina/llm-algo-leetcode #137](https://github.com/datawhalechina/llm-algo-leetcode/issues/137)

理论材料：

- [Scaling to 1000 GPUs](https://mlsysbook.ai/mlsysim/tutorials/06_scaling_1000_gpus.html)（3D 并行 / 通信 / 气泡 / 可靠性）
- [The 9M Question](https://mlsysbook.ai/mlsysim/tutorials/08_the_9m_question.html)（TCO / 电费 / 碳排放）
- [Declarative DSE](https://mlsysbook.ai/mlsysim/tutorials/12_design_space_exploration.html)（复习+进阶）

新工具：

- `DistributedModel` —— 3D 并行（DP/TP/PP）、通信与气泡
- `EconomicsModel` —— Capex/Opex/TCO
- `SustainabilityModel` —— 能耗、碳、水
- `ReliabilityModel` / `CheckpointModel` —— MTBF 与 checkpoint 策略
- YAML 集群评估 —— 3-lens scorecard（Feasibility + Performance + Macro）

全部为解析仿真，Colab CPU runtime 可跑。
"""))

t5.append(code(INSTALL))

t5.append(code('''
import math
import warnings
warnings.filterwarnings("ignore")

import mlsysim
from mlsysim.core.units import Q_
from mlsysim.solvers import (DistributedModel, EconomicsModel, SustainabilityModel,
                             ReliabilityModel, CheckpointModel)
from mlsysim.show import table, info
from mlsysim.systems.types import Fleet

print("mlsysim 版本:", mlsysim.__version__)

llama70b = mlsysim.Models.Language.Llama3_70B
gpt4     = mlsysim.Models.Language.GPT4          # 1.76T total / ~280B active (MoE, 第三方估计)
DGX      = mlsysim.Systems.Nodes.DGX_H100        # 8 x H100, NVLink 900 GB/s
IB_NDR   = mlsysim.Systems.Fabrics.InfiniBand_NDR

dist = DistributedModel()
econ = EconomicsModel()
sust = SustainabilityModel()

def mk_fleet(count, name=None):
    return Fleet(name=name or f"{count} node(s)", node=DGX, count=count, fabric=IB_NDR)

fleet8  = mk_fleet(1, "1 node / 8 GPUs")
fleet32 = mk_fleet(4, "4 nodes / 32 GPUs")

# 训练态每参数字节数（混合精度 Adam）：fp16 权重 2 + 梯度 2 + FP32 主权重 4 + 动量 4 + 方差 4 = 16
BYTES_PER_PARAM_TRAIN = 16

def train_mem_gb_per_gpu(model, tp, pp):
    """一阶估计：状态按 TP*PP 分片，不含激活值（下界）。"""
    return model.parameters.to("count").magnitude * BYTES_PER_PARAM_TRAIN / (tp * pp) / 1e9


def mag(x, unit=None):
    """统一取数值：Quantity 可选转换单位；裸数字直接返回（兼容不同字段的类型差异）。"""
    if hasattr(x, "magnitude"):
        return x.to(unit).magnitude if unit else x.magnitude
    return x


W70_FP16_GB = llama70b.size_in_bytes().to("GB").magnitude
'''))

t5.append(md("""
---
## E1 单节点 8×H100：3D 并行策略怎么选？

70B FP16 权重 140 GB，训练态（Adam）约 70.6B × 16B ≈ **1.1 TB**——单卡 80 GB 必然放不下，
必须靠 TP×PP 分片。下面穷举 8 卡上所有合法 (TP, PP)：先做**内存可行性筛选**，再看效率与吞吐。

**预测区**：哪个组合会是冠军？TP=8（通信走 NVLink）一定赢吗？____
"""))

t5.append(code('''
# E1: DistributedModel 在单节点上的策略扫描（推理态 vs 训练态两本账）
rows, records_t5 = [], []
for tp, pp in [(tp, pp) for tp in (1, 2, 4, 8) for pp in (1, 2, 4, 8)
               if tp * pp <= 8 and 8 % (tp * pp) == 0]:
    r = dist.solve(model=llama70b, fleet=fleet8, batch_size=64,
                   tp_size=tp, pp_size=pp, precision="fp16", efficiency=0.45,
                   seq_len=2048, overlap_comm=True)
    w_per_gpu = W70_FP16_GB / tp                          # 推理态：只有权重
    st_per_gpu = train_mem_gb_per_gpu(llama70b, tp, pp)   # 训练态：权重+梯度+Adam
    infer_ok, train_ok = w_per_gpu <= 80, st_per_gpu <= 80
    thr = mag(r.effective_throughput)
    records_t5.append({"tp": tp, "pp": pp, "thr": thr,
                       "infer_ok": infer_ok, "train_ok": train_ok})
    rows.append([tp, pp, r.parallelism.get("dp", "?"),
                 f"{w_per_gpu:.0f}", f"{st_per_gpu:.0f}",
                 "OK" if infer_ok else "OOM",
                 "OK" if train_ok else "OOM",
                 f"{r.scaling_efficiency:.0%}",
                 f"{thr:.2f}" if infer_ok else "-"])

table(["TP", "PP", "DP", "权重GB/卡", "权重+Adam GB/卡", "推理态", "训练态",
       "扩展效率", "吞吐(样本/s)"], rows)

ok = [x for x in records_t5 if x["infer_ok"]]
best8 = max(ok, key=lambda x: x["thr"])
n_train_ok = sum(1 for x in records_t5 if x["train_ok"])
print()
print(f"单节点推理态参考最优: TP={best8['tp']}, PP={best8['pp']} (吞吐 {best8['thr']:.2f} 样本/s)")
print(f"训练态可行组合数: {n_train_ok} —— 70B+Adam 参数状态 ~1.13TB，8 卡怎么切都放不下")
'''))

t5.append(md("""
**E1 解读 —— 3D 并行的核心权衡（以及一个关键发现）**：

- **DP 降通信**：梯度 AllReduce 的消息量 ∝ 模型大小/TP；DP 不引入气泡，但受临界 batch size 限制
- **TP 降内存、走 NVLink**：权重/梯度/优化器都按 TP 分片，通信快；但看表：即使 TP=8，训练态每卡仍需 ~141 GB → **OOM**
- **“训练态全 OOM”是本题最重要的发现**：70B + Adam 的参数状态约 1.13 TB，单个 8 卡节点无论怎么切都装不下——所以真实的 70B 训练至少要跨节点分摊优化器状态（E2 的 32 卡），或用 ZeRO-Offload / 启用 checkpoint 等手段压缩；“推理放得下 ≠ 训练放得下”
- **PP 用层间串行换内存**，引入 (pp−1)/(m+pp−1) 的气泡；microbatch 越多气泡越小
"""))

t5.append(md("""
---
## E2 扩展到多节点：通信开销如何增长

先在 **32 卡**上选出**训练态内存可行**的最优 (TP, PP)，再把节点数从 1 扫到 8，
观察 DP AllReduce 与气泡占比的变化（不满足整除性的规模会自动回退到能整除的策略并标注）。

**预测区**：通信占比会从 ____ % 涨到 ____ %
"""))

t5.append(code('''
# E2-a: 在 32 卡上选“训练态内存可行”的最优策略（tp*pp >= ~15 才装得下 Adam 状态）
records32 = []
for tp, pp in [(tp, pp) for tp in (1, 2, 4, 8) for pp in (1, 2, 4, 8)
               if tp * pp <= 32 and 32 % (tp * pp) == 0]:
    if train_mem_gb_per_gpu(llama70b, tp, pp) > 80:
        continue
    r = dist.solve(model=llama70b, fleet=fleet32, batch_size=max(64, 32),
                   tp_size=tp, pp_size=pp, precision="fp16", efficiency=0.45,
                   seq_len=2048, overlap_comm=True)
    records32.append({"tp": tp, "pp": pp, "thr": mag(r.effective_throughput)})

best32 = max(records32, key=lambda x: x["thr"])
tb, pb = best32["tp"], best32["pp"]
print(f"32 卡训练态可行组合 {len(records32)} 个；最优: TP={tb}, PP={pb} "
      f"(吞吐 {best32['thr']:.2f} 样本/s)")


# E2-b: 节点数扫描。整除性不满足时回退到能整除的组合，并在表中标注。
def choose_strategy(n_gpu):
    if n_gpu % (tb * pb) == 0:
        return tb, pb
    for fb_tp, fb_pp in [(4, 2), (2, 2), (2, 1), (1, 2), (1, 1)]:
        if n_gpu % (fb_tp * fb_pp) == 0:
            return fb_tp, fb_pp
    return 1, 1


def comm_frac(rr):
    return mag(rr.communication_latency, "ms") / mag(rr.step_latency_total, "ms")


rows, dist_by_count = [], {}
for n_nodes in [1, 2, 4, 8]:
    fl = mk_fleet(n_nodes)
    n_gpu = 8 * n_nodes
    tp_i, pp_i = choose_strategy(n_gpu)
    r = dist.solve(model=llama70b, fleet=fl, batch_size=max(64, n_gpu),
                   tp_size=tp_i, pp_size=pp_i, precision="fp16",
                   efficiency=0.45, seq_len=2048, overlap_comm=True)
    dist_by_count[n_nodes] = r
    strat = f"TP{tp_i}/PP{pp_i}"
    if (tp_i, pp_i) != (tb, pb):
        strat += " (回退)"
    rows.append([n_nodes, n_gpu, strat, r.parallelism.get("dp", "?"),
                 f"{mag(r.communication_latency, 'ms'):.1f} ms",
                 f"{comm_frac(r):.1%}",
                 f"{r.bubble_fraction:.1%}",
                 f"{r.scaling_efficiency:.1%}"])

table(["节点", "GPU", "采用策略", "DP", "通信耗时", "通信占比", "气泡占比", "扩展效率"], rows)

print()
print(f"通信占比 1 节点 -> 4 节点: {comm_frac(dist_by_count[1]):.1%} -> "
      f"{comm_frac(dist_by_count[4]):.1%}；扩展效率 "
      f"{dist_by_count[1].scaling_efficiency:.1%} -> {dist_by_count[4].scaling_efficiency:.1%}")
'''))

t5.append(md("""
**E2 解读**：

- 单节点 DP=1 几乎没有 DP AllReduce；跨节点后 DP 变大，梯度要在 IB NDR（400Gb/s）上做**分层 AllReduce**（节点内 NVLink ring + 跨节点 ring），通信占比随之上涨
- 这是 **Amdahl 定律**的具象化：通信是串行分量，N 翻倍不能让步时间减半
- 注：1 节点行是“回退”策略（32 卡的最优组在 8 卡上放不下也除不尽），对比趋势时以同策略的 2→4→8 行为准
- 缓解手段（都是 `DistributedModel.solve` 的参数，可以自己再试）：`overlap_comm=True` 用计算盖住通信、`zero_stage` 分散优化器状态、增大 `microbatch_count` 压气泡
"""))

t5.append(md("""
---
## E3 EconomicsModel：32×H100 训练 70B 一个月的 TCO

**预测区**：Capex 和 Opex 谁占大头？____
"""))

t5.append(code('''
# E3: 总拥有成本 = Capex(硬件采购摊销) + Opex(电费 + 运维)
tco = econ.solve(fleet=fleet32, duration_days=30, mfu=0.45)

total = tco.tco_usd
info("TCO 分析 (32 x H100, 30 天)",
     Capex=f"${tco.capex_usd:,.0f} ({tco.capex_usd/total:.0%})",
     Opex_电费=f"${tco.opex_energy_usd:,.0f} ({tco.opex_energy_usd/total:.0%})",
     Opex_运维=f"${tco.opex_maintenance_usd:,.0f} ({tco.opex_maintenance_usd/total:.0%})",
     总计=f"${total:,.0f}",
     总能耗=f"{tco.total_energy_kwh.to('MWh'):.0f} MWh",
     区域=tco.region_name)
'''))

t5.append(md("""
**E3 解读 —— 为什么大规模训练的成本不是线性的（The 9M Question）**：

- 卡数翻倍 ≠ 成本翻倍：网络设备、整机柜、供电冷却（PUE）、运维人力都有自己的非线性阶梯
- **Capex 占大头**意味着真正的杠杆是**利用率**：同样的卡，MFU 从 0.3 提到 0.45 等效免费多出 50% 算力
- 电费看似小头，但换到脏电网或碳价内部化的地区就显著上浮——下一节把“碳”变成可见数字
"""))

t5.append(md("""
---
## E4 SustainabilityModel：同一个训练，三地碳排放差几倍

**预测区**：魁北克(水电) / 美国均值 / 波兰(煤电)，碳足迹最大差距 ≈ ____ 倍？
"""))

t5.append(code('''
# E4: 地理位置是一等公民的系统变量
grids = [mlsysim.Infrastructure.Grids.Quebec,
         mlsysim.Infrastructure.Grids.Norway,
         mlsysim.Infrastructure.Grids.US_Avg,
         mlsysim.Infrastructure.Grids.Poland]

rows, carbon = [], {}
for g in grids:
    r = sust.solve(fleet=fleet32, duration_days=30, datacenter=g)
    carbon[r.region_name] = r.carbon_footprint_kg
    rows.append([r.region_name,
                 f"{r.total_energy_kwh.to('MWh'):.0f} MWh",
                 f"{r.carbon_footprint_kg/1000:.1f} t CO2",
                 f"{r.water_usage_liters/1000:.0f} kL 水",
                 f"PUE {r.pue:.2f}"])

table(["区域", "能耗", "碳排放", "水耗", "PUE"], rows)

vals = list(carbon.values())
print()
print(f"最脏/最干净之比: {max(vals)/min(vals):.0f}x")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(list(carbon.keys()), [v/1000 for v in carbon.values()], color="#16a34a")
ax.set_ylabel("tonnes CO2eq / 30 days")
ax.set_title("Same job, different grid")
ax.grid(axis="y", alpha=.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
'''))

t5.append(md("""
**E4 解读**：能耗基本相同（差异只来自 PUE），**碳强度（gCO2/kWh）制造了几十倍的差距**。
选址是一个决策动作，效果胜过一切工程优化——这是大厂扎堆水电富集区建数据中心的原因。
"""))

t5.append(md("""
---
## E5 YAML 集群评估：一键输出 3-lens scorecard

把集群写成声明式 YAML，`mlsysim eval` 一次性给出 **Feasibility（可行性）/ Performance（性能）/ Macro（经济+碳）** 三个镜头。
"""))

t5.append(code('''
# E5: 写 cluster.yaml
yaml_text = """
version: "1.0"
name: "Llama-3 70B on 32x H100 (Quebec, 30d)"

workload:
  name: "Llama3_70B"
  batch_size: 256
  seq_len: 2048

hardware:
  name: "H100"
  accelerators: 32
  precision: "fp16"
  efficiency: 0.45

ops:
  region: "Quebec"
  duration_days: 30.0
"""
with open("cluster.yaml", "w") as f:
    f.write(yaml_text)

print(yaml_text)
'''))

t5.append(code('''
# 3-lens scorecard（文本版）
!mlsysim eval cluster.yaml
'''))

t5.append(code('''
# JSON 版（便于程序化处理）
!mlsysim eval cluster.yaml -o json | head -50
'''))

t5.append(md("""
**E5 解读 —— 为什么三个镜头要一起看**：

- Level 1 Feasibility：装不装得下（显存可行性）——不过关其他免谈
- Level 2 Performance：瓶颈在哪（Memory/Compute bound）、MFU 多少——技术好不好
- Level 3 Macro：TCO、电费、碳——**值不值、绿不绿**；YAML 里没写 ops 段时显示 SKIPPED（不是报错）
- 正确流程是把三层当成一张体检单：任何一层红灯，方案回炉
"""))

t5.append(md("""
---
## O1（选做）ReliabilityModel + CheckpointModel：故障与存档的经济学

集群越大越易碎：单卡 MTTF ~5 万小时，N 卡集群 MTBF ≈ MTTF/N。
Young-Daly 公式给出最优 checkpoint 间隔 `T_opt = sqrt(2 × δ × M)`（δ=写一次 checkpoint 的时间，M=集群 MTBF）。

**预测区**：32 卡集群的 MTBF ≈ ____ 小时？30 天预期故障 ____ 次？
"""))

t5.append(code('''
# O1-a: 32 GPU / 30 天任务的可靠性画像
rel = ReliabilityModel().solve(fleet=fleet32, job_duration_hours=30*24, checkpoint_time_s=60.0)
mtbf_h = rel.fleet_mtbf.to("hour").magnitude
opt_h = rel.optimal_checkpoint_interval.to("hour").magnitude

info("可靠性分析 (32 x H100, 30 天)",
     集群MTBF=f"{mtbf_h:.1f} 小时",
     预期故障次数=f"{rel.expected_failures:.1f} 次",
     最优checkpoint间隔=f"{opt_h:.2f} 小时",
     Goodput=f"{rel.goodput_ratio:.1%}")

yd_h = math.sqrt(2 * 60 * mtbf_h * 3600) / 3600     # Young-Daly 校验
print()
print(f"Young-Daly 手算: sqrt(2 * 60s * {mtbf_h:.1f}h) = {yd_h:.2f} 小时（应与上面一致）")
'''))

t5.append(code('''
# O1-b: checkpoint 尺寸与 I/O 冲击（Adam: 每参数 14 字节）
ckpt = CheckpointModel()
rows, ckpt_by_interval = [], {}
for interval_h in [0.5, 1, 2, 4, 8]:
    c = ckpt.solve(model=llama70b, hardware=DGX.accelerator, optimizer="adam",
                   checkpoint_interval_hours=interval_h)
    ckpt_by_interval[interval_h] = c
    rows.append([f"{interval_h} h",
                 f"{c.checkpoint_size.to('GB'):.0f} GB",
                 f"{c.write_time_seconds.to('second'):.0f} s",
                 f"{c.mfu_penalty_pct:.2%}",
                 "是" if c.storage_bottleneck else "否"])

table(["checkpoint 间隔", "单份大小", "写入耗时", "MFU 损失", "存储成瓶颈?"], rows)

write_s_at_opt = ckpt_by_interval[min(ckpt_by_interval, key=lambda k: abs(k - opt_h))].write_time_seconds.to("second").magnitude
print()
print(f"按最优间隔 ~{opt_h:.2f}h 存档: 30 天约 {30*24/opt_h:.0f} 次 x {write_s_at_opt:.0f}s 写入，"
      f"外加每次故障平均回滚半个间隔的工作量")
'''))

t5.append(md("""
**O1 解读**：checkpoint 不是免费的保险——写盘打断训练（MFU 损失列），太频繁浪费 I/O，太稀疏则故障回滚损失惨重。
Young-Daly 平衡两者；集群越大 MTBF 越短，最优间隔自动收紧。
"""))

t5.append(md("""
---
## O2（选做）多目标 DSE：天数 × TCO × 碳的 Pareto 前沿

思路：固定训练总量（70B × 2T tokens），对 {集群规模 × 并行策略} 的每个可行配置计算：
① 训练天数（Iron Law：`C = 6PD`，可达算力 = N × 峰值 × MFU × 扩展效率η）；
② 该时长下的 TCO；③ 碳排放（美国均值电网）。
然后手动求三维 Pareto 前沿。

> 这里不用 DSE 引擎而用手动循环，是因为我们要保留**每个候选的三个指标**做支配关系判断（DSE 只回 Top-5）。
"""))

t5.append(code('''
# O2: 构造候选并求 Pareto 前沿
PEAK = 989e12                                       # H100 FP16 dense FLOP/s
P = llama70b.parameters.to("count").magnitude
C_total = 6 * P * 2e12                              # Chinchilla: C = 6PD
MFU = 0.45

candidates = []
for n_nodes in [1, 2, 4, 8, 16]:
    fl = mk_fleet(n_nodes)
    n_gpu = 8 * n_nodes
    for tp, pp in [(tp, pp) for tp in (1, 2, 4, 8) for pp in (1, 2, 4, 8)
                   if tp * pp <= n_gpu and n_gpu % (tp * pp) == 0]:
        if train_mem_gb_per_gpu(llama70b, tp, pp) > 80:
            continue                                 # 内存可行性筛选
        try:
            r = dist.solve(model=llama70b, fleet=fl, batch_size=max(64, n_gpu),
                           tp_size=tp, pp_size=pp, precision="fp16",
                           efficiency=MFU, seq_len=2048, overlap_comm=True)
        except Exception:
            continue
        days = C_total / (n_gpu * PEAK * MFU * r.scaling_efficiency) / 86400
        t = econ.solve(fleet=fl, duration_days=days, mfu=MFU)
        s = sust.solve(fleet=fl, duration_days=days,
                       datacenter=mlsysim.Infrastructure.Grids.US_Avg)
        candidates.append({"nodes": n_nodes, "tp": tp, "pp": pp,
                           "eta": r.scaling_efficiency, "days": days,
                           "tco": t.tco_usd, "carbon": s.carbon_footprint_kg})

print(f"共 {len(candidates)} 个可行候选")


def dominates(a, b):   # a 支配 b：三目标都不差且至少一项严格更好
    no_worse = (a["days"] <= b["days"] and a["tco"] <= b["tco"] and a["carbon"] <= b["carbon"])
    strictly = (a["days"] < b["days"] or a["tco"] < b["tco"] or a["carbon"] < b["carbon"])
    return no_worse and strictly


frontier = [c for c in candidates
            if not any(dominates(o, c) for o in candidates if o is not c)]
frontier.sort(key=lambda c: c["days"])

rows = [[c["nodes"], c["tp"], c["pp"], f"{c['eta']:.0%}", f"{c['days']:.1f}",
         f"${c['tco']/1e6:.2f}M", f"{c['carbon']/1000:.0f} t"] for c in frontier]
table(["节点", "TP", "PP", "eta", "训练天数", "TCO", "碳(t)"], rows)
'''))

t5.append(code('''
# O2: 可视化 + 平衡点选择
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.4))
xs = [c["days"] for c in candidates]
ys = [c["tco"]/1e6 for c in candidates]
cs = [c["carbon"]/1000 for c in candidates]
sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=42)
fig.colorbar(sc, label="carbon (tonnes)")
fx = [c["days"] for c in frontier]
fy = [c["tco"]/1e6 for c in frontier]
ax.plot(fx, fy, "r.--", lw=1, label="Pareto frontier")
ax.set_xlabel("training days (lower = better)")
ax.set_ylabel("TCO ($M, lower = better)")
ax.set_title("color = carbon (tonnes, lower = better)")
ax.legend()
ax.grid(alpha=.3)
plt.tight_layout()
plt.show()


# “平衡方案”：三指标归一化名次之和最小（不属于任何单目标最优，却最难被拒绝）
def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    pos = [0] * len(vals)
    for rk, idx in enumerate(order):
        pos[idx] = rk
    return pos

rk_days = ranks([c["days"] for c in candidates])
rk_tco  = ranks([c["tco"] for c in candidates])
rk_carb = ranks([c["carbon"] for c in candidates])
balanced = min(candidates, key=lambda c: rk_days[candidates.index(c)] +
                                        rk_tco[candidates.index(c)] +
                                        rk_carb[candidates.index(c)])
print("平衡方案:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in balanced.items()})
'''))

t5.append(md("""
**O2 结论框架**：

- 前沿上的点各有性格：最快（最贵）、最低 TCO（往往是中等规模）、最低碳（清洁电网 + 适度时长）
- 平衡方案的辩护词：“它在三个维度上都排进前 __%，任何单一目标的方案都会在其他维度付出 __ 倍代价”——工程决策区别于刷榜的地方就在这里
"""))

t5.append(md("""
---
## O3（选做）GPT-4 级别（1.76T）在万卡集群上的时间-成本-碳预测

Iron Law：`Time = 6PD / (N × Peak × MFU × η)`。取 P=1.76T、D=1.5T tokens（issue 设定）、N=10000×H100。
η 取自 Production_2K（2048 卡）上 TP=8/PP=8 可行配置的扩展效率——**外推假设，报告里要注明**。

**预测区**：预计训练时长 ____ 天？TCO ____ 亿美元量级？
"""))

t5.append(code('''
# O3: 万卡推演
PEAK = 989e12
prod2k = mlsysim.Systems.Clusters.Production_2K      # 2048 GPUs
r_eta = dist.solve(model=gpt4, fleet=prod2k, batch_size=2048,
                   tp_size=8, pp_size=8, precision="fp16", efficiency=0.45,
                   seq_len=2048, overlap_comm=True)
eta = r_eta.scaling_efficiency

P_gpt4 = gpt4.parameters.to("count").magnitude       # 1.76e12
D_gpt4 = 1.5e12                                      # tokens (issue 设定)
C_gpt4 = 6 * P_gpt4 * D_gpt4

fleet10k = mlsysim.Systems.Clusters.Training_10K     # 10000 x H100
N = fleet10k.total_accelerators
days = C_gpt4 / (N * PEAK * 0.45 * eta) / 86400

t10k = econ.solve(fleet=fleet10k, duration_days=days, mfu=0.45)
s_qc = sust.solve(fleet=fleet10k, duration_days=days,
                  datacenter=mlsysim.Infrastructure.Grids.Quebec)
s_pl = sust.solve(fleet=fleet10k, duration_days=days,
                  datacenter=mlsysim.Infrastructure.Grids.Poland)

info("GPT-4 级别训练推演 (10000 x H100)",
     总计算量=f"{C_gpt4:.2e} FLOP",
     扩展效率=f"{eta:.1%} (测自 2K 卡 TP8/PP8, 外推假设)",
     预计时长=f"{days:.0f} 天",
     TCO=f"${t10k.tco_usd/1e6:,.0f}M (capex 占 {t10k.capex_usd/t10k.tco_usd:.0%})",
     能耗=f"{t10k.total_energy_kwh.to('GWh'):.1f} GWh",
     碳_Quebec=f"{s_qc.carbon_footprint_kg/1000:,.0f} t",
     碳_Poland=f"{s_pl.carbon_footprint_kg/1000:,.0f} t")
'''))

t5.append(md("""
**O3 讨论**：

- 对照公开估计（GPT-4 ≈ 2500 万 A100-day）：我们用 1 万张 H100、η=__%、MFU 45% 得到 ____ 天——量级是否合理？差异来自哪些假设（tokens 数、MoE 只按总参数计 6PD、η 外推）？
- 成本结构：Capex 占 ____%；如果集群利用率再降 10 个百分点，账单如何变化？
- 选址：同一训练 Quebec vs Poland 相差 ____ 倍碳——CTO 的选址理由应该写在方案第一页
"""))

t5.append(md("""
---
## 打卡对照清单（issue #137）

**最小打卡（E1–E3）**
- [ ] E1：单节点最优 DP/TP/PP 组合与吞吐 + 内存筛选的解释
- [ ] E2：1→4 节点通信占比从 __% 涨到 __%
- [ ] E3：30 天 TCO、Capex/Opex 比例、“成本非线性”一段话

**学有余力 1（+E4 E5 O1）**
- [ ] E4：三地碳排对比 + “选址单枪匹马改变碳足迹”
- [ ] E5：YAML 3-lens scorecard 输出 + 三镜头价值
- [ ] O1：MTBF、最优 checkpoint 间隔、MFU 冲击

**学有余力 2（+O2 O3）**
- [ ] O2：Pareto 前沿图 + 平衡方案辩护
- [ ] O3：GPT-4 级训练的时间-成本-碳完整预测（设计方案可用 O3 输出作骨架）

> 截图建议：scorecard 文本、碳排柱状图、Pareto 散点图是最有说服力的三张图。
"""))

save("task3_kv_cache_model_scale.ipynb", nb("task3_kv_cache_model_scale", t3))
save("task4_data_algo_system_opt.ipynb", nb("task4_data_algo_system_opt", t4))
save("task5_cluster_cost_integration.ipynb", nb("task5_cluster_cost_integration", t5))
print("done")
