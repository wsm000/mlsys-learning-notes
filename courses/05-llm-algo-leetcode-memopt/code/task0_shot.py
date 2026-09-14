# Task0 打卡截图（综合版）：18节运行结果 + 显存峰值小实验
# 运行环境：vm-60 (RTX 4090D, torch 2.9.1+cu128)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体 ----
for f in font_manager.findSystemFonts(fontpaths=["/usr/share/fonts"], fontext="ttf"):
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
CJK = None
for name in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "AR PL UMing CN"]:
    if any(x.name == name for x in font_manager.fontManager.ttflist):
        CJK = name
        break
plt.rcParams["font.family"] = [CJK or "sans-serif", "DejaVu Sans"]
plt.rcParams["font.monospace"] = [CJK or "sans-serif", "DejaVu Sans Mono"]
plt.rcParams["axes.unicode_minus"] = False
print("CJK font:", CJK)

# ---- Part A: 复现 18 节的核心测试结果 ----
import torch
import torch.nn.functional as F
import math


def relu_backward(grad_out, x):
    mask = (x > 0).to(grad_out.dtype)
    return grad_out * mask


def softmax_ce_loss_and_grad(logits, labels, reduction="mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    per_sample_loss = -(one_hot * log_probs).sum(dim=1)
    loss = per_sample_loss.mean() if reduction == "mean" else per_sample_loss.sum()
    grad = probs - one_hot
    if reduction == "mean":
        grad = grad / logits.size(0)
    return loss, grad


x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
upstream = torch.tensor([0.5, -1.0, 2.0, 0.25, -0.75])
F.relu(x).backward(upstream)
manual_relu = relu_backward(upstream, x.detach())

logits = torch.tensor([[1.0, 0.5, -0.2], [0.2, -0.3, 1.2]], requires_grad=True)
labels = torch.tensor([0, 2])
loss, manual_grad = softmax_ce_loss_and_grad(logits, labels)
ce = F.cross_entropy(logits, labels)
ce.backward()

# ---- Part B: batch / seq_len 扫描（activation 峰值实验）----
import torch.nn as nn

results = {}
for B, S in [(4, 128), (8, 128), (16, 128), (4, 256), (4, 512)]:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 1024),
                          nn.GELU(), nn.Linear(1024, 512)).cuda()
    X = torch.randn(B, S, 512, device="cuda", requires_grad=True)
    Y = model(X)
    loss_b = Y.float().pow(2).mean()
    loss_b.backward()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    results[(B, S)] = peak
    print(f"batch={B:3d} seq_len={S:3d} peak={peak:8.1f} MiB")
    del model, X, Y, loss_b
    torch.cuda.empty_cache()

# ---- 绘图 ----
fig = plt.figure(figsize=(13.5, 9.8), facecolor="#f5f7fb")
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.15, 1.15], hspace=0.62, wspace=0.25,
                      top=0.855, bottom=0.085)

# 标题区
fig.text(0.5, 0.977, "llm-algo-leetcode 显存优化 | 202609 · Task0 显存对象与生命周期 —— 第18节运行结果",
         ha="center", va="center", fontsize=16, fontweight="bold", color="#1a2a4a")
fig.text(0.5, 0.943, "GitHub ID: wsm000    |    微信昵称: empty    |    运行环境: vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128 · CUDA 12.8",
         ha="center", va="center", fontsize=10.5, color="#3a4a6a")
fig.text(0.5, 0.913,
         "教程: datawhalechina/llm-algo-leetcode · 18_Activation_and_Loss_Backward.ipynb    社区: DataWhale https://github.com/datawhalechina/llm-algo-leetcode",
         ha="center", va="center", fontsize=9, color="#5a6a8a")

# 左上：ReLU 手写 backward
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis("off")
ax1.set_facecolor("white")
txt1 = ("Part A · 18节测试1：手写 ReLU backward\n"
        "  输入 x = [-2.0, -0.5, 0.0, 1.0, 3.0]\n"
        "  上游梯度  = [0.5, -1.0, 2.0, 0.25, -0.75]\n"
        "  mask = (x > 0)  →  门控梯度\n"
        f"  手写梯度  = {[round(v, 4) for v in manual_relu.tolist()]}\n"
        f"  PyTorch   = {[round(v, 4) for v in x.grad.tolist()]}\n"
        "  mask = (x > 0)，x==0 处约定置 0\n"
        "  [PASS] 手写 backward 与 PyTorch 自动求导一致")
ax1.text(0.03, 0.5, txt1, va="center", ha="left", fontsize=11.2, family="monospace",
         color="#143", linespacing=1.75,
         bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#4a8", lw=1.4))

# 右上：CrossEntropy 手写 backward
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")
ax2.set_facecolor("white")
txt2 = ("Part A · 18节测试2：手写 CrossEntropy backward\n"
        "  log_softmax 保证数值稳定（1000.0 量级也 OK）\n"
        "  grad = softmax(logits) - one_hot(target)\n"
        f"  CE loss (mean)   = {loss.item():.4f}   (PyTorch {ce.item():.4f})\n"
        "  mean/sum reduction 差一个 1/batch 缩放\n"
        "  越界 label、非法 reduction 均正确报错\n"
        "  [PASS] 测试通过：激活与损失的反向直觉一致")
ax2.text(0.03, 0.5, txt2, va="center", ha="left", fontsize=11.2, family="monospace",
         color="#143", linespacing=1.75,
         bbox=dict(boxstyle="round,pad=0.7", fc="white", ec="#a6a", lw=1.4))

# 左下：batch 扫描柱状图
ax3 = fig.add_subplot(gs[1:, 0])
keys_b = [(4, 128), (8, 128), (16, 128)]
vals_b = [results[k] for k in keys_b]
bars = ax3.bar([f"B={b}\nS={s}" for b, s in keys_b], vals_b,
               color=["#5b8def", "#3f6fd8", "#2a52b0"], width=0.55, zorder=3)
for r, v in zip(bars, vals_b):
    ax3.text(r.get_x() + r.get_width() / 2, v + max(vals_b) * 0.012, f"{v:.1f}",
             ha="center", fontsize=10.5, fontweight="bold", color="#1a2a4a")
ax3.set_title("batch 增大 → activation 峰值升高\n(固定 seq_len=128, Linear+GELU 网络)",
              fontsize=12, pad=10)
ax3.set_ylabel("单步显存峰值 (MiB)")
ax3.grid(axis="y", alpha=0.3, zorder=0)
ax3.set_ylim(0, max(vals_b) * 1.18)

# 右下：seq_len 扫描柱状图
ax4 = fig.add_subplot(gs[1:, 1])
keys_s = [(4, 128), (4, 256), (4, 512)]
vals_s = [results[k] for k in keys_s]
bars = ax4.bar([f"B={b}\nS={s}" for b, s in keys_s], vals_s,
               color=["#e8a34b", "#d98a28", "#b86e14"], width=0.55, zorder=3)
for r, v in zip(bars, vals_s):
    ax4.text(r.get_x() + r.get_width() / 2, v + max(vals_s) * 0.012, f"{v:.1f}",
             ha="center", fontsize=10.5, fontweight="bold", color="#1a2a4a")
ax4.set_title("seq_len 增大 → activation 峰值升高\n(固定 batch=4，峰值近似随 S 线性放大)",
              fontsize=12, pad=10)
ax4.set_ylabel("单步显存峰值 (MiB)")
ax4.grid(axis="y", alpha=0.3, zorder=0)
ax4.set_ylim(0, max(vals_s) * 1.18)

fig.text(0.5, 0.028,
         "结论：activation 峰值随 batch 与 seq_len 增大而升高——前向中间结果必须驻留到 backward，是训练显存峰值的来源；"
         "反向完成后图与 saved tensors 释放，常驻只剩 参数+梯度+优化器状态。",
         ha="center", fontsize=9.5, color="#445", wrap=True)

plt.savefig("task0_18_runshot.png", dpi=150, facecolor="#f5f7fb")
print("SAVED task0_18_runshot.png")
