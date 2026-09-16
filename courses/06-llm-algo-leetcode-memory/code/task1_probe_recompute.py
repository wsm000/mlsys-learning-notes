# -*- coding: utf-8 -*-
"""训练态四种组合对照：naive/FA x 不重算/整块 checkpointing。
判据问题：要把 4x S^2 拿下来，应该重算"整个 block"还是"只重算 P"？
配置：B=1, S=4096, d=512, H=8, dh=64, ffn=1376, L=4 层, fp32 参数 + bf16 autocast + AdamW
运行环境：vm-60 · NVIDIA RTX 4090D · torch 2.9.1+cu128
"""
import math, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEV = torch.device("cuda")
MB = 2 ** 20
torch.manual_seed(0)
B, S, D, H, FFN, L = 1, 4096, 512, 8, 1376, 4
DH = D // H
SCALE = 1.0 / math.sqrt(DH)
print("GPU:", torch.cuda.get_device_name(0), "| B=%d S=%d d=%d H=%d L=%d" % (B, S, D, H, L), flush=True)


class Block(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False); self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False); self.o = nn.Linear(D, D, bias=False)
        self.w1 = nn.Linear(D, FFN, bias=False); self.w3 = nn.Linear(D, FFN, bias=False)
        self.w2 = nn.Linear(FFN, D, bias=False)

    def _attn(self, h):
        b, s, _ = h.shape
        q = self.q(h).view(b, s, H, DH).transpose(1, 2)
        k = self.k(h).view(b, s, H, DH).transpose(1, 2)
        v = self.v(h).view(b, s, H, DH).transpose(1, 2)
        if self.mode == "naive":
            sc = torch.matmul(q, k.transpose(-1, -2)) * SCALE     # 额外一份 S^2 物化
            p = F.softmax(sc, dim=-1)                              # P 必须留给反向
            o = torch.matmul(p, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v)            # backward 重算 P
        return self.o(o.transpose(1, 2).reshape(b, s, -1))

    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        h = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Net(nn.Module):
    def __init__(self, mode, use_ckpt):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.emb = nn.Embedding(32000, D)
        self.blocks = nn.ModuleList([Block(mode) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, 32000, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            if self.use_ckpt:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.head(self.lnf(x))


idx = torch.randint(0, 32000, (B, S), device=DEV)
target = torch.randint(0, 32000, (B, S), device=DEV)
S2 = H * S * S * 2 / MB
print("单层 S^2 一份 = %.0f MiB | 全层朴素激活量级 = %.1f GiB" % (S2, S2 * 4 * L / 1024), flush=True)

res = {}
print("\n%-26s %10s %12s %10s %10s" % ("组合", "step ms", "peak MiB", "peak/基", "时间/基"))
print("-" * 74)
for mode, ckpt, label in [("naive", False, "1 naive + 不重算"),
                          ("naive", True, "2 naive + 整块 checkpointing"),
                          ("fa", False, "3 FA-backward + 不重算"),
                          ("fa", True, "4 FA-backward + 整块 checkpointing")]:
    torch.manual_seed(0)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net = Net(mode, ckpt).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(idx)
            loss = F.cross_entropy(logits.view(-1, 32000), target.view(-1))
        loss.backward()
        opt.step()
    for _ in range(2):
        step()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step(); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / MB
    for _ in range(2):
        step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 5 * 1e3
    res[label] = (t, peak)
    print("%-26s %10.1f %12.1f %10s %10s" % (label, t, peak, "-", "-"), flush=True)
    del net, opt
    torch.cuda.empty_cache()

b_t, b_p = res["1 naive + 不重算"]
print("-" * 74)
for k, (t, p) in res.items():
    print("%-26s %10.1f %12.1f %9.2fx %9.2fx" % (k, t, p, p / b_p, t / b_t), flush=True)
fa_t, fa_p = res["3 FA-backward + 不重算"]
ck_t, ck_p = res["2 naive + 整块 checkpointing"]
bo_t, bo_p = res["4 FA-backward + 整块 checkpointing"]
print("\n判据数据：")
print("  FA 单独          : 峰值 %.2fx, 时间 %.2fx" % (fa_p / b_p, fa_t / b_t))
print("  checkpoint 单独  : 峰值 %.2fx, 时间 %.2fx" % (ck_p / b_p, ck_t / b_t))
print("  两者都用         : 峰值 %.2fx, 时间 %.2fx" % (bo_p / b_p, bo_t / b_t))
print("\nPROBE_CKPT_DONE", flush=True)
