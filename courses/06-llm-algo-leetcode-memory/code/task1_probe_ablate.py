# -*- coding: utf-8 -*-
"""消融：FA-backward 情况下峰值 1776 MiB 到底被谁占住。
配置固定 B=1, S=4096, d=512, H=8, ffn=1376, L=4, fp32 参数 + bf16 autocast + AdamW。
逐个关掉可疑项，看峰值掉多少。
"""
import math, time
import torch, torch.nn as nn, torch.nn.functional as F

DEV = torch.device("cuda")
MB = 2 ** 20
torch.manual_seed(0)
B, S, D, H, FFN = 1, 4096, 512, 8, 1376
DH = D // H
print("GPU:", torch.cuda.get_device_name(0), flush=True)


class Block(nn.Module):
    def __init__(self, use_mlp=True):
        super().__init__()
        self.use_mlp = use_mlp
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False); self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False); self.o = nn.Linear(D, D, bias=False)
        if use_mlp:
            self.w1 = nn.Linear(D, FFN, bias=False); self.w3 = nn.Linear(D, FFN, bias=False)
            self.w2 = nn.Linear(FFN, D, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, s, H, DH).transpose(1, 2)
        k = self.k(h).view(b, s, H, DH).transpose(1, 2)
        v = self.v(h).view(b, s, H, DH).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(o.transpose(1, 2).reshape(b, s, -1))
        if self.use_mlp:
            h = self.ln2(x)
            x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class Net(nn.Module):
    def __init__(self, vocab, layers, use_mlp=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.blocks = nn.ModuleList([Block(use_mlp) for _ in range(layers)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


def run(vocab=32000, layers=4, use_mlp=True, manual_ce=False, label=""):
    torch.manual_seed(0)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net = Net(vocab, layers, use_mlp).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    idx = torch.randint(0, vocab, (B, S), device=DEV)
    tgt = torch.randint(0, vocab, (B, S), device=DEV)
    P = sum(p.numel() for p in net.parameters())
    base = torch.cuda.memory_allocated()

    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(idx)
            if manual_ce:
                loss = F.nll_loss(F.log_softmax(logits.view(-1, vocab), dim=-1), tgt.view(-1))
            else:
                loss = F.cross_entropy(logits.view(-1, vocab), tgt.view(-1))
        fwd_peak = torch.cuda.max_memory_allocated()
        loss.backward()
        return fwd_peak

    fp = fwd_bwd()
    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated()
    steady_after_bwd = torch.cuda.memory_allocated()
    opt.step()
    steady = torch.cuda.memory_allocated()
    pk = (bwd_peak - base) / MB
    print("   %-38s P=%6.2fM  参数+梯度 %.0f MiB | Adam %.0f MiB | forward峰值 %.0f | 整步峰值 %.0f"
          % (label, P / 1e6, (steady_after_bwd - base - 0) / MB, (steady - steady_after_bwd) / MB,
             (fp - base) / MB, pk), flush=True)
    del net, opt, idx, tgt
    torch.cuda.empty_cache()
    return pk


print("\n消融结果 (FA-backward, 不重算):")
full = run(label="full: V=32000, L=4, 含 MLP, fused CE")
v1k = run(vocab=1000, label="V 32000 -> 1000 (只动 logits/CE)")
l1 = run(layers=1, label="L 4 -> 1 (看单层斜率)")
no_mlp = run(use_mlp=False, label="去掉 MLP (只留 attention)")
man = run(manual_ce=True, label="manual CE(log_softmax 材料化)")

print("\n差额归因:")
print("   logits + CE 相关        : %8.1f MiB  (V=32000 与 V=1000 之差)" % (full - v1k))
print("   每层贡献 (attention+MLP): %8.1f MiB/层 (L=4 与 L=1 之差 / 3)" % ((full - l1) / 3))
print("   MLP 部分                : %8.1f MiB  (含 MLP 与不含之差)" % (full - no_mlp))
print("   manual CE 额外材料化    : %8.1f MiB" % (man - full))
print("\nPROBE_ABLATE_DONE", flush=True)
