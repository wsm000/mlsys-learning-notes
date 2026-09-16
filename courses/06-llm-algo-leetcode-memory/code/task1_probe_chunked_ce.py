# -*- coding: utf-8 -*-
"""定位 ③(FA-backward) 剩余峰值：把 CE 按序列分块，参数完全不变，只看 O(N*V) 项的影响。
配置：B=1, S=4096, d=512, H=8, ffn=1376, L=4, V=32000, fp32 参数 + bf16 autocast + AdamW。
"""
import time
import torch, torch.nn as nn, torch.nn.functional as F

DEV = torch.device("cuda")
MB = 2 ** 20
torch.manual_seed(0)
B, S, D, H, FFN, V, L = 1, 4096, 512, 8, 1376, 32000, 4
DH = D // H
print("GPU:", torch.cuda.get_device_name(0), "| B=%d S=%d V=%d L=%d" % (B, S, V, L), flush=True)
print("单项 O(N*V) 张量 = S*V*2B = %.0f MiB" % (S * V * 2 / MB), flush=True)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False); self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False); self.o = nn.Linear(D, D, bias=False)
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
        h = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def trunk(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lnf(x)


def measure(chunk, label, iters=5):
    torch.manual_seed(0)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    idx = torch.randint(0, V, (B, S), device=DEV)
    tgt = torch.randint(0, V, (B, S), device=DEV)
    base = torch.cuda.memory_allocated()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = net.trunk(idx)
            if chunk is None:
                loss = F.cross_entropy(net.head(h).view(-1, V), tgt.view(-1))
            else:
                tot = 0.0
                n = 0
                for i in range(0, S, chunk):
                    lg = net.head(h[:, i:i + chunk])
                    tot = tot + F.cross_entropy(lg.reshape(-1, V), tgt[:, i:i + chunk].reshape(-1), reduction="sum")
                    n += lg.shape[0] * lg.shape[1]
                loss = tot / n
        loss.backward()
        opt.step()

    step(); step()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step(); torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / MB
    step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / iters * 1e3
    print("   %-34s peak %8.1f MiB   step %7.1f ms" % (label, peak, t), flush=True)
    del net, opt, idx, tgt
    torch.cuda.empty_cache()
    return peak, t


print("\n参数完全相同，只改 CE 的物化方式：")
p_full, t_full = measure(None, "full CE: logits [B,S,V] 一次物化")
p_1k, t_1k = measure(1024, "chunked CE: 每块 1024 token")
p_512, t_512 = measure(512, "chunked CE: 每块 512 token")
print("\n结论数据：")
print("   full -> chunk1024 : 峰值 %+.1f MiB, 时间 %+.1f ms" % (p_1k - p_full, t_1k - t_full))
print("   full -> chunk512  : 峰值 %+.1f MiB, 时间 %+.1f ms" % (p_512 - p_full, t_512 - t_full))
print("   O(N*V) 三项(logits/CE中间/dlogits) 理论 %.0f MiB" % (3 * S * V * 2 / MB))
print("\nPROBE_CHUNK_DONE", flush=True)
