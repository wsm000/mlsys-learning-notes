# -*- coding: utf-8 -*-
"""Task1 颗粒度探针（补测）：逐张量字节账 + 算子之间的搬运时间 + 不可预测项。

运行环境：vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128 · triton 3.5.1
对应笔记：courses/06-llm-algo-leetcode-memory/notes/task01-hardware-and-vram-ledger.md
"""
import time, math
import torch, torch.nn.functional as F

DEV = torch.device("cuda")
MB = 2 ** 20
torch.manual_seed(0)
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)


def timed(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def real_bytes(shape, dtype):
    torch.cuda.empty_cache()
    b = torch.cuda.memory_allocated()
    t = torch.empty(shape, dtype=dtype, device=DEV)
    u = torch.cuda.memory_allocated() - b
    del t; torch.cuda.empty_cache()
    return u


def sz(shape, dt_b=2):
    n = 1
    for x in shape:
        n *= x
    return n * dt_b / MB


# ============ A. 逐项参数账：TinyGPT (B=8,S=256,V=32000,d=512,L=6,H=8,KVH=8,DH=64,ffn=1376) ============
B, S, V, D, L, H, KVH, DH, FFN = 8, 256, 32000, 512, 6, 8, 8, 64, 1376
print("\n[A] TinyGPT 逐项参数账 (fp32)")
per_layer = {
    "q_proj W": D * H * DH, "k_proj W": D * KVH * DH, "v_proj W": D * KVH * DH,
    "o_proj W": H * DH * D, "MLP gate/up/down W": D * FFN * 2 + FFN * D, "norm W": 2 * D,
}
tot = 0
for k, v in per_layer.items():
    print("   %-20s %10d params x%d层 = %8.2f MiB" % (k, v, L, v * L * 4 / MB))
    tot += v * L
print("   %-20s %10d params      = %8.2f MiB" % ("embedding+pos", V * D + 512 * D, (V * D + 512 * D) * 4 / MB))
print("   %-20s %10d params      = %8.2f MiB" % ("lm_head", D * V, D * V * 4 / MB))
tot += V * D + 512 * D + D * V
print("   合计 %d params -> 参数 %.2f / 梯度 %.2f / AdamW(m+v) %.2f MiB"
      % (tot, tot * 4 / MB, tot * 4 / MB, tot * 8 / MB))

# ============ B. 同一 step 的中间张量（真实 shape 实分配验证） ============
print("\n[B] 中间张量字节账 (bf16 实分配验证)")
for name, shape in [("logits [B,S,V]", (B, S, V)), ("CE/softmax 中间 [B,S,V]", (B, S, V)),
                    ("scores [B,H,S,S]", (B, H, S, S)), ("Q/K/V 各 [B,H,S,DH]", (B, H, S, DH)),
                    ("MLP gate/up 各 [B,S,FFN]", (B, S, FFN)), ("LN 输出 [B,S,D]", (B, S, D))]:
    print("   %-24s %-16s 理论 %8.3f MiB | 实测 %8.3f MiB | fp32 %8.3f"
          % (name, str(shape), sz(shape), real_bytes(shape, torch.bfloat16) / MB, sz(shape, 4)))

# ============ C. 放大到 LLaMA-3-8B 级（只算不分配） ============
B2, S2, V2, D2, L2, H2, KVH2, DH2, FFN2 = 1, 8192, 128256, 4096, 32, 32, 8, 128, 14336
print("\n[C] LLaMA-3-8B 级 (B=1,S=8192,V=128256,d=4096,L=32,H=32,KVH=8,DH=128) 理论字节")
print("   logits [B,S,V]            = %9.1f MiB  (CE 中间再要一份，共 2x)" % sz((B2, S2, V2)))
print("   scores [B,H,S,S]          = %9.1f MiB" % sz((B2, H2, S2, S2)))
print("   单层 Q/K/V 激活           = %9.1f MiB" % (3 * sz((B2, H2, S2, DH2))))
print("   单层 MLP gate/up/silu     = %9.1f MiB" % (3 * sz((B2, S2, FFN2))))
print("   单层 LN+residual          = %9.1f MiB" % (3 * sz((B2, S2, D2))))
print("   全层 KV cache GQA(8头)    = %9.1f MiB" % (2 * L2 * sz((B2, KVH2, S2, DH2))))
print("   全层 KV cache MHA(32头)   = %9.1f MiB" % (2 * L2 * sz((B2, H2, S2, DH2))))

# ============ D. 算子之间的搬运：naive attention 三段拆解 ============
print("\n[D] 标准 attention 三段拆解（B=1,H=32,DH=128,bf16）")
print("   %6s %9s %9s %9s %9s %9s | %s" % ("seq", "QK^T ms", "softmax", "PV ms", "三段和", "融合SDPA", "各段实到带宽"))
for N in (1024, 2048, 4096):
    q = torch.randn(1, H, N, DH, dtype=torch.bfloat16, device=DEV)
    k = torch.randn_like(q); v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(DH)
    def op_qk(): return torch.matmul(q, k.transpose(-1, -2))
    sc = op_qk()
    def op_sm(): return F.softmax(sc, dim=-1)
    pr = op_sm()
    def op_pv(): return torch.matmul(pr, v)
    def op_fu(): return F.scaled_dot_product_attention(q, k, v)
    t = [timed(f) * 1e3 for f in (op_qk, op_sm, op_pv, op_fu)]
    s2 = H * N * N * 2
    b_qk = 2 * H * N * DH * 2 + s2
    b_sm = 2 * s2
    b_pv = s2 + 2 * H * N * DH * 2
    print("   %6d %9.3f %9.3f %9.3f %9.3f %9.3f | QK %6.1fMB %4.0fGB/s · SM %6.1fMB %4.0fGB/s · PV %6.1fMB %4.0fGB/s"
          % (N, t[0], t[1], t[2], t[0] + t[1] + t[2], t[3],
             b_qk / MB, b_qk / (t[0] * 1e-3) / 1e9, b_sm / MB, b_sm / (t[1] * 1e-3) / 1e9,
             b_pv / MB, b_pv / (t[2] * 1e-3) / 1e9), flush=True)
    del q, k, v, sc, pr
    torch.cuda.empty_cache()

# ============ E. 带宽口径（独立张量，避免 src==dst 被跳过） ============
print("\n[E] 内存带宽口径 (64MiB 张量)")
n = 64 * 2 ** 20
a = torch.empty(n, dtype=torch.uint8, device=DEV).fill_(1)
b = torch.empty(n, dtype=torch.uint8, device=DEV).fill_(2)
c = torch.empty_like(a)
print("   copy  (读64+写64) : %7.1f GB/s" % (2 * n / timed(lambda: c.copy_(a)) / 1e9))
print("   add   (读128+写64): %7.1f GB/s" % (3 * n / timed(lambda: torch.add(a, b, out=c)) / 1e9))
print("   fill  (只写64)    : %7.1f GB/s" % (n / timed(lambda: c.fill_(3)) / 1e9))
del a, b, c
torch.cuda.empty_cache()

# ============ F. 不可预测项 ============
print("\n[F] 账本预测不到的部分")
x = torch.zeros(256, device=DEV)
print("   kernel launch 开销 ≈ %.2f us/次 (256 元素 add_)" % (timed(lambda: x.add_(1.0), iters=2000) * 1e6))
big = torch.empty(64 * 2 ** 20, dtype=torch.uint8, device=DEV)
del big
print("   分配 64MiB 后释放: reserved=%.2f MiB / allocated=%.2f MiB (释放 ≠ 归还驱动)"
      % (torch.cuda.memory_reserved() / MB, torch.cuda.memory_allocated() / MB))
tmp = torch.empty(8, 256, 32000, dtype=torch.bfloat16, device=DEV)
print("   logits 理论 %.1f MiB / 实测 %.1f MiB (分配粒度取整)"
      % (sz((8, 256, 32000)), real_bytes((8, 256, 32000), torch.bfloat16) / MB))
del tmp, x
print("\nPROBE_GRANULAR_DONE", flush=True)
