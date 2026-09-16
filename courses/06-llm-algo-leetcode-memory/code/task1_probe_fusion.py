# -*- coding: utf-8 -*-
"""Task1 融合实验（补测）：把 attention 里"算子边界上的额外 S^2 往返"逐笔消掉。

变体（推理，B=1,H=32,D=128,bf16）：
  A naive      : matmul(q,kT)*scale -> softmax -> matmul(.,v)     (5 趟 S^2)
  B fold-scale : qs=q*scale -> matmul -> softmax -> matmul        (4 趟 S^2)
  C triton-sm  : matmul -> 自写 triton(scale+softmax) -> matmul    (3 趟 S^2)
  D sdpa       : F.scaled_dot_product_attention                   (0 趟)
  E triton-fa  : 自写 FlashAttention(tiling+online softmax)       (0 趟) + tile 调参
训练态另测 A/B/D：反向要消费 P，所以峰值由 S^2 决定而非由"少一次物化"决定。

运行环境：vm-60 · NVIDIA RTX 4090D · torch 2.9.1+cu128 · triton 3.5.1
"""
import math, time
import torch, torch.nn.functional as F
import triton
import triton.language as tl

DEV = torch.device("cuda")
MB = 2 ** 20
torch.manual_seed(0)
H, D = 32, 128
print("GPU:", torch.cuda.get_device_name(0), "| triton", triton.__version__, flush=True)


@triton.jit
def _scaled_softmax_kernel(out_ptr, in_ptr, n_rows, n_cols, scale, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row < n_rows:
        cols = tl.arange(0, BLOCK)
        m = cols < n_cols
        x = tl.load(in_ptr + row * stride_row + cols, mask=m, other=float("-inf"))
        x = x.to(tl.float32) * scale
        mx = tl.max(x, axis=0)
        e = tl.exp(x - mx)
        s = tl.sum(e, axis=0)
        tl.store(out_ptr + row * stride_row + cols, (e / s).to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _flash_fwd(Q, K, V, O, sm_scale,
               sqb, sqh, sqm, sqd, skb, skh, skn, skd,
               svb, svh, svn, svd, sob, soh, som, sod,
               H, N, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // H
    off_h = off_hb % H
    qo = off_b * sqb + off_h * sqh
    ko = off_b * skb + off_h * skh
    vo = off_b * svb + off_h * svh
    oo = off_b * sob + off_h * soh
    offs_m = start_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + qo + offs_m[:, None] * sqm + offs_d[None, :] * sqd)
    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    for start_n in range(0, N, BN):
        k = tl.load(K + ko + (start_n + offs_n)[:, None] * skn + offs_d[None, :] * skd)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + vo + (start_n + offs_n)[:, None] * svn + offs_d[None, :] * svd)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + oo + offs_m[:, None] * som + offs_d[None, :] * sod, acc.to(O.dtype.element_ty))


def timed(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def peak_of(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(); torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) / MB


def run_sm_fused(s, scale, block=4096):
    out = torch.empty_like(s)
    rows = s.shape[0] * s.shape[1] * s.shape[2]
    _scaled_softmax_kernel[(rows,)](out, s, rows, s.shape[-1], scale, s.stride(-2), block)
    return out


def run_flash(q, k, v, scale, BM=128, BN=64, W=8, S=3):
    Bx, Hx, N, Dx = q.shape
    o = torch.empty_like(q)
    _flash_fwd[(N // BM, Bx * Hx)](q, k, v, o, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Hx, N, Dx, BM, BN, num_warps=W, num_stages=S)
    return o


# ============ 推理态：五变体阶梯 ============
print("\n[1] 推理态阶梯 (S^2 单趟 = 1024 MiB @N=4096)")
print("   %6s %8s %8s %8s %8s %8s | %9s %9s" % ("seq", "A ms", "B ms", "C ms", "D ms", "E ms", "D峰值MiB", "E峰值MiB"))
res = {}
for N in (1024, 2048, 4096):
    q = torch.randn(1, H, N, D, dtype=torch.bfloat16, device=DEV)
    k = torch.randn_like(q); v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(D)
    var = {
        "A": lambda: torch.matmul(F.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1), v),
        "B": lambda: torch.matmul(F.softmax(torch.matmul(q * scale, k.transpose(-1, -2)), dim=-1), v),
        "C": lambda: torch.matmul(run_sm_fused(torch.matmul(q, k.transpose(-1, -2)), scale), v),
        "D": lambda: F.scaled_dot_product_attention(q, k, v),
        "E": lambda: run_flash(q, k, v, scale),
    }
    ref = torch.matmul(F.softmax(torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale, dim=-1), v.float())
    ts, errs = {}, {}
    for name, fn in var.items():
        outs = fn(); torch.cuda.synchronize()
        errs[name] = (outs.float() - ref).abs().max().item()
        ts[name] = timed(fn) * 1e3
        del outs
    _, pk_d = peak_of(var["D"])
    _, pk_e = peak_of(var["E"])
    print("   %6d %8.3f %8.3f %8.3f %8.3f %8.3f | %9.1f %9.1f" % (N, ts["A"], ts["B"], ts["C"], ts["D"], ts["E"], pk_d, pk_e), flush=True)
    print("          误差 vs fp32: A %.2e B %.2e C %.2e D %.2e E %.2e"
          % (errs["A"], errs["B"], errs["C"], errs["D"], errs["E"]), flush=True)
    res[N] = ts
    del q, k, v, ref
    torch.cuda.empty_cache()

N = 4096
t = res[N]
s2mb = H * N * N * 2 / MB
print("\n   逐笔回收 (N=4096, S^2 单趟 %.0f MiB):" % s2mb)
print("     A->B 折掉 *scale 的额外物化      : %+.3f ms  (2 趟 = %.0f MB -> %.0f GB/s)"
      % (t["B"] - t["A"], 2 * s2mb, 2 * s2mb * MB / ((t["A"] - t["B"]) * 1e-3) / 1e9))
print("     B->C triton 融合 scale+softmax   : %+.3f ms" % (t["C"] - t["B"]))
print("     C->D 全融合(不再物化 S^2)        : %+.3f ms  (3 趟 -> %.0f GB/s)"
      % (t["D"] - t["C"], 3 * s2mb * MB / ((t["C"] - t["D"]) * 1e-3) / 1e9))
print("     A->D 端到端                      : %+.3f ms (%.1fx)" % (t["D"] - t["A"], t["A"] / t["D"]))

# ============ tile 调参 ============
print("\n[2] 自写 FlashAttention 的 tile/warp/stage 扫描 (N=4096)")
q = torch.randn(1, H, N, D, dtype=torch.bfloat16, device=DEV)
k = torch.randn_like(q); v = torch.randn_like(q)
scale = 1.0 / math.sqrt(D)
ref = F.scaled_dot_product_attention(q, k, v)
t_sdpa = timed(lambda: F.scaled_dot_product_attention(q, k, v)) * 1e3
best = None
for BM, BN, W, S in [(64, 64, 4, 2), (64, 64, 4, 3), (64, 32, 4, 3), (128, 64, 4, 3),
                     (128, 64, 8, 3), (128, 128, 8, 2), (64, 128, 4, 3), (128, 32, 4, 3)]:
    try:
        fn = lambda: run_flash(q, k, v, scale, BM, BN, W, S)
        out = fn(); torch.cuda.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        tt = timed(fn) * 1e3
        tf = 4 * H * N * N * D / (tt * 1e-3) / 1e12
        print("     BM=%-4d BN=%-4d warps=%d stages=%d  %8.3f ms %7.1f TFLOPS  %.2fx SDPA  err %.1e"
              % (BM, BN, W, S, tt, tf, tt / t_sdpa, err), flush=True)
        if best is None or tt < best[0]:
            best = (tt, BM, BN, W, S, tf, err)
        del out
    except Exception as exc:
        print("     BM=%d BN=%d warps=%d stages=%d FAILED: %s" % (BM, BN, W, S, str(exc)[:60]), flush=True)
print("     最佳 %.3f ms (BM=%d BN=%d warps=%d stages=%d, %.1f TFLOPS) vs SDPA %.3f ms"
      % (best[0], best[1], best[2], best[3], best[4], best[5], t_sdpa))

# ============ 训练态：反向要消费 P ============
print("\n[3] 训练态 (fwd+bwd) 对照：峰值由 P 与反向临时量决定，不由'少一次物化'决定")
print("   %6s %-4s %10s %12s %10s" % ("seq", "var", "step ms", "peak MiB", "peak/S^2"))
for Nn in (1024, 2048, 4096):
    sc = 1.0 / math.sqrt(D)
    s2 = H * Nn * Nn * 2 / MB
    qq = torch.randn(1, H, Nn, D, dtype=torch.bfloat16, device=DEV, requires_grad=True)
    kk = torch.randn(1, H, Nn, D, dtype=torch.bfloat16, device=DEV, requires_grad=True)
    vv = torch.randn(1, H, Nn, D, dtype=torch.bfloat16, device=DEV, requires_grad=True)
    go = torch.randn(1, H, Nn, D, dtype=torch.bfloat16, device=DEV)
    out = {}
    for name in ("A", "B", "D"):
        def step():
            if qq.grad is not None:
                qq.grad = None; kk.grad = None; vv.grad = None
            if name == "A":
                o = torch.matmul(F.softmax(torch.matmul(qq, kk.transpose(-1, -2)) * sc, dim=-1), vv)
            elif name == "B":
                o = torch.matmul(F.softmax(torch.matmul(qq * sc, kk.transpose(-1, -2)), dim=-1), vv)
            else:
                o = F.scaled_dot_product_attention(qq, kk, vv)
            (o.float() * go.float()).sum().backward()
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        step(); torch.cuda.synchronize()
        pk = (torch.cuda.max_memory_allocated() - base) / MB
        tt = timed(step, iters=10, warm=3) * 1e3
        out[name] = (tt, pk)
        print("   %6d %-4s %10.3f %12.1f %9.2fx" % (Nn, name, tt, pk, pk / s2), flush=True)
    print("          A-B 时间差 %+.3f ms | 峰值差 %+.1f MiB (= q*scale 缓冲 %.1f MiB) | D 峰值是 B 的 %.2f"
          % (out["B"][0] - out["A"][0], out["B"][1] - out["A"][1],
             (H * Nn * D * 2) / MB, out["D"][1] / out["B"][1]), flush=True)
    del qq, kk, vv, go
    torch.cuda.empty_cache()

print("\nPROBE_FUSION_DONE", flush=True)
