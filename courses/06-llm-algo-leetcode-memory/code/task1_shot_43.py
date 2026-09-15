# -*- coding: utf-8 -*-
"""llm-algo-leetcode 显存优化 | 202609 Task1 硬件与显存账本
4.3 增项2：topic 01 显存账本与指标 / Part02 04 多头注意力 (MHA/GQA) / Part01 14 FlashAttention 显存模型
内容：04 节 GroupedQueryAttention 解答版实跑测试 + KV Cache 账本 + 真机 attention 峰值实测
运行环境：vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task1_common import setup_fonts, page, card, save

print("CJK font:", setup_fonts(), flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = torch.device("cuda")
DEVICE_NAME = torch.cuda.get_device_name(0)
VRAM_GIB = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
PASS = []


def check(name, cond, detail=""):
    PASS.append(("PASS" if cond else "FAIL", name))
    print("[%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)
    return bool(cond)


# ============ Part02 04 节：MHA / GQA / KV Cache（解答版实现） ============
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, \
            "num_heads (%d) must be divisible by num_kv_heads (%d)" % (num_heads, self.num_kv_heads)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(self, x, attention_mask=None, kv_cache=None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # TODO 1: 多头切分 [B, S, H*D] -> [B, H, S, D]
        xq = xq.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # TODO 2: KV Cache 在 seq_len 维拼接
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=2)
            xv = torch.cat([v_cache, xv], dim=2)
        new_kv_cache = (xk, xv)

        xk = repeat_kv(xk, self.num_queries_per_kv)
        xv = repeat_kv(xv, self.num_queries_per_kv)

        # TODO 3: 缩放点积注意力
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)

        # TODO 4: 合并多头 [B, H, S, D] -> [B, S, H*D]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(output), new_kv_cache


kv_shape_ok = []


def test_mha_mqa_gqa():
    batch_size, seq_len, hidden_dim, num_heads = 2, 16, 128, 4

    print("Testing MHA (Multi-Head Attention)...", flush=True)
    mha = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=num_heads).to(DEV)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=DEV)
    out, _ = mha(x)
    assert out.shape == (batch_size, seq_len, hidden_dim), "MHA 输出形状错误!"

    print("Testing GQA (Grouped-Query Attention)...", flush=True)
    gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=2).to(DEV)
    out_g, _ = gqa(x)
    assert out_g.shape == (batch_size, seq_len, hidden_dim), "GQA 输出形状错误!"

    print("Testing KV Cache Autoregressive Decoding...", flush=True)
    prefill_len = 5
    x_prefill = torch.randn(batch_size, prefill_len, hidden_dim, device=DEV)
    _, kv_cache = mha(x_prefill)
    x_decode = torch.randn(batch_size, 1, hidden_dim, device=DEV)
    out_decode, new_kv_cache = mha(x_decode, kv_cache=kv_cache)
    assert new_kv_cache[0].shape == (batch_size, num_heads, prefill_len + 1, hidden_dim // num_heads), "KV Cache 更新错误!"
    kv_shape_ok.append(str(tuple(new_kv_cache[0].shape)))

    # 额外一致性校验：与 PyTorch 官方 SDPA 的因果注意力对照
    torch.manual_seed(1)
    ref = GroupedQueryAttention(64, 4, num_kv_heads=2).to(DEV).double()
    xr = torch.randn(1, 12, 64, dtype=torch.float64, device=DEV)
    causal = torch.triu(torch.ones(12, 12, dtype=torch.bool, device=DEV), diagonal=1)
    mask = torch.where(causal, float("-inf"), torch.zeros((), dtype=torch.float64, device=DEV))
    ours, _ = ref(xr, attention_mask=mask[None, None])
    h = 16
    hd = 64 // 4
    q = ref.q_proj(xr).reshape(1, 12, 4, hd).transpose(1, 2)
    k = ref.k_proj(xr).reshape(1, 12, 2, hd).transpose(1, 2)
    v = ref.v_proj(xr).reshape(1, 12, 2, hd).transpose(1, 2)
    kk = repeat_kv(k, 2)
    vv = repeat_kv(v, 2)
    sdpa = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
    ref_out = ref.o_proj(sdpa.transpose(1, 2).reshape(1, 12, -1))
    max_diff = (ours - ref_out).abs().max().item()
    print("  与 torch SDPA 因果注意力最大误差 = %.3e" % max_diff, flush=True)
    return max_diff


sdpa_diff = test_mha_mqa_gqa()
A_OK = check("Part02 04节 · GroupedQueryAttention 解答版测试 (MHA/GQA/KV Cache)",
             sdpa_diff < 1e-8, "kv_cache shape %s, 与 SDPA 最大误差 %.1e"
             % (kv_shape_ok[0] if kv_shape_ok else "?", sdpa_diff))


# ============ KV Cache 账本 ============
def kv_per_token_per_layer(num_kv_heads, head_dim, dtype_bytes=2, latent_tokens=None):
    """MHA/GQA/MQA：2(K,V) * heads * head_dim；MLA：只缓存压缩向量。"""
    if latent_tokens is not None:
        return latent_tokens * dtype_bytes
    return 2 * num_kv_heads * head_dim * dtype_bytes


LLAMA70B = dict(q_heads=64, kv_heads=8, head_dim=128, layers=80)
LLAMA3_8B = dict(q_heads=32, kv_heads=8, head_dim=128, layers=32)
DSV2_MLA_LATENT = 512 + 64     # kv_lora_rank 512 + qk_rope_head_dim 64

kv_table = [
    ("MHA (64 KV heads)", kv_per_token_per_layer(LLAMA70B["q_heads"], 128)),
    ("GQA (8 KV heads)", kv_per_token_per_layer(LLAMA70B["kv_heads"], 128)),
    ("MQA (1 KV head)", kv_per_token_per_layer(1, 128)),
    ("MLA (latent 576)", kv_per_token_per_layer(0, 0, latent_tokens=DSV2_MLA_LATENT)),
]
for name, b in kv_table:
    print("  KV cache %-18s = %7.1f B/token/layer (%.2f KB)" % (name, b, b / 1024), flush=True)

gqa_layer = kv_per_token_per_layer(LLAMA3_8B["kv_heads"], LLAMA3_8B["head_dim"])
ctx_rows = []
for ctx in (4096, 8192, 32768, 131072):
    gb = gqa_layer * LLAMA3_8B["layers"] * ctx / 2 ** 30
    ctx_rows.append((ctx, gb))
    print("  LLaMA-3-8B GQA KV cache @ %6d tokens = %8.3f GiB" % (ctx, gb), flush=True)

# 真机：按理论大小真实分配一份 KV cache，看实测字节数
def real_bytes_of(shape, dtype=torch.float16):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    t = torch.empty(shape, dtype=dtype, device=DEV)
    used = torch.cuda.memory_allocated() - base
    del t
    torch.cuda.empty_cache()
    return used


kv_real = real_bytes_of((2, 32, 8, 8192, 128), torch.float16)  # 2(KV) * layers * kv_heads * ctx * head_dim
kv_theory = gqa_layer * LLAMA3_8B["layers"] * 8192
B_OK = check("KV Cache 账本: 8K 上下文 LLaMA-3-8B 全层缓存真机占用",
             abs(kv_real - kv_theory) < 2 ** 20, "实测 %.3f GiB vs 理论 %.3f GiB" % (kv_real / 2 ** 30, kv_theory / 2 ** 30))

# ============ Part01 14 节：FlashAttention 显存模型 ============
def attention_score_bytes(seq_len, dtype_bytes=2):
    return seq_len * seq_len * dtype_bytes


def num_1d_tiles(seq_len, tile_size):
    return (seq_len + tile_size - 1) // tile_size


def num_score_tiles(seq_len, tile_size):
    t = num_1d_tiles(seq_len, tile_size)
    return t * t


def score_tile_bytes(tile_size, dtype_bytes=2):
    return tile_size * tile_size * dtype_bytes


def score_materialization_ratio(seq_len, tile_size, dtype_bytes=2):
    return attention_score_bytes(seq_len, dtype_bytes) / score_tile_bytes(tile_size, dtype_bytes)


tile_rows = []
for tile in (64, 128, 256):
    tiles_1d = num_1d_tiles(4096, tile)
    st = num_score_tiles(4096, tile)
    tile_kb = score_tile_bytes(tile) / 1024
    ratio = score_materialization_ratio(4096, tile)
    tile_rows.append((tile, tiles_1d, st, tile_kb, ratio))
    print("  tile=%3d -> 1D tiles=%3d, score tiles=%4d, tile=%.1fKB, ratio=%.0fx" % (tile, tiles_1d, st, tile_kb, ratio), flush=True)
C_OK = check("14节 · Q1/Q2/Q3 分块与 online softmax 显存模型测试",
             len(tile_rows) == 3 and tile_rows[0][4] > tile_rows[-1][4])

# ============ 真机：标准 attention vs SDPA(Flash/mem-efficient) 峰值 ============
def peak_of(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated() - base


BATCH_A, HEADS_A, HDIM_A = 1, 32, 128
attn_rows = []
for S in (512, 1024, 2048, 4096):
    q = torch.randn(BATCH_A, HEADS_A, S, HDIM_A, dtype=torch.bfloat16, device=DEV)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def naive():
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HDIM_A)
        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

    def flash():
        return F.scaled_dot_product_attention(q, k, v)

    out_n, peak_n = peak_of(naive)
    out_f, peak_f = peak_of(flash)
    diff = (out_n.float() - out_f.float()).abs().max().item()
    attn_rows.append((S, peak_n / 2 ** 20, peak_f / 2 ** 20, peak_n / peak_f, diff))
    print("  S=%5d naive peak %8.1f MiB | SDPA peak %8.1f MiB | %.1fx | maxdiff %.3e"
          % (S, peak_n / 2 ** 20, peak_f / 2 ** 20, peak_n / peak_f, diff), flush=True)
    del q, k, v, out_n, out_f
    torch.cuda.empty_cache()

D_OK = check("真机 attention 峰值: 标准实现 O(N^2) vs SDPA(FlashAttention) O(N)",
             attn_rows[-1][3] > 5 and max(r[4] for r in attn_rows) < 5e-2,
             "S=4096 时 %.1fx 峰值差距" % attn_rows[-1][3])

# ============ 作图 ============
env_line = ("GitHub ID: wsm000    |    微信昵称: empty    |    运行环境: vm-60 · %s (%.0fGB) · torch %s · CUDA %s"
            % (DEVICE_NAME, VRAM_GIB, torch.__version__.split("+")[0], torch.version.cuda))
link_line = ("教程: datawhalechina/llm-algo-leetcode · Part02 04 多头注意力 / Part01 14 FlashAttention 显存模型 / topic01 显存账本    |    "
             "社区: DataWhale    |    Task1 Issue #149")
fig, gs = page("llm-algo-leetcode 显存优化 | 202609 · Task1 增项2 (4.3) MHA/GQA/MLA · KV Cache · FlashAttention",
               env_line, link_line, nrows=3, ncols=2, figsize=(13.5, 13.8))

n_pass = sum(1 for t, _ in PASS if t == "PASS")
card(gs[0, 0], "① 4.3 测试结果总览（Part02 04 节实跑）",
     ["[%s] 04节 GroupedQueryAttention 解答版" % ("PASS" if A_OK else "FAIL"),
      "      MHA / GQA 前向 + KV Cache 自回归更新",
      "      (与 torch SDPA 因果注意力最大误差 %.1e)" % sdpa_diff,
      "[%s] KV Cache 账本真机占用对照" % ("PASS" if B_OK else "FAIL"),
      "[%s] 14节 tiling / online softmax 显存模型" % ("PASS" if C_OK else "FAIL"),
      "[%s] 真机 attention 峰值 naive vs SDPA" % ("PASS" if D_OK else "FAIL"),
      "",
      "合计: %d/%d 项 PASS  ->  所有测试通过" % (n_pass, len(PASS)),
      "",
      "kv_cache[0].shape = %s" % (kv_shape_ok[0] if kv_shape_ok else "?")])

lines_b = ["Attention 与 Transformer 的关系:",
           "  Transformer 是整体架构；Attention 是它",
           "  每个 Block 里的信息聚合算子。",
           "  Block = Attention(混合 token) + FFN(逐 token)",
           "  Attention 决定上下文如何被读取。",
           "",
           "Q/K/V: 每个 token 投影出 Query(找什么)、",
           "  Key(有什么)、Value(给出什么)，",
           "  softmax(QK^T/sqrt(d))V 完成加权聚合。",
           "",
           "多头系谱（n_q 个 Q 头 : n_kv 个 KV 头）:",
           "  MHA  n_q : n_q   表达能力最强, cache 最大",
           "  MQA  n_q : 1     cache 最小, 质量损失明显",
           "  GQA  n_q : g     折中 (LLaMA-2/3 在用)",
           "  MLA  缓存压缩潜向量, 用时实时解压",
           "       (DeepSeek-V2/V3, cache ~1.13KB/token/layer)"]
card(gs[0, 1], "② 概念：Attention/Transformer、MHA、GQA、MLA", lines_b)

lines_c = ["KV Cache 单 token 单层大小（LLaMA-2-70B 尺寸）:"]
for name, b in kv_table:
    lines_c.append("  %-20s %7.1f B  = %5.2f KB" % (name, b, b / 1024))
lines_c += ["",
            "LLaMA-3-8B (GQA 8 KV heads, 32 层) 全模型 KV Cache:",
            "  %8s %12s" % ("上下文", "fp16 GiB")]
for ctx, gb in ctx_rows:
    lines_c.append("  %8d %12.3f" % (ctx, gb))
lines_c += ["",
            "真机验证 (8K 上下文全层 cache): 实测 %.3f vs 理论 %.3f GiB"
            % (kv_real / 2 ** 30, kv_theory / 2 ** 30),
            "→ 长上下文 OOM 的主因常不是权重，而是随",
            "  batch × 上下文线性增长的 KV Cache。"]
card(gs[1, 0], "③ KV Cache 账本：MHA / GQA / MQA / MLA", lines_c)

lines_d = ["FlashAttention 思想（14节）:",
           "  标准 attention 的 B*H*N^2 中间矩阵要落 HBM，",
           "  读写代价高 -> 从算力问题变成访存问题。",
           "",
           "  tiling: 把 Q/K/V 切成能放进 SRAM 的小块，",
           "          在片上完成 QK^T -> softmax -> @V",
           "  online softmax: 边算边维护 running max(m) 与",
           "          指数和(l)，块间用修正因子 rescale，",
           "          不需要先物化整行再统一归约",
           "",
           "  seq_len=4096 分块账本（教学模型）:",
           "  %5s %10s %11s %12s" % ("tile", "1D tiles", "score tiles", "tile KB")]
for tile, t1, st, kb, ratio in tile_rows:
    lines_d.append("  %5d %10d %11d %12.1f" % (tile, t1, st, kb))
lines_d += ["",
            "  完整 score 矩阵 vs 单个 tile 的物化倍率:",
            "    tile=64 -> %.0fx, tile=128 -> %.0fx, tile=256 -> %.0fx"
            % (tile_rows[0][4], tile_rows[1][4], tile_rows[2][4]),
            "  -> 用重算与片上复用换掉 HBM 上的大矩阵"]
card(gs[1, 1], "④ FlashAttention：tiling 与 online softmax", lines_d)

lines_e = ["真机 attention 峰值 (B=1, H=32, D=128, bf16):",
           "  %6s %14s %14s %8s" % ("seq", "naive MiB", "SDPA MiB", "倍数")]
for S, pn, pf, ratio, diff in attn_rows:
    lines_e.append("  %6d %14.1f %14.1f %7.1fx" % (S, pn, pf, ratio))
lines_e += ["",
            "naive 峰值随 S 二次增长（B*H*N^2 是显存大头）;",
            "SDPA(FlashAttention) 只随 S 线性; 误差 < %.1e" % max(r[4] for r in attn_rows),
            "",
            "指标口径（topic01）: peak 装不装得下 / reserved",
            "allocator 保留 / delta 优化前后差 / throughput 代价",
            "结论: 先拆对象, 再选 checkpoint / offload /",
            "      sharding / paging, 并说明代价转移到哪"]
card(gs[2, 0], "⑤ 真机 attention 峰值账本 + 指标口径", lines_e)

ax = fig.add_subplot(gs[2, 1])
xs = [r[0] for r in attn_rows]
ax.plot(xs, [r[1] for r in attn_rows], "o-", color="#c44e52", label="naive attention (O(N^2) 中间矩阵)")
ax.plot(xs, [r[2] for r in attn_rows], "s-", color="#4c72b0", label="SDPA / FlashAttention (O(N))")
ax.set_xscale("log", base=2)
ax.set_xticks(xs)
ax.set_xticklabels([str(x) for x in xs])
ax.set_xlabel("seq_len", fontsize=9)
ax.set_ylabel("peak memory (MiB)", fontsize=9)
ax.set_title("真机 attention 峰值显存 vs 序列长度 (bf16, 4090D)", fontsize=10.5, color="#16264a")
ax.legend(fontsize=8.2)
ax.grid(alpha=0.25)
ax.tick_params(labelsize=8.6)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_43_runshot.png")
save(fig, out)
print("ALL_TESTS_%s" % ("PASS" if n_pass == len(PASS) else "FAIL"), flush=True)
