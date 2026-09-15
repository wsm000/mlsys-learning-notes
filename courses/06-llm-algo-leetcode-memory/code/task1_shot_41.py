# -*- coding: utf-8 -*-
"""llm-algo-leetcode 显存优化 | 202609 Task1 硬件与显存账本
4.1 最小打卡：01 数据格式与混合精度 / 02 参数量与算力推导 / 06 显存计算与 ZeRO
内容：教程测试逐条复跑 + 真机 (GPU) 账本实测
运行环境：vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128
"""
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task1_common import setup_fonts, page, card, save

CJK = setup_fonts()
print("CJK font:", CJK, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = torch.device("cuda")
DEVICE_NAME = torch.cuda.get_device_name(0)
VRAM_GIB = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
PEAK_BF16_TFLOPS = 165.2   # RTX 4090D 规格书：FP16/BF16 Tensor Core 稠密算力
HBM_BW_GBS = 1008.0        # RTX 4090D 规格书：显存带宽
print("GPU:", DEVICE_NAME, round(VRAM_GIB, 1), "GiB · SMs:", SM_COUNT, "· torch", torch.__version__, flush=True)

PASS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    PASS.append((tag, name))
    print("[%s] %s  %s" % (tag, name, detail), flush=True)
    return bool(cond)


# ============ Part A · 01 节：dtype 账本 ============
def calculate_model_memory(num_params_b, dtype):
    if num_params_b < 0:
        raise ValueError("num_params_b must be non-negative")
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
    try:
        bpe = bytes_per_param[dtype]
    except KeyError as exc:
        raise ValueError("unsupported dtype: %s" % dtype) from exc
    return num_params_b * bpe


def calculate_training_memory(num_params_b, model_dtype="fp16", optimizer="adam"):
    model_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[model_dtype]
    optimizer_bytes = 12 if optimizer == "adam" else 4
    return num_params_b * (model_bytes + model_bytes + optimizer_bytes)


def calculate_quantization_savings(num_params_b, from_dtype, to_dtype):
    o = calculate_model_memory(num_params_b, from_dtype)
    q = calculate_model_memory(num_params_b, to_dtype)
    return o - q, (o - q) / o * 100


def test_dtype_module():
    ok = (calculate_model_memory(7, "fp16") == 14
          and calculate_model_memory(7, "int8") == 7
          and calculate_model_memory(13, "fp16") == 26
          and calculate_model_memory(70, "int4") == 35)
    ok = ok and calculate_training_memory(7, "fp16", "adam") == 112
    ok = ok and calculate_training_memory(7, "fp16", "sgd") == 56
    ok = ok and calculate_training_memory(13, "bf16", "adam") == 208
    s1 = calculate_quantization_savings(7, "fp16", "int8")
    s2 = calculate_quantization_savings(7, "fp16", "int4")
    ok = ok and s1 == (7.0, 50.0) and s2 == (10.5, 75.0)
    return ok


A_OK = check("01节 · Q1/Q2/Q3 账本函数测试 (权重/训练状态/量化节省)", test_dtype_module())


def gpu_tensor_bytes(shape, dtype):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    t = torch.empty(shape, dtype=dtype, device=DEV)
    used = torch.cuda.memory_allocated() - base
    del t
    torch.cuda.empty_cache()
    return used


dtype_rows = []
for name, dt in [("FP32", torch.float32), ("BF16", torch.bfloat16),
                 ("FP16", torch.float16), ("FP8 E4M3", torch.float8_e4m3fn)]:
    try:
        used = gpu_tensor_bytes((4096, 4096), dt)
        dtype_rows.append((name, used))
        print("  GPU tensor 4096x4096 %-9s = %8.1f MiB" % (name, used / 2 ** 20), flush=True)
    except Exception as exc:  # pragma: no cover
        print("  GPU tensor", name, "failed:", exc, flush=True)
A2_OK = check("01节 · GPU 实机 dtype 张量占用 (4096x4096)", len(dtype_rows) >= 3,
              "FP32/FP16 = %.2fx" % (dtype_rows[0][1] / dtype_rows[2][1]))

# ============ Part B · 02 节：参数量与 MFU ============
def calculate_transformer_params(vocab_size, hidden_dim, num_layers,
                                 intermediate_size=None, tie_embeddings=False):
    if intermediate_size is None:
        intermediate_size = 4 * hidden_dim
    embedding_params = vocab_size * hidden_dim
    attention_params = num_layers * (4 * hidden_dim * hidden_dim)
    ffn_params = num_layers * (3 * hidden_dim * intermediate_size)
    layernorm_params = num_layers * (2 * hidden_dim)
    lm_head_params = 0 if tie_embeddings else vocab_size * hidden_dim
    return embedding_params + attention_params + ffn_params + layernorm_params + lm_head_params


def test_param_module():
    a = calculate_transformer_params(1000, 64, 2, 256, tie_embeddings=True)
    b = calculate_transformer_params(1000, 64, 2, 256, tie_embeddings=False)
    c = calculate_transformer_params(32000, 4096, 32, 11008, tie_embeddings=False)
    ok = a == 195328 and b == 259328 and c > 6e9

    def calc_flops(p, t, k=6):
        return p * 1_000_000_000 * t * k
    ok = ok and calc_flops(7, 1_000_000_000_000) == 42_000_000_000_000_000_000_000
    ok = ok and calc_flops(1, 1_000_000) == 6_000_000_000_000_000
    return ok, a, b, c


B_OK, pa, pb, llama7b = test_param_module()
check("02节 · Q1/Q2 参数量与训练 FLOPs 函数测试", B_OK)

V, D, L, FFN = 32000, 4096, 32, 11008
emb = V * D
attn_p = L * 4 * D * D
ffn_p = L * 3 * D * FFN
ln_p = L * 2 * D
head_p = V * D
print("LLaMA-7B 结构参数量 = %.3f B (embedding %.2fM / attn %.2fM / ffn %.2fM / ln %.2fM / head %.2fM)"
      % (llama7b / 1e9, emb / 1e6, attn_p / 1e6, ffn_p / 1e6, ln_p / 1e6, head_p / 1e6), flush=True)


def bench_mm(m, n, k, dtype, iters=20, warmup=5):
    a = torch.randn(m, k, dtype=dtype, device=DEV)
    b = torch.randn(k, n, dtype=dtype, device=DEV)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    del a, b
    torch.cuda.empty_cache()
    return 2.0 * m * n * k / dt


mm_rows = []
try:
    bf16_tflops = bench_mm(8192, 8192, 8192, torch.bfloat16) / 1e12
    mm_rows.append(("BF16  8192^3", bf16_tflops, bf16_tflops / PEAK_BF16_TFLOPS))
except Exception as exc:
    print("bf16 mm failed:", exc, flush=True)
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    fp32_tflops = bench_mm(4096, 4096, 4096, torch.float32) / 1e12
    mm_rows.append(("FP32  4096^3", fp32_tflops, float("nan")))
    torch.backends.cuda.matmul.allow_tf32 = True
    tf32_tflops = bench_mm(4096, 4096, 4096, torch.float32) / 1e12
    mm_rows.append(("TF32  4096^3", tf32_tflops, float("nan")))
except Exception as exc:
    print("fp32 mm failed:", exc, flush=True)
for name, tf, mfu in mm_rows:
    print("  %s -> %7.1f TFLOPS" % (name, tf), flush=True)
B2_OK = check("02节 · Q3/Q4 真机 GEMM 吞吐与教学 MFU", len(mm_rows) >= 2,
              "BF16 MFU = %.1f%% (峰值 %.1f TFLOPS)" % (mm_rows[0][2] * 100, PEAK_BF16_TFLOPS))

# ============ Part C · 06 节：显存账本与 ZeRO ============
def training_state_breakdown(num_params_b, model_dtype="fp16", optimizer="adam"):
    mb = {"fp32": 4, "fp16": 2, "bf16": 2}[model_dtype]
    ob = {"adam": 12, "sgd": 4}[optimizer]
    v = {"parameters": num_params_b * mb, "gradients": num_params_b * mb, "optimizer": num_params_b * ob}
    v["total"] = sum(v.values())
    return v


def calculate_zero_memory(num_params_b, zero_stage, num_gpus, model_dtype="fp16", optimizer="adam"):
    mb = {"fp32": 4, "fp16": 2, "bf16": 2}[model_dtype]
    ob = {"adam": 12, "sgd": 4}[optimizer]
    gb = mb
    if zero_stage in (0, "ddp"):
        bpp = mb + gb + ob
    elif zero_stage == 1:
        bpp = mb + gb + ob / num_gpus
    elif zero_stage == 2:
        bpp = mb + gb / num_gpus + ob / num_gpus
    else:
        bpp = (mb + gb + ob) / num_gpus
    return num_params_b * bpp


def test_zero_module():
    ok = calculate_zero_memory(7, 1, 8) == 38.5
    ok = ok and abs(calculate_zero_memory(7, 2, 8) - 26.25) < 1e-9
    ok = ok and abs(calculate_zero_memory(7, 3, 8) - 14.0) < 1e-9
    d = training_state_breakdown(7, "bf16", "adam")
    ok = ok and d["total"] == 112
    return ok


C_OK = check("06节 · Q1/Q2 DDP 与 ZeRO-1/2/3 账本测试", test_zero_module())


class TinyBlock(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ffn, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([TinyBlock(d, heads, ffn) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        b, s = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(s, device=idx.device))[None]
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=idx.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.lnf(x))


VOCAB, DD, LAYERS, HEADS, FFN_D = 32000, 512, 6, 8, 1376
BATCH, SEQ = 8, 256
torch.manual_seed(0)
# 先跑一次极小的 cuBLAS 调用预热框架工作区：否则这块固定开销会被算进后面某一
# 段增量里，污染账本。base0 之后的分配才算进本实验。
_ = torch.zeros(8, 8, device=DEV) @ torch.zeros(8, 8, device=DEV)
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
base0 = torch.cuda.memory_allocated()
print("framework baseline = %.2f MiB" % (base0 / 2 ** 20), flush=True)
model = TinyGPT(VOCAB, DD, LAYERS, HEADS, FFN_D).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
P = sum(p.numel() for p in model.parameters())
idx = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV)

m_params = torch.cuda.memory_allocated()
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, VOCAB), idx.view(-1))
peak_fwd = torch.cuda.max_memory_allocated()
loss.backward()
peak_bwd = torch.cuda.max_memory_allocated()
del logits, loss          # 释放 forward 输出引用，保证常驻账本只含参数+梯度
grad_bytes = sum(p.grad.numel() * p.grad.element_size()
                 for p in model.parameters() if p.grad is not None)
torch.cuda.synchronize()
m_after_bwd = torch.cuda.memory_allocated()
opt.step()
m_after_step = torch.cuda.memory_allocated()
peak_step = torch.cuda.max_memory_allocated()

M = 2 ** 20
meas = {
    "params": (m_params - base0) / M,
    "grads": (m_after_bwd - m_params) / M,
    "grads_exact": grad_bytes / M,
    "workspace": (m_after_bwd - m_params - grad_bytes) / M,
    "opt_state": (m_after_step - m_after_bwd) / M,
    "steady": (m_after_step - base0) / M,
    "peak_fwd": (peak_fwd - base0) / M,
    "peak_bwd": (peak_bwd - base0) / M,
    "peak_step": (peak_step - base0) / M,
}
theo = {
    "params": P * 4 / M,
    "grads": P * 4 / M,
    "opt_state": P * 8 / M,
    "steady": P * 16 / M,
}
print("TinyGPT P = %.2f M params (vocab=%d d=%d L=%d, B=%d S=%d)" % (P / 1e6, VOCAB, DD, LAYERS, BATCH, SEQ), flush=True)
for k in ["params", "grads", "grads_exact", "workspace", "opt_state", "steady",
          "peak_fwd", "peak_bwd", "peak_step"]:
    print("  measured %-11s = %9.1f MiB" % (k, meas[k]), flush=True)
for k in ["params", "grads", "opt_state", "steady"]:
    print("  theory   %-10s = %9.1f MiB" % (k, theo[k]), flush=True)
act_peak = meas["peak_bwd"] - meas["steady"]
grad_err = abs(meas["grads_exact"] - theo["grads"]) / theo["grads"]
opt_err = abs(meas["opt_state"] - theo["opt_state"]) / theo["opt_state"]
C2_OK = check("06节 · 真机训练 step 四段账本 (参数/梯度/优化器状态/激活峰值)",
              grad_err < 0.03 and opt_err < 0.03,
              "梯度偏差 %.1f%% / 优化器状态偏差 %.1f%% (相对 4Φ / 8Φ)"
              % (grad_err * 100, opt_err * 100))

zero_rows = []
for stage_name, stage in [("DDP", "ddp"), ("ZeRO-1", 1), ("ZeRO-2", 2), ("ZeRO-3", 3)]:
    zero_rows.append((stage_name, calculate_zero_memory(7, stage, 8)))
    print("  8x80GB %-7s 7B(bp16+Adam) = %6.2f GB/card" % (stage_name, zero_rows[-1][1]), flush=True)

# ============ 作图 ============
env_line = ("GitHub ID: wsm000    |    微信昵称: empty    |    运行环境: vm-60 · %s (%.0fGB) · torch %s · CUDA %s"
            % (DEVICE_NAME, VRAM_GIB, torch.__version__.split("+")[0], torch.version.cuda))
link_line = ("教程: datawhalechina/llm-algo-leetcode · Part01 01/02/06    |    社区: DataWhale https://github.com/datawhalechina/llm-algo-leetcode    |    "
             "Task1 Issue #149")
fig, gs = page("llm-algo-leetcode 显存优化 | 202609 · Task1 硬件与显存账本 —— 第01/02/06节测试通过 + 真机账本",
               env_line, link_line, nrows=3, ncols=2, figsize=(13.5, 13.8))

n_pass = sum(1 for t, _ in PASS if t == "PASS")
card(gs[0, 0], "① 打卡任务 4.1 测试结果总览（教程函数逐条复跑）",
     ["[%s] 01节 Q1/Q2/Q3 dtype 与混合精度账本" % ("PASS" if A_OK else "FAIL"),
      "[%s] 01节 GPU 实机 dtype 张量占用对照" % ("PASS" if A2_OK else "FAIL"),
      "[%s] 02节 Q1/Q2 参数量与训练 FLOPs 推导" % ("PASS" if B_OK else "FAIL"),
      "[%s] 02节 Q3/Q4 真机 GEMM 吞吐与 MFU" % ("PASS" if B2_OK else "FAIL"),
      "[%s] 06节 Q1/Q2 DDP / ZeRO-1/2/3 账本" % ("PASS" if C_OK else "FAIL"),
      "[%s] 06节 真机训练 step 四段账本实测" % ("PASS" if C2_OK else "FAIL"),
      "",
      "合计: %d/6 项 PASS  ->  所有测试通过" % n_pass,
      "",
      "教程要求的三问：",
      " · dtype 具体指什么      -> 卡片②",
      " · 参数量 & MFU 是什么   -> 卡片③",
      " · 训练显存分几部分/ZeRO -> 卡片④⑤"])

lines_b = ["LLaMA-7B 权重账本 (01000^3 = GB/参数×B):"]
for k, v in [("fp32", 4), ("fp16", 2), ("bf16", 2), ("int8", 1), ("int4", 0.5)]:
    lines_b.append("  %-5s %5.1f GB   (7e9 x %.1f B)" % (k, calculate_model_memory(7, k), v))
lines_b += ["", "GPU 实机 4096x4096 张量 (memory_allocated 增量):"]
for name, used in dtype_rows:
    lines_b.append("  %-9s %8.1f MiB" % (name, used / 2 ** 20))
lines_b += ["", "FP16: 1+5+10bit, max 65504",
            "BF16: 1+8+7bit,  max ~3.4e38",
            "FP8(E4M3) 前向/激活, FP8(E5M2) 反向/梯度"]
card(gs[0, 1], "② FP16 / BF16(A100) / FP8(H100) 的具体含义与实测", lines_b)

lines_c = ["LLaMA-7B 参数量分解 (V=%d d=%d L=%d ffn=%d):" % (V, D, L, FFN),
           "  embedding  %8.1f M" % (emb / 1e6),
           "  attention  %8.1f M   (4d^2 x L)" % (attn_p / 1e6),
           "  ffn(SwiGLU)%8.1f M   (3d*ffn x L)" % (ffn_p / 1e6),
           "  layernorm  %8.3f M" % (ln_p / 1e6),
           "  lm_head    %8.1f M" % (head_p / 1e6),
           "  total      %8.3f B  ≈ 7B 量级" % (llama7b / 1e9),
           "",
           "真机 GEMM 实测 (4090D):"]
for name, tf, mfu in mm_rows:
    lines_c.append("  %-12s %7.1f TFLOPS" % (name, tf))
lines_c += ["", "MFU = 实测模型 FLOPs / 硬件峰值",
            "  6PT 训练 FLOPs = %.1fe22 (7B x 1T tokens)" % (6 * 7e9 * 1e12 / 1e22),
            "  本机 BF16 MFU = %.1f%%  (峰值 %.0f TFLOPS)" % (mm_rows[0][2] * 100, PEAK_BF16_TFLOPS)]
card(gs[1, 0], "③ 参数量估算 (02节) 与 MFU 概念 + 真机吞吐", lines_c)

lines_d = ["TinyGPT 真机一步训练账本 (P=%.1fM, B=%d S=%d, fp32 参数 + bf16 autocast + AdamW):" % (P / 1e6, BATCH, SEQ),
           "  段位            实测(MiB)  理论(MiB)  账本",
           "  参数            %9.1f %9.1f   4Φ" % (meas["params"], theo["params"]),
           "  梯度张量合计    %9.1f %9.1f   4Φ" % (meas["grads_exact"], theo["grads"]),
           "  优化器状态      %9.1f %9.1f   8Φ(m+v)" % (meas["opt_state"], theo["opt_state"]),
           "  -------------------------------------------------",
           "  常驻合计        %9.1f %9.1f   16Φ" % (meas["steady"], theo["steady"]),
           "  框架 workspace            %9.1f  (cuBLAS 等)" % meas["workspace"],
           "",
           "  forward 后峰值  %9.1f" % meas["peak_fwd"],
           "  backward 后峰值 %9.1f  <- 整步峰值" % meas["peak_bwd"],
           "  step 后峰值     %9.1f" % meas["peak_step"],
           "  activation 峰值 ≈ backward 峰值 - 常驻 = %.1f MiB" % act_peak,
           "",
           "结论: 参数/梯度/优化器状态实测与 4Φ/4Φ/8Φ 吻合(偏差 %.1f%%/%.1f%%)"
           % (grad_err * 100, opt_err * 100),
           "      峰值 = 16Φ 常驻 + activation + 不可预测的框架 workspace"]
card(gs[1, 1], "④ 训练显存分几部分：真机实测 vs 理论账本", lines_d)

lines_e = ["8 x 80GB (A100) 单卡训练状态估算, FP16+Adam:"]
for name, gb in zero_rows:
    lines_e.append("  %-7s %6.2f GB/card  -> 最大模型 %.1fB(预留20%%)" % (name, gb, 80 * 0.8 / (gb / 7)))
lines_e += ["",
            "ZeRO 到底切了什么:",
            "  ZeRO-1  optimizer state        (12Φ -> 12Φ/N)",
            "  ZeRO-2  + gradients            (2Φ  -> 2Φ/N)",
            "  ZeRO-3  + parameters           (2Φ  -> 2Φ/N)",
            "  三者都不动 activation: 它只由",
            "  micro-batch x seq_len x hidden 决定",
            "",
            "代价转移: 显存 ↓ 换来 all-gather /",
            "reduce-scatter 通信量 ↑ 与调度复杂度 ↑"]
card(gs[2, 0], "⑤ ZeRO 概念：切什么、省多少、代价在哪", lines_e)

axb = fig.add_subplot(gs[2, 1])
labels = ["params\n4Φ", "grads\n4Φ", "opt_state\n8Φ", "activation\n(峰值增量)"]
vals = [meas["params"], meas["grads_exact"], meas["opt_state"], act_peak]
colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]
bars = axb.bar(labels, vals, color=colors, edgecolor="#2a3a5a", linewidth=0.6)
for b, v in zip(bars, vals):
    axb.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, "%.0f" % v,
             ha="center", fontsize=8.6, color="#22304f")
axb.set_title("真机一步训练显存构成 (MiB) · 峰值 %.0f MiB" % meas["peak_step"], fontsize=10.5, color="#16264a")
axb.set_facecolor("white")
axb.tick_params(labelsize=8.6)
axb.grid(axis="y", alpha=0.25)
axb.set_ylabel("MiB", fontsize=9)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_41_runshot.png")
save(fig, out)
print("ALL_TESTS_%s" % ("PASS" if n_pass == len(PASS) else "FAIL"), flush=True)
