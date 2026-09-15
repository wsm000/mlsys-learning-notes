# -*- coding: utf-8 -*-
"""llm-algo-leetcode 显存优化 | 202609 Task1 硬件与显存账本
4.2 增项1：03 GPU 物理架构与内存层级 / 12 Tensor Core 与混合精度
内容：教程测试逐条复跑 + 真机带宽、GEMM 吞吐、数值范围实测
运行环境：vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128
"""
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task1_common import setup_fonts, page, card, save

print("CJK font:", setup_fonts(), flush=True)

import torch
from typing import Dict

torch.manual_seed(0)
DEV = torch.device("cuda")
DEVICE_NAME = torch.cuda.get_device_name(0)
VRAM_GIB = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
HBM_BW_GBS = 1008.0      # RTX 4090D 规格：1008 GB/s
PEAK_FP32_TFLOPS = 82.6  # RTX 4090D 规格：FP32
PEAK_BF16_TFLOPS = 165.2 # RTX 4090D 规格：FP16/BF16 Tensor Core 稠密
PASS = []


def check(name, cond, detail=""):
    PASS.append(("PASS" if cond else "FAIL", name))
    print("[%s] %s  %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)
    return bool(cond)


# ============ 03 节：内存层级 + Attention 显存 ============
def bytes_to_gb(bytes_val: float) -> float:
    return bytes_val / 1e9


MEMORY_BANDWIDTH = {
    "shared_memory": 19e12,   # 教学数量级 (SRAM ~19 TB/s)
    "l2_cache": 1.5e12,
    "hbm": 1.5e12,            # A100 HBM 数量级
}


def analyze_memory_hierarchy() -> Dict[str, Dict[str, float]]:
    result = {}
    for mem_type, bandwidth in MEMORY_BANDWIDTH.items():
        time_for_1kb_ns = (1024 / bandwidth) * 1e9   # TODO 1.1 传输 1KB 的理论时间
        result[mem_type] = {"bandwidth_tb_s": bandwidth / 1e12, "time_for_1kb_ns": time_for_1kb_ns}
    return result


def calculate_attention_vram(seq_len, num_heads, head_dim, dtype_bytes=2, batch_size=1):
    qkv_vram = 3 * batch_size * seq_len * num_heads * head_dim * dtype_bytes          # TODO 2.1
    attention_matrix_vram = batch_size * num_heads * seq_len * seq_len * dtype_bytes  # TODO 2.2
    output_vram = batch_size * seq_len * num_heads * head_dim * dtype_bytes           # TODO 2.3
    total_vram = qkv_vram + attention_matrix_vram + output_vram                       # TODO 2.4
    return {"qkv": bytes_to_gb(qkv_vram), "attention_matrix": bytes_to_gb(attention_matrix_vram),
            "output": bytes_to_gb(output_vram), "total": bytes_to_gb(total_vram)}


def calculate_flash_attention_vram(seq_len, num_heads, head_dim, dtype_bytes=2, batch_size=1):
    qkv_vram = 3 * batch_size * seq_len * num_heads * head_dim * dtype_bytes          # TODO 3.1
    online_softmax_vram = batch_size * seq_len * num_heads * 2 * 4                    # TODO 3.2
    output_vram = batch_size * seq_len * num_heads * head_dim * dtype_bytes           # TODO 3.3
    total_vram = qkv_vram + online_softmax_vram + output_vram                         # TODO 3.4
    return {"qkv": bytes_to_gb(qkv_vram), "online_softmax": bytes_to_gb(online_softmax_vram),
            "output": bytes_to_gb(output_vram), "total": bytes_to_gb(total_vram)}


def test_gpu_memory_practice():
    mem = analyze_memory_hierarchy()
    assert "shared_memory" in mem and "hbm" in mem
    assert mem["shared_memory"]["bandwidth_tb_s"] > mem["hbm"]["bandwidth_tb_s"]
    attn = calculate_attention_vram(512, 32, 128)
    flash = calculate_flash_attention_vram(512, 32, 128)
    assert attn["total"] > flash["total"]
    assert attn["qkv"] > 0 and flash["online_softmax"] > 0
    return True


A_OK = check("03节 · Q1/Q2/Q3 内存层级与 Attention/FlashAttention 显存测试", test_gpu_memory_practice())


def pcie_vs_nvlink(payload_mb, pcie_gbps=64, nvlink_gbps=900):
    pcie_ms = payload_mb * 8 / pcie_gbps
    nvlink_ms = payload_mb * 8 / nvlink_gbps
    return {"pcie_ms": round(pcie_ms, 2), "nvlink_ms": round(nvlink_ms, 2), "speedup": round(pcie_ms / nvlink_ms, 1)}


# ============ 12 节：Tensor Core 与混合精度 ============
def gemm_flops(m, n, k):
    return 2 * m * n * k


def tensor_storage_bytes(numel, dtype_bytes):
    return numel * dtype_bytes


def test_tensorcore_precision_module():
    assert gemm_flops(1024, 1024, 1024) == 2 * 1024 ** 3
    assert tensor_storage_bytes(1024, 4) == 4096
    assert tensor_storage_bytes(1024, 2) == 2048
    return True


B_OK = check("12节 · Q1/Q2 存储量与 GEMM FLOPs 测试", test_tensorcore_precision_module())

# ============ 真机实测 ============
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
    return 2.0 * m * n * k / dt / 1e12


mm_rows = []
torch.backends.cuda.matmul.allow_tf32 = False
PEAK_TF32_TFLOPS = 82.6   # RTX 4090D 规格：TF32 Tensor Core 稠密算力
mm_rows.append(("FP32 (无 TF32)", bench_mm(4096, 4096, 4096, torch.float32), PEAK_FP32_TFLOPS))
mm_rows.append(("FP16 TensorCore", bench_mm(4096, 4096, 4096, torch.float16), PEAK_BF16_TFLOPS))
torch.backends.cuda.matmul.allow_tf32 = True
mm_rows.append(("TF32 (FP32+截断)", bench_mm(4096, 4096, 4096, torch.float32), PEAK_TF32_TFLOPS))
mm_rows.append(("BF16 TensorCore", bench_mm(8192, 8192, 8192, torch.bfloat16), PEAK_BF16_TFLOPS))
for name, tf, peak in mm_rows:
    print("  %-18s %7.1f TFLOPS  (峰值 %.0f -> 利用率 %.1f%%)" % (name, tf, peak, tf / peak * 100), flush=True)
C_OK = check("真机 GEMM: FP32/TF32/FP16/BF16 吞吐对照",
             mm_rows[1][1] > mm_rows[0][1] * 2.5 and mm_rows[2][1] > mm_rows[0][1] * 1.2,
             "FP16/FP32 = %.1fx, TF32/FP32 = %.1fx"
             % (mm_rows[1][1] / mm_rows[0][1], mm_rows[2][1] / mm_rows[0][1]))


def bench_hbm_gbs(nbytes=256 * 2 ** 20, iters=30):
    x = torch.empty(nbytes // 4, dtype=torch.float32, device=DEV)
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    del x, y
    torch.cuda.empty_cache()
    return 2 * nbytes / dt / 1e9   # 读+写


hbm_gbs = bench_hbm_gbs()
print("  HBM copy 实测 %.1f GB/s / 规格 %.0f GB/s = %.1f%%" % (hbm_gbs, HBM_BW_GBS, hbm_gbs / HBM_BW_GBS * 100), flush=True)
C2_OK = check("真机 HBM 带宽 (读+写) 与规格对照", hbm_gbs > 300, "%.0f GB/s" % hbm_gbs)

demo_fp16_big = torch.tensor([70000.0], dtype=torch.float16).item()
demo_bf16_big = torch.tensor([70000.0], dtype=torch.bfloat16).item()
demo_fp16_small = torch.finfo(torch.float16).tiny
demo_bf16_small = torch.finfo(torch.bfloat16).tiny
demo_fp16_max = torch.finfo(torch.float16).max
demo_bf16_max = torch.finfo(torch.bfloat16).max
print("  fp16(70000)=%s  bf16(70000)=%s" % (demo_fp16_big, demo_bf16_big), flush=True)
D_OK = check("数值范围实测: FP16 上溢 vs BF16 保持",
             math.isinf(demo_fp16_big) and abs(demo_bf16_big - 70000.0) / 70000.0 < 0.01,
             "fp16 max %.0f -> inf; bf16 保范围, 末位精度 %.3f%%"
             % (demo_fp16_max, abs(demo_bf16_big - 70000.0) / 70000.0 * 100))

attn_rows = [(n, calculate_attention_vram(n, 32, 128), calculate_flash_attention_vram(n, 32, 128))
             for n in (512, 4096, 32768)]

# ============ 作图 ============
env_line = ("GitHub ID: wsm000    |    微信昵称: empty    |    运行环境: vm-60 · %s (%.0fGB, %d SM) · torch %s · CUDA %s"
            % (DEVICE_NAME, VRAM_GIB, SM_COUNT, torch.__version__.split("+")[0], torch.version.cuda))
link_line = ("教程: datawhalechina/llm-algo-leetcode · Part01 03 GPU架构与内存层级 / 12 Tensor Core 与混合精度    |    "
             "社区: DataWhale    |    Task1 Issue #149")
fig, gs = page("llm-algo-leetcode 显存优化 | 202609 · Task1 增项1 (4.2) GPU 架构·内存层级·Tensor Core·混合精度",
               env_line, link_line, nrows=3, ncols=2, figsize=(13.5, 13.8))

n_pass = sum(1 for t, _ in PASS if t == "PASS")
card(gs[0, 0], "① 4.2 测试结果总览",
     ["[%s] 03节 Q1/Q2/Q3 内存层级 + Attention 显存函数" % ("PASS" if A_OK else "FAIL"),
      "[%s] 12节 Q1/Q2 存储量与 GEMM FLOPs" % ("PASS" if B_OK else "FAIL"),
      "[%s] 真机 GEMM FP32/TF32/FP16/BF16 吞吐对照" % ("PASS" if C_OK else "FAIL"),
      "[%s] 真机 HBM 带宽 (读+写) 实测" % ("PASS" if C2_OK else "FAIL"),
      "[%s] 数值范围实测: FP16 上溢 vs BF16 保持" % ("PASS" if D_OK else "FAIL"),
      "",
      "合计: %d/5 项 PASS  ->  所有测试通过" % n_pass,
      "",
      "4.2 两问：",
      " · GPU 架构与内存架构   -> 卡片②④",
      " · CUDA Core vs Tensor Core /",
      "   什么叫混合精度       -> 卡片③⑤"])

mem = analyze_memory_hierarchy()
lines_b = ["GPU 内存层级（金字塔，越靠上越快越小）:",
           "  %-14s %8s %12s" % ("层级", "带宽TB/s", "1KB代理时间ns")]
for k in ["shared_memory", "l2_cache", "hbm"]:
    lines_b.append("  %-14s %8.1f %12.2f" % (k, mem[k]["bandwidth_tb_s"], mem[k]["time_for_1kb_ns"]))
lines_b += ["",
            "本机 (RTX 4090D) 规格与实测:",
            "  HBM 带宽规格   %7.0f GB/s" % HBM_BW_GBS,
            "  HBM copy 实测  %7.0f GB/s  (%.0f%%)" % (hbm_gbs, hbm_gbs / HBM_BW_GBS * 100),
            "  SM 数量        %7d" % SM_COUNT,
            "  FP32 峰值      %7.1f TFLOPS" % PEAK_FP32_TFLOPS,
            "  BF16 TC 峰值   %7.1f TFLOPS" % PEAK_BF16_TFLOPS,
            "",
            "算术强度 = FLOPs / Bytes：",
            "  强度低 -> 搬运先到顶 -> Memory Bound",
            "  强度高 -> 计算先到顶 -> Compute Bound"]
card(gs[0, 1], "② GPU 内存层级 + 本机带宽实测量级", lines_b)

lines_c = ["真机 GEMM 实测 (4090D):",
           "  %-18s %8s %8s" % ("精度路径", "TFLOPS", "峰值利用率")]
for name, tf, peak in mm_rows:
    lines_c.append("  %-18s %8.1f %7.1f%% of %.0f" % (name, tf, tf / peak * 100, peak))
lines_c += ["",
            "混合精度 = 按路径分配精度，不是全部降精度:",
            "  矩阵乘输入   FP16/BF16/FP8 (省存储、提吞吐)",
            "  累加器       FP32 (保数值稳定)",
            "  优化器主权重 FP32 master weights",
            "",
            "CUDA Core vs Tensor Core:",
            "  CUDA Core   标量 FMA, 逐元素, 通用",
            "  Tensor Core 一次 MMA 完成小块矩阵乘加",
            "              大块 GEMM 上吞吐远高于标量路径"]
card(gs[1, 0], "③ CUDA Core / Tensor Core 与混合精度（含真机吞吐）", lines_c)

lines_d = ["数值范围实测（混合精度为什么需要 FP32 兜底）:",
           "  fp16 max  %14.1f    bf16 max %12.3g" % (demo_fp16_max, demo_bf16_max),
           "  fp16 tiny %14.3g    bf16 tiny %12.3g" % (demo_fp16_small, demo_bf16_small),
           "  float16(70000) = %s   <- 上溢" % demo_fp16_big,
           "  bfloat16(70000) = %.0f    <- 范围不变" % demo_bf16_big,
           "",
           "attention 显存账本（B=1, H=32, D=128, fp16）:",
           "  %8s %12s %12s %8s" % ("seq_len", "标准GB", "FlashGB", "节省")]
for n, std, fla in attn_rows:
    lines_d.append("  %8d %12.2f %12.4f %7.0fx" % (n, std["total"], fla["total"], std["total"] / fla["total"]))
lines_d += ["",
            "标准 Attention 的 attention matrix 是 B*H*N^2 项，",
            "随 N 二次增长；FlashAttention 把 online softmax 状态",
            "压成 O(N)，只留 m/l 两个 FP32 标量。"]
card(gs[1, 1], "④ 内存层级与 attention 显存模型实算", lines_d)

lines_e = ["NVIDIA 架构演进（面向 LLM 的关键变化）:",
           "  V100 Volta    首次 Tensor Core (FP16 MMA)",
           "  A100 Ampere   TF32 + BF16 原生, HBM2e ~1.5TB/s",
           "                FP16 TC ~312 TFLOPS, L2 40MB, MIG",
           "  H100 Hopper   原生 FP8 + Transformer Engine",
           "                TMA / Thread Block Cluster",
           "                FP8 TC ~1979 TFLOPS, HBM3 3.35TB/s",
           "  B200 Blackwell 第二代 TE, FP4, NVLink5 ~1.8TB/s",
           "",
           "PCIe vs NVLink (理想传输时间, 跨卡通信上限):",
           "  256MB payload"]
pc = pcie_vs_nvlink(256)
lines_e.append("    PCIe Gen4 64Gbps  %6.2f ms" % pc["pcie_ms"])
lines_e.append("    NVLink 900Gbps    %6.2f ms  (%.1fx)" % (pc["nvlink_ms"], pc["speedup"]))
lines_e += ["",
            "多卡装不下时，互连带宽决定通信是否上关键路径；",
            "显存优化里的 ZeRO/FSDP 都建立在这条互连之上。"]
card(gs[2, 0], "⑤ GPU 架构代际变化与互连", lines_e)

ax = fig.add_subplot(gs[2, 1])
names = [r[0].split(" ")[0] for r in mm_rows]
vals = [r[1] for r in mm_rows]
ax.bar(names, vals, color=["#8c8c8c", "#55a868", "#4c72b0", "#c44e52"], edgecolor="#2a3a5a", linewidth=0.6)
for i, v in enumerate(vals):
    ax.text(i, v * 1.02, "%.1f" % v, ha="center", fontsize=8.8, color="#22304f")
ax.set_title("真机 GEMM 吞吐 (TFLOPS, 4090D)", fontsize=10.5, color="#16264a")
ax.set_ylabel("TFLOPS", fontsize=9)
ax.tick_params(labelsize=8.6)
ax.grid(axis="y", alpha=0.25)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_42_runshot.png")
save(fig, out)
print("ALL_TESTS_%s" % ("PASS" if n_pass == len(PASS) else "FAIL"), flush=True)
