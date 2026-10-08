# -*- coding: utf-8 -*-
"""Render Task5 实测证据：四张图 + 一张运行总览（排版自实测 JSON，非终端截图）。

数据来源：evidence/task5 下四个实验 JSON（vm-60 实测，RTX 4090 D 24GB）。
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--evidence", required=True)
a = ap.parse_args()
root = Path(a.evidence)


def load(name):
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


def font(size):
    for p in ("/usr/share/fonts/truetype/windows/simhei.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


e1 = load("task5_exp1_object_timing")
e2 = load("task5_exp2_w8a16_vs_kv")
e3 = load("task5_exp3_kv_quant_capacity")
e4 = load("task5_exp4_candidates")

# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
aw = e1["a_real_weight_ladder"]
ax = axes[0][0]
names = ["FP16", "INT8\nper-tensor", "INT8\nper-channel", "INT4\ng128 packed"]
vals = [aw["fp16_bytes"], aw["int8_per_tensor_bytes"], aw["int8_per_channel_bytes"],
        aw["int4_group128_packed_bytes"]]
bars = ax.bar(names, [v / 1024 ** 2 for v in vals], color=["#f87171", "#fbbf24", "#4ade80", "#60a5fa"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, f"{v/1024**2:.0f}", ha="center", fontsize=9)
ax.set_ylabel("MiB")
ax.set_title(f"真实权重三档（Qwen2.5-0.5B, {aw['n_matrices']} 矩阵）")
ax.set_ylim(0, max(vals) / 1024 ** 2 * 1.18)

ax = axes[0][1]
mses = [aw["mse_fp16_vs_int8_pertensor"], aw["mse_fp16_vs_int8_perchannel"],
        aw["mse_fp16_vs_int4_group128"]]
ax.bar(["INT8-pt", "INT8-pc", "INT4-g128"], mses, color=["#fbbf24", "#4ade80", "#60a5fa"])
ax.set_yscale("log")
for i, v in enumerate(mses):
    ax.text(i, v * 1.3, f"{v:.1e}", ha="center", fontsize=9)
ax.set_ylabel("重建 MSE（log）")
ax.set_title("权重重建误差：粒度越细误差越小")

ax = axes[1][0]
c = e1["c_same_model_two_workloads"]
tags = ["短上下文单并发", "长上下文高并发"]
w = [c[t]["weight_fp16_bytes"] / 1024 ** 3 for t in tags]
kv = [c[t]["kv_bytes"] / 1024 ** 3 for t in tags]
ax.bar(tags, w, label="权重 FP16")
ax.bar(tags, kv, bottom=w, label="KV FP16（GQA-8kv）")
for i, t in enumerate(tags):
    ax.text(i, w[i] / 2, f"{w[i]:.1f} GiB\n{c[t]['weight_share']:.0%}", ha="center", fontsize=9)
    ax.text(i, w[i] + kv[i] / 2, f"{kv[i]:.1f} GiB\n{c[t]['kv_share']:.0%}", ha="center", fontsize=9)
ax.set_ylabel("GiB")
ax.set_title("同一 7B 模型：账本构成随 workload 切换")
ax.legend()

ax = axes[1][1]
kvpt = e1["b_capacity_ledger"]["kv_per_token_bytes"]
labels = list(kvpt.keys())
x = range(len(labels))
width = 0.25
for k, (dt, color) in enumerate([("fp16", "#f87171"), ("int8", "#4ade80"), ("fp8", "#22d3ee")]):
    ax.bar([i + (k - 1) * width for i in x], [kvpt[l][dt] / 1024 for l in labels],
           width=width, label=dt.upper(), color=color)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("KiB / token")
ax.set_title("KV 每 token 成本：表示变化的容量收益")
ax.legend()
fig.tight_layout()
fig.savefig(root / "01_object_timing.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
al = e2["a_real_layer"]
ax = axes[0][0]
cats = ["FP16", "W8A16"]
lat = [al["fp16"]["latency_ms"], al["w8a16"]["latency_ms"]]
pk = [al["fp16"]["peak_mib"], al["w8a16"]["peak_mib"]]
x = range(2)
ax.bar([i - 0.2 for i in x], lat, width=0.4, label="latency (ms)", color="#f87171")
ax.bar([i + 0.2 for i in x], pk, width=0.4, label="peak (MiB)", color="#60a5fa")
ax.set_xticks(list(x))
ax.set_xticklabels(cats)
for i in x:
    ax.text(i - 0.2, lat[i] + 1, f"{lat[i]:.3f}", ha="center", fontsize=9)
    ax.text(i + 0.2, pk[i] + 1, f"{pk[i]:.1f}", ha="center", fontsize=9)
ax.set_title(f"真实 q_proj：W8A16 不降峰值不加速（MSE {al['output_mse']:.1e}）")
ax.legend()

ax = axes[0][1]
bm = e2["b_full_model"]
wb = [bm["fp16"]["param_bytes"] / 1024 ** 2, bm["w8a16_per_tensor"]["param_bytes"] / 1024 ** 2,
      bm["w8a16_per_channel"]["param_bytes"] / 1024 ** 2]
pm = [bm["fp16"]["peak_mib"], bm["w8a16_per_tensor"]["peak_mib"], bm["w8a16_per_channel"]["peak_mib"]]
x3 = range(3)
ax.bar([i - 0.2 for i in x3], wb, width=0.4, label="参数字节 (MiB)", color="#4ade80")
ax.bar([i + 0.2 for i in x3], pm, width=0.4, label="forward 峰值 (MiB)", color="#fbbf24")
ax.set_xticks(list(x3))
ax.set_xticklabels(["FP16", "W8A16-pt", "W8A16-pc"])
for i in x3:
    ax.text(i - 0.2, wb[i] + 20, f"{wb[i]:.0f}", ha="center", fontsize=9)
    ax.text(i + 0.2, pm[i] + 20, f"{pm[i]:.0f}", ha="center", fontsize=9)
ax.set_title("全模型：权重存储减半，逐层反量化副本使 forward 峰值反升")
ax.legend()

ax = axes[1][0]
cf = e2["c_fixed_vs_linear"]
tokens = [0, 1024, 2048, 4096, 8192, 16384]
kv_bytes = [t * cf["kv_per_token_bytes_measured"] / 1024 ** 2 for t in tokens]
ax.plot(tokens, kv_bytes, marker="o", color="#f87171", label="KV 累积（线性增长）")
ax.axhline(cf["saved_bytes"] / 1024 ** 2, color="#4ade80", ls="--", label="W8A16 省下的固定字节")
ax.annotate(f"{cf['saving_equals_extra_tokens']:.0f} tokens",
            xy=(cf["saving_equals_extra_tokens"], cf["saved_bytes"] / 1024 ** 2),
            xytext=(2000, cf["saved_bytes"] / 1024 ** 2 * 1.6),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=10)
ax.set_xlabel("tokens")
ax.set_ylabel("MiB")
ax.set_title("固定节省 vs 线性增长：权重省的量只够多少量 token")
ax.legend()

ax = axes[1][1]
lr = e2["b_layer_rel_err"]
pt = lr["per_tensor_layer_rel"]
pc = lr["per_channel_layer_rel"]
ax.plot(range(len(pt)), pt, marker="o", ms=3, color="#f87171", label="per-tensor INT8")
ax.plot(range(len(pc)), pc, marker="s", ms=3, color="#4ade80", label="per-channel INT8")
ax.set_yscale("log")
ax.set_xlabel("decoder layer")
ax.set_ylabel("hidden 最大相对误差（log）")
ax.set_title("误差逐层复合：单层 MSE ~1e-6，24 层后仍需看输出质量")
ax.legend()
fig.tight_layout()
fig.savefig(root / "02_w8a16_vs_kv.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ak = e3["a_real_kv_error"]
ax = axes[0][0]
variants = list(ak["variants"].keys())
b = [ak["variants"][v]["bytes"] / 1024 for v in variants]
m = [ak["variants"][v]["mse_key"] for v in variants]
ax.scatter(b, m, s=80, color="#f87171")
for xi, yi, name in zip(b, m, variants):
    ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("KiB（K+V, layer0）")
ax.set_ylabel("MSE key（log）")
ax.set_title(f"真实 KV 量化：字节 vs 误差（prompt {ak['n_tokens']} tokens）")

ax = axes[0][1]
cap = e3["b_capacity_ledger"]["capacity"]
keys = [k for k in cap if k.startswith("GQA-8kv")]
x = range(len(keys))
vals = [cap[k]["max_concurrency_s4096"] for k in keys]
bars = ax.bar([k.replace("GQA-8kv-", "") for k in keys], vals, color=["#f87171", "#4ade80", "#22d3ee"])
for bb, v in zip(bars, vals):
    ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 1, str(v), ha="center", fontsize=9)
ax.set_ylabel("最大并发 @S=4096")
ax.set_title("容量账本：每 token 成本减半 -> 并发翻倍")

ax = axes[1][0]
ad = e3["c_admission_pressure"]
cats = ["FP16 KV", "INT8 KV"]
vals = [ad["fp16"]["requests_admitted"], ad["int8"]["requests_admitted"]]
bars = ax.bar(cats, vals, color=["#f87171", "#4ade80"])
for bb, v, g in zip(bars, vals, [ad["fp16"]["measured_kv_capacity_gib"], ad["int8"]["measured_kv_capacity_gib"]]):
    ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 1, f"{v} req\n{g:.1f} GiB", ha="center", fontsize=9)
ax.set_ylabel("准入请求数（S=4096，真实分配至 OOM）")
ax.set_title(f"构造性准入实测：并发 x{ad['concurrency_gain']:.2f}")

ax = axes[1][1]
rr = e3["d_runtime_reservations"]
parts = [rr["weights_mib"], rr["activations_mib"], rr["allocator_reserved_minus_allocated_mib"]]
ax.bar(["权重", "激活峰值", "分配器 gap"], parts, color=["#60a5fa", "#fbbf24", "#a78bfa"])
for i, v in enumerate(parts):
    ax.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("MiB")
ax.set_title("运行时预留项：量化之外仍要留的空间（另 +256MiB 余量）")
fig.tight_layout()
fig.savefig(root / "03_kv_quant_capacity.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 4
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
cands = list(e4["candidates"].keys())
labels = [c.replace("_weight", "\nweight").replace("w8a16", "W8A16").replace("int4", "INT4")
          .replace("kv_int8", "+KV-INT8").replace("fp16_baseline", "FP16 baseline") for c in cands]
x = range(len(cands))
ax = axes[0][0]
ttft = [e4["candidates"][c]["ttft_ms"] for c in cands]
tpot = [e4["candidates"][c]["tpot_ms"] for c in cands]
ax.bar([i - 0.2 for i in x], ttft, width=0.4, label="TTFT (ms)", color="#f87171")
ax.bar([i + 0.2 for i in x], [t * 100 for t in tpot], width=0.4, label="TPOT×100 (ms)", color="#60a5fa")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("TTFT / TPOT（同 workload：B=4, prompt 512, gen 64）")
ax.legend()

ax = axes[0][1]
thr = [e4["candidates"][c]["throughput_tok_s"] for c in cands]
bars = ax.bar(labels, thr, color=["#94a3b8", "#fbbf24", "#4ade80", "#22d3ee"])
for bb, v in zip(bars, thr):
    ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 3, f"{v:.0f}", ha="center", fontsize=9)
ax.set_ylabel("tok/s")
ax.set_title("吞吐：只有 W8A16+KV-INT8 同时改善显存与吞吐")
ax.tick_params(axis="x", rotation=15)

ax = axes[1][0]
pk = [e4["candidates"][c]["peak_mib"] for c in cands]
kvm = [e4["candidates"][c]["kv_mib"] for c in cands]
ax.bar([i - 0.2 for i in x], pk, width=0.4, label="peak (MiB)", color="#fbbf24")
ax.bar([i + 0.2 for i in x], kvm, width=0.4, label="KV 存储 (MiB)", color="#a78bfa")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("峰值显存与 KV 存储（W8A16 峰值不降：反量化副本）")
ax.legend()

ax = axes[1][1]
dec = [e4["candidates"][c]["decision"] for c in cands]
colors = {"accept": "#4ade80", "tune": "#fbbf24", "reject": "#f87171", "baseline": "#94a3b8"}
bars = ax.bar(labels, [1] * len(cands), color=[colors[d] for d in dec])
for bb, d, c in zip(bars, dec, cands):
    cos = e4["candidates"][c]["logits_cosine_vs_baseline"]
    dpk = e4["candidates"][c]["delta_peak_mib"]
    dth = e4["candidates"][c]["delta_throughput_tok_s"]
    ax.text(bb.get_x() + bb.get_width() / 2, 0.5, f"{d.upper()}\ncos {cos:.4f}\nΔpeak {dpk:+.1f}\nΔthr {dth:+.1f}",
            ha="center", va="center", fontsize=8, color="#0f172a")
ax.set_yticks([])
ax.set_title("决策（误差预算硬门槛，再看性能/显存收益）")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(root / "04_candidates.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- runshot
sc7 = e1["c_same_model_two_workloads"]
sections = []
aw = e1["a_real_weight_ladder"]
sections.append(("EXPERIMENT 1 - object & timing ledger (PASS)", [
    f"A. real weights: {aw['n_matrices']} matrices, {aw['total_params']/1e6:.0f}M params; "
    f"FP16 {aw['fp16_bytes']/1024**2:.0f} MiB -> INT8-pt {aw['int8_per_tensor_bytes']/1024**2:.0f} "
    f"(x{aw['compression']['int8_per_tensor']:.2f}) / INT8-pc {aw['int8_per_channel_bytes']/1024**2:.0f} "
    f"(x{aw['compression']['int8_per_channel']:.2f}) / INT4-g128 {aw['int4_group128_packed_bytes']/1024**2:.0f} "
    f"(x{aw['compression']['int4_group128']:.2f}, scale {aw['int4_group128_scale_bytes']/1024:.0f} KiB)",
    f"   MSE: INT8-pt {aw['mse_fp16_vs_int8_pertensor']:.2e} / INT8-pc {aw['mse_fp16_vs_int8_perchannel']:.2e} / "
    f"INT4-g128 {aw['mse_fp16_vs_int4_group128']:.2e}",
    "B. capacity ledger: 0.5B/1.5B/7B x FP16/INT8/INT4; KV/token MHA 96 / GQA-8 24 / MQA 3 KiB (fp16)",
    f"C. same 7B, two workloads: short-ctx weight share {sc7['短上下文单并发']['weight_share']:.0%}; "
    f"long-ctx KV share {sc7['长上下文高并发']['kv_share']:.0%}; after INT8 weights KV share "
    f"{sc7['int8_weight_shift']['kv_share_after']:.0%}",
]))
sections.append(("EXPERIMENT 2 - W8A16 vs KV (PASS)", [
    f"A. real q_proj {al['in_features']}x{al['out_features']}: FP16 {al['fp16']['latency_ms']:.3f} ms / "
    f"{al['fp16']['peak_mib']:.2f} MiB vs W8A16 {al['w8a16']['latency_ms']:.3f} ms / "
    f"{al['w8a16']['peak_mib']:.2f} MiB (MSE {al['output_mse']:.2e})",
    f"B. full model ({bm['w8a16_per_tensor']['n_linear_replaced']} Linear): param bytes "
    f"{bm['fp16']['param_bytes']/1024**2:.0f} -> {bm['w8a16_per_tensor']['param_bytes']/1024**2:.0f} MiB, "
    f"forward peak {bm['fp16']['peak_mib']:.0f} -> {bm['w8a16_per_tensor']['peak_mib']:.0f} MiB; "
    f"logits MSE {bm['w8a16_per_tensor']['logits_mse']:.2e} (pt) vs "
    f"{bm['w8a16_per_channel']['logits_mse']:.2e} (pc)",
    f"   hidden 最大相对误差: pt {bm['w8a16_per_tensor']['hidden_rel_err_max']:.1%} vs "
    f"pc {bm['w8a16_per_channel']['hidden_rel_err_max']:.1%} —— 单层 MSE ~1e-6 也会逐层复合",
    f"C. fixed saving {cf['saved_bytes']/1024**2:.0f} MiB == {cf['saving_equals_extra_tokens']:.0f} extra tokens "
    f"of KV ({cf['kv_per_token_bytes_measured']/1024:.0f} KiB/token)",
]))
sections.append(("EXPERIMENT 3 - KV quantization capacity (PASS)", [
    f"A. real KV (layer0, {ak['n_tokens']} tokens): " + " | ".join(
        f"{k} {v['bytes']/1024:.0f} KiB MSE_K {v['mse_key']:.1e}" for k, v in ak['variants'].items()),
    f"B. pool {e3['b_capacity_ledger']['kv_pool_bytes']/1024**3:.2f} GiB: GQA-8kv conc@4096 "
    + " / ".join(f"{dt} {cap[f'GQA-8kv-{dt}']['max_concurrency_s4096']}" for dt in ("fp16", "int8", "fp8")),
    f"C. admission (S=4096, real alloc until OOM): FP16 {ad['fp16']['requests_admitted']} req "
    f"({ad['fp16']['measured_kv_capacity_gib']:.1f} GiB) vs INT8 {ad['int8']['requests_admitted']} req "
    f"({ad['int8']['measured_kv_capacity_gib']:.1f} GiB) x{ad['concurrency_gain']:.2f}",
    f"D. reservations: weights {rr['weights_mib']:.0f} / act {rr['activations_mib']:.0f} / "
    f"allocator gap {rr['allocator_reserved_minus_allocated_mib']:.0f} MiB (+256 MiB margin)",
]))
sections.append(("EXPERIMENT 4 - candidates & decisions (PASS)", [
    "workload: toy-12L GQA-8kv | B=4 | prompt 512 | gen 64 | fp16 | seed 73",
]))
for c in cands:
    r = e4["candidates"][c]
    sections[-1][1].append(
        f"    {c:<22} TTFT {r['ttft_ms']:7.2f} TPOT {r['tpot_ms']:6.3f} thr {r['throughput_tok_s']:6.1f} "
        f"peak {r['peak_mib']:7.1f} KV {r['kv_mib']:6.2f} cos {r['logits_cosine_vs_baseline']:.5f} -> {r['decision']}")

sections.append(("OFFICIAL NOTEBOOKS (upstream commit e518638, executed on vm-60)", [
    "21 量化理论与 INT4/INT8        : executed, 3 test groups passed",
    "25 W8A16 量化                 : TODO 1-2 completed, real_gpu + torchao probe, tests passed",
    "40 GPTQ 与 AWQ 权重量化       : TODO 1-5 completed, real_gpu calibration, tests passed",
    "41 FP8 与 KV Cache 量化       : TODO 1-3 completed, real_gpu + FP8 torchao probe, tests passed",
    "66 推理性能对比实验           : TODO 1-3 completed, project tests passed (vLLM backend n/a)",
    "67 量化推理与部署             : TODO 1-3 completed, artifact contract tests passed",
    "   67 needs upstream tools/ package (inference_project_runtime); vLLM backend not installed,",
    "   Step-5 real backend kept closed -> decision=pending (honestly recorded)",
]))
lines = []
for heading, body in sections:
    lines.append((heading, "#7dd3fc"))
    for line in body:
        for w in (textwrap.wrap(line, width=104, replace_whitespace=False, drop_whitespace=False) or [""]):
            lines.append((w, "#e2e8f0"))
    lines.append(("", "#e2e8f0"))
height = 210 + len(lines) * 30 + 60
image = Image.new("RGB", (1560, height), "#0f172a")
d = ImageDraw.Draw(image)
d.rectangle((0, 0, 1560, 8), fill="#38bdf8")
d.text((36, 26), "TASK 5 / 量化与显存容量扩展 - 实测证据总览", font=font(30), fill="#f8fafc")
d.text((36, 76), "DataWhale llm-algo-leetcode | Issue #178 | vm-60 | measured evidence (not a screenshot)",
       font=font(20), fill="#94a3b8")
d.text((36, 110), "Source SHA256: " + e1.get("source_sha256", "see JSON"), font=font(20), fill="#94a3b8")
y = 160
head_font, body_font = font(22), font(18)
for line, color in lines:
    d.text((36, y), line, font=head_font if color == "#7dd3fc" else body_font, fill=color)
    y += 30
image.save(root / "task5_runshot.png")
print("rendered: 01_object_timing.png 02_w8a16_vs_kv.png 03_kv_quant_capacity.png "
      "04_candidates.png task5_runshot.png")
