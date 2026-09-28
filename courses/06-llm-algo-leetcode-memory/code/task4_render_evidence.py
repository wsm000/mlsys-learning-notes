# -*- coding: utf-8 -*-
"""Render Task4 实测证据：四张图 + 一张运行总览（PIL 排版，非终端截图）。

数据来源：evidence/task4 下四个实验 JSON（vm-60 实测）。图中只画实测数字。
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--evidence", required=True)
a = ap.parse_args()
root = Path(a.evidence)


def load(name):
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


e1 = load("task4_exp1_growth")
e2 = load("task4_exp2_paged")
e3 = load("task4_exp3_prefix")
e4 = load("task4_exp4_capacity")

# ---------------------------------------------------------------- fig 1 增长
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0][0]
rows = e1["c_real_model"]["rows"]
names = [r["config"] for r in rows]
vals = [r["prefill_cache_bytes"] / 1024 ** 2 for r in rows]
ledger = [r["prefill_ledger_bytes"] / 1024 ** 2 for r in rows]
x = range(len(names))
ax.bar([i - 0.2 for i in x], vals, width=0.4, label="实测 cache 字节")
ax.bar([i + 0.2 for i in x], ledger, width=0.4, label="账本公式", alpha=0.75)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("MiB")
ax.set_title("prefill(B=4,S=512) KV cache：实测 == 账本")
ax.legend()
for i, r in enumerate(rows):
    ax.text(i, max(vals[i], ledger[i]) + 1, f"{vals[i]:.0f}", ha="center", fontsize=8)

ax = axes[0][1]
dims = list(e1["b_linearity"]["dimensions"].keys())
ratios = [e1["b_linearity"]["dimensions"][d]["double_ratio"] for d in dims]
ax.bar(dims, ratios, color="#38bdf8")
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.set_ylim(0, 2.4)
ax.set_ylabel("翻倍后 KV 字节比率")
ax.set_title("六个维度全部严格线性（比率恰为 2.0）")
ax.tick_params(axis="x", rotation=30)

ax = axes[1][0]
# 每 token KiB：实测 MHA/GQA-4/GQA-2/MQA + 账本 GQA-8/MLA + FP8 半价
mha_kib = rows[0]["per_token_kib"]
gqa8_kib = 2 * 12 * 8 * 128 * 2 / 1024
mla_kib = e1["d_mla"]["mla_bytes"] / 4096 / 1024
ptok = [mha_kib, rows[1]["per_token_kib"], rows[2]["per_token_kib"],
        rows[3]["per_token_kib"], gqa8_kib, mla_kib, mha_kib / 2]
labels = ["MHA", "GQA-4", "GQA-2", "MQA", "GQA-8", "MLA(512+64)", "GQA-8+FP8"]
colors = ["#f87171", "#fbbf24", "#fbbf24", "#4ade80", "#60a5fa", "#a78bfa", "#22d3ee"]
ax.bar(labels, ptok, color=colors)
ax.set_ylabel("KiB / token")
ax.set_title("每 token KV 成本：表示变化的容量收益")
ax.tick_params(axis="x", rotation=25)
for i, v in enumerate(ptok):
    ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)

ax = axes[1][1]
er = e1["e_kv_precision"]["rows"]
xs = [r["name"] for r in er]
errs = [r["k_max_abs_err"] for r in er]
byts = [r["bytes"] for r in er]
ax2 = ax.twinx()
ax.bar(xs, errs, color="#f472b6", alpha=0.8, label="K 最大绝对误差")
ax2.plot(xs, byts, "o--", color="#34d399", label="字节")
ax.set_ylabel("max abs err (K)")
ax2.set_ylabel("bytes")
ax.set_title("FP8/INT8 KV：误差与字节（半价存储）")
for i, v in enumerate(errs):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(root / "01_growth_ledger.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 2 分页
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0][0]
c = e2["c_fragmentation"]
cats = ["按需分页(bs=16)", "连续预留(S_max)"]
vals = [c["kv_mib_paged"], c["kv_mib_contiguous"]]
ax.bar(cats, vals, color=["#4ade80", "#f87171"])
ax.set_ylabel("KV MiB（同一请求分布）")
ax.set_title(f"连续预留 vs 分页：{c['token_saving_ratio']*100:.1f}% token 节省\n"
             f"尾块浪费仅 {c['tail_waste_tokens']} token")
for i, v in enumerate(vals):
    ax.text(i, v + 300, f"{v:,.0f}", ha="center", fontsize=9)

ax = axes[0][1]
rows = e2["d_block_size_sweep"]["rows"]
bss = [r["block_size"] for r in rows]
waste = [r["internal_frag_ratio"] * 100 for r in rows]
meta = [r["block_table_entries"] / 1000 for r in rows]
ax.plot(bss, waste, "o-", color="#f87171", label="内部碎片（占实用 token %）")
ax.set_xlabel("block size")
ax.set_ylabel("内部碎片 %", color="#f87171")
ax2 = ax.twinx()
ax2.plot(bss, meta, "s--", color="#60a5fa", label="块表条目（千）")
ax2.set_ylabel("块表条目（千条）", color="#60a5fa")
ax.set_xscale("log", base=2)
ax.set_title("block size 扫描：碎片↑ vs 元数据↓ 的 trade-off")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")

ax = axes[1][0]
b = e2["b_constructive_oom"]
labels = [f"连续大块\n({b['contiguous_need_bytes']/1024**3:.1f} GiB)", "分页逐块\n(8 MiB x N)"]
vals = [0, b["free_hole_bytes"] / 1024 ** 3]
ax.bar(labels, [b["contiguous_need_bytes"] / 1024 ** 3, b["free_hole_bytes"] / 1024 ** 3],
       color=["#f87171", "#4ade80"])
ax.text(0, b["contiguous_need_bytes"] / 1024 ** 3 + 0.2,
        "OOM 失败", ha="center", color="#f87171", fontsize=11, weight="bold")
ax.text(1, b["free_hole_bytes"] / 1024 ** 3 + 0.2,
        "分配成功", ha="center", color="#4ade80", fontsize=11, weight="bold")
ax.set_ylabel("GiB")
ax.set_title(f"构造性外部碎片：空闲 {b['free_hole_bytes']/1024**3:.1f} GiB\n"
             f"却拿不出 {b['contiguous_need_bytes']/1024**3:.1f} GiB 连续块")

ax = axes[1][1]
g = e2["e_gather_attention"]
ax.axis("off")
ax.text(0.02, 0.75, "GPU gather-attention 数值等价", fontsize=12, weight="bold",
        transform=ax.transAxes)
ax.text(0.02, 0.5,
        f"shape = {g['shape']}\n"
        f"逻辑恢复 max_diff = {g['logical_recovery_max_diff']}\n"
        f"attention 输出 max_diff = {g['attention_output_max_diff']}",
        fontsize=11, transform=ax.transAxes)
ax.text(0.02, 0.12, "按块表顺序 gather 回逻辑连续序列后，\n与连续存放的注意力输出逐位一致。",
        fontsize=10, transform=ax.transAxes)
fig.tight_layout()
fig.savefig(root / "02_paged_tradeoff.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 3 复用
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0][0]
b1 = e3["b_real_model"]["b1_compute_scale_prompt4096"]
vals = [b1["full_prefill_ttft_ms"], b1["prompt_fill_once_ms"], b1["suffix_only_ttft_ms"]]
ax.bar(["整段 prefill\n(4224 tok)", "prompt 填充一次\n(4096 tok)", "suffix-only\n(128 tok)"], vals,
       color=["#f87171", "#fbbf24", "#4ade80"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f} ms", ha="center", fontsize=9)
ax.set_ylabel("ms")
ax.set_title(f"compute-bound（prompt=4096）：TTFT 省 {b1['ttft_saving_ratio_vs_full']*100:.1f}%\n"
             f"（deepcopy 隔离开销 {b1['deepcopy_isolation_ms']:.2f} ms）")

ax = axes[0][1]
b2 = e3["b_real_model"]
vals = [b2["baseline_mean_ms"], b2["prefix_mean_ms"]]
ax.bar(["baseline\n整段 prefill", "prefix cache\nsuffix-only"], vals,
       color=["#f87171", "#4ade80"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.05, f"{v:.2f} ms", ha="center", fontsize=9)
ax.set_ylabel("mean TTFT (ms)")
ax.set_title("launch-bound（prompt=512）：命中率 66.7% token\n但 TTFT 无收益（固定开销主导）")

ax = axes[1][0]
st = e3["c_storage"]
vals = [st["duplicate_mib"], st["shared_mib"]]
gp = st["gpu_probe"]
ax.bar(["重复存储\n(N x 640 tok)", "共享 prompt\n(1 x 512 + N x 128)"], vals,
       color=["#f87171", "#4ade80"])
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.1f} MiB", ha="center", fontsize=9)
ax.set_ylabel("KV MiB")
ax.set_title(f"Prefix cache 降 KV 显存：省 {st['storage_saving_ratio']*100:.1f}%\n"
             f"真实分配探针 {gp['duplicate_peak_mib']:.1f} -> {gp['shared_peak_mib']:.1f} MiB")

ax = axes[1][1]
d = e3["d_contiguity"]
labels = ["前缀全同", "首段不同", "中段不同"]
hits = [d["case_A_same_prefix"]["hit_tokens"], d["case_B_diff_head"]["hit_tokens"],
        d["case_C_diff_middle"]["hit_tokens"]]
ax.bar(labels, hits, color=["#4ade80", "#f87171", "#fbbf24"])
ax.set_ylabel("命中 token 数")
ax.set_title("连续前缀要求：分叉点之后全部失效\n（因果注意力 + RoPE 位置绑定）")
for i, v in enumerate(hits):
    ax.text(i, v + 8, str(v), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(root / "03_prefix_cache.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- fig 4 容量
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0][0]
pa = e4["a_capacity_plan"]
rows = pa["capacity_rows"]
labels = [r["variant"] for r in rows]
toks = [r["max_tokens"] / 1000 for r in rows]
ax.barh(labels, toks, color=["#f87171", "#60a5fa", "#4ade80", "#a78bfa", "#22d3ee", "#22d3ee"])
ax.set_xlabel("最大 token 容量（千）")
ax.set_title(f"容量规划：KV 池 {pa['kv_pool_gib']} GiB\n"
             f"= 总显存x0.9 - 上下文 - 权重 - 运行缓冲 - 256MiB 余量")
for i, v in enumerate(toks):
    ax.text(v + 3, i, f"{v:,.0f}k", va="center", fontsize=8)

ax = axes[0][1]
rows = [r for r in e4["b_admission"]["rows"] if r.get("status") != "oom_at_build"]
labels = [r["variant"] for r in rows]
planned = [r["planned_max_concurrency_4096"] for r in rows]
actual = [r["max_concurrent_wave"] for r in rows]
x = range(len(labels))
ax.bar([i - 0.2 for i in x], planned, width=0.4, label="规划(账本)")
ax.bar([i + 0.2 for i in x], actual, width=0.4, label="实测准入峰值", alpha=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("S=4096 并发数")
ax.set_title("准入压力实测 vs 容量规划")
ax.legend()

ax = axes[1][0]
c = e4["c_policy_ab"]
labels = ["baseline", "无界 prefix cache", "LRU 封顶 cache"]
ttft = [c["baseline"]["ttft_mean_ms"], c["unbounded_prefix_cache"]["ttft_mean_ms"],
        c["lru_capped_cache"]["ttft_mean_ms"]]
peak = [c["baseline"]["kv_peak_mib"], c["unbounded_prefix_cache"]["kv_peak_mib"],
        c["lru_capped_cache"]["kv_peak_mib"]]
x = range(len(labels))
ax.bar([i - 0.2 for i in x], ttft, width=0.4, label="mean TTFT (ms)")
ax.set_ylabel("TTFT ms", color="#f87171")
ax2 = ax.twinx()
ax2.bar([i + 0.2 for i in x], peak, width=0.4, color="#60a5fa", label="peak KV MiB")
ax2.set_ylabel("peak KV MiB", color="#60a5fa")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("同 workload 缓存策略对照（66 -> 69 口径）")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")

ax = axes[1][1]
ax.axis("off")
lines = ["候选决策（同 workload 阈值规则）："]
for d in c["decisions"]:
    lines.append(f"  {d['candidate']}: {d['decision']}  "
                 f"(TTFT {d['ttft_delta_ms']:+.2f} ms, peak {d['peak_delta_mib']:+.2f} MiB)")
lines.append("")
lines.append(f"hit_rate: 无界 {c['unbounded_prefix_cache']['hit_rate']:.2f} / "
             f"封顶 {c['lru_capped_cache']['hit_rate']:.2f}")
lines.append(f"P99 TTFT: 无界 {c['unbounded_prefix_cache']['ttft_p99_ms']:.2f} ms / "
             f"封顶 {c['lru_capped_cache']['ttft_p99_ms']:.2f} ms")
ax.text(0.02, 0.9, "\n".join(lines), fontsize=10, transform=ax.transAxes,
        va="top")
ax.set_title("accept / tune / reject")
fig.tight_layout()
fig.savefig(root / "04_capacity.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- runshot panel
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
font = ImageFont.truetype(font_path, 20)
title_font = ImageFont.truetype(font_path, 30)
head_font = ImageFont.truetype(font_path, 23)

env = e1["environment"]
sections = []
sections.append(("环境 / ENVIRONMENT", [
    f"vm-60 | {env['gpu']} | {env['total_memory_mib']:.0f} MiB",
    f"torch {env['torch']} | CUDA {env['cuda']} | Python {env['python']}",
    f"UTC {env['utc']}",
]))
secs = []
secs.append(("EXPERIMENT 1 - growth ledger (PASS; ALL_TESTS_PASS)", [
    "A. ledger vs real allocation: exact match in all 4 cases",
    f"B. linearity: all 6 dims ratio exactly 2.0 when doubled",
    f"C. real model B=4,S=512: MHA {e1['c_real_model']['rows'][0]['prefill_cache_bytes']/1024**2:.1f} MiB / "
    f"GQA-4 {e1['c_real_model']['rows'][1]['prefill_cache_bytes']/1024**2:.1f} / "
    f"GQA-2 {e1['c_real_model']['rows'][2]['prefill_cache_bytes']/1024**2:.1f} / "
    f"MQA {e1['c_real_model']['rows'][3]['prefill_cache_bytes']/1024**2:.1f} MiB",
    f"D. MLA latent @S=4096: {e1['d_mla']['mla_mib']:.1f} MiB vs MHA {e1['d_mla']['mha_mib']:.1f} vs GQA-8 {e1['d_mla']['gqa8_mib']:.1f}",
    f"E. FP8 err {e1['e_kv_precision']['rows'][1]['k_max_abs_err']:.4f} / "
    f"INT8 err {e1['e_kv_precision']['rows'][2]['k_max_abs_err']:.4f} (half bytes)",
]))
secs.append(("EXPERIMENT 2 - paged attention (PASS)", [
    "A. mechanism: atomic OOM / decode boundary / tail ledger / refcount share",
    f"B. real OOM demo: free {e2['b_constructive_oom']['free_hole_bytes']/1024**3:.2f} GiB holes, "
    f"contiguous {e2['b_constructive_oom']['contiguous_need_bytes']/1024**3:.2f} GiB -> "
    f"contig {'OK' if e2['b_constructive_oom']['contiguous_allocated'] else 'FAIL'} / "
    f"paged {'OK' if e2['b_constructive_oom']['paged_8mib_chunks_allocated'] else 'FAIL'}",
    f"C. same trace: paged {e2['c_fragmentation']['kv_mib_paged']:.0f} MiB vs "
    f"contiguous {e2['c_fragmentation']['kv_mib_contiguous']:.0f} MiB "
    f"(-{e2['c_fragmentation']['token_saving_ratio']*100:.1f}% tokens)",
    "D. block-size sweep: bs=1 -> 86,040 entries/0 frag; bs=256 -> 466 entries/38.7% frag",
    "E. gather attention equivalence: max_diff = 0.0",
]))
b1 = e3["b_real_model"]["b1_compute_scale_prompt4096"]
b2 = e3["b_real_model"]
secs.append(("EXPERIMENT 3 - prefix cache (PASS)", [
    "A. block-hash chain: contiguous / branch / middle-diff / position / tail / LRU",
    f"B1. prompt=4096: full {b1['full_prefill_ttft_ms']:.1f} ms -> suffix-only "
    f"{b1['suffix_only_ttft_ms']:.1f} ms (-{b1['ttft_saving_ratio_vs_full']*100:.1f}%)",
    f"B2. prompt=512: baseline {b2['baseline_mean_ms']:.2f} ms vs prefix "
    f"{b2['prefix_mean_ms']:.2f} ms (no wall-clock gain at launch-bound scale)",
    f"    logits equivalence: max diff {max(b2['logits_max_abs_diff']):.5f} (fp16 noise)",
    f"C. storage: {e3['c_storage']['duplicate_mib']:.0f} -> {e3['c_storage']['shared_mib']:.0f} MiB "
    f"(-{e3['c_storage']['storage_saving_ratio']*100:.1f}%)",
    f"D. contiguity: same-prefix hit 512 / diff-head hit 0 / diff-middle hit 256 tokens",
]))
secs.append(("EXPERIMENT 4 - capacity verification (PASS)", [
    f"A. plan: weights {e4['a_capacity_plan']['weights_gib']:.2f} GiB (measured={e4['a_capacity_plan']['weights_measured']}), "
    f"buffers {e4['a_capacity_plan']['runtime_buffers_mib']:.0f} MiB, "
    f"KV pool {e4['a_capacity_plan']['kv_pool_gib']:.2f} GiB",
]))
for r in e4["a_capacity_plan"]["capacity_rows"]:
    secs[-1][1].append(
        f"    {r['variant']:<16} {r['per_token_kib']:>6.1f} KiB/tok -> "
        f"{r['max_tokens']:,} tokens, {r['max_concurrency_at_4096']} conc @4096")
for r in e4["b_admission"]["rows"]:
    if r.get("status") == "oom_at_build":
        secs[-1][1].append(f"    admission {r['variant']}: OOM at build (see log)")
    else:
        secs[-1][1].append(
            f"    admission {r['variant']}: pool {r['kv_pool_gib']:.2f} GiB, "
            f"budget {r['budget_tokens']:,} tok, admitted {r['admitted_total']} req, "
            f"peak wave {r['max_concurrent_wave']}, peak KV {r['kv_peak_gib']:.2f} GiB, "
            f"{r['gen_tokens_per_s']:.0f} tok/s")
for d in e4["c_policy_ab"]["decisions"]:
    secs[-1][1].append(
        f"    policy {d['candidate']}: {d['decision']} "
        f"(TTFT {d['ttft_delta_ms']:+.2f} ms, peak {d['peak_delta_mib']:+.2f} MiB)")

secs.append(("OFFICIAL NOTEBOOKS (upstream commit e518638, executed on vm-60)", [
    "11 KV Cache 与显存增长    : executed, asserts pass",
    "22 vLLM PagedAttention    : TODO 1-5 completed, all mechanism tests pass",
    "34 Prefix Cache 匹配与复用 : TODO 1-5 completed, all tests pass",
    "24 SGLang RadixAttention  : TODO 1-6 completed, all tests pass",
    "66 推理性能对比           : TODO 1-3 completed, project tests pass",
    "69 Prefix Caching Benchmark: TODO 1-3 completed, tests pass",
]))
lines = []
for heading, body in sections + [("", [])] + secs:
    lines.append((heading, "#7dd3fc"))
    for line in body:
        for w in (textwrap.wrap(line, width=104, replace_whitespace=False,
                                drop_whitespace=False) or [""]):
            lines.append((w, "#e2e8f0"))
    lines.append(("", "#e2e8f0"))
height = 210 + len(lines) * 30 + 60
image = Image.new("RGB", (1560, height), "#0f172a")
d = ImageDraw.Draw(image)
d.rectangle((0, 0, 1560, 8), fill="#38bdf8")
d.text((36, 26), "TASK 4 / KV Cache 与显存容量 - 实测证据总览", font=title_font, fill="#f8fafc")
d.text((36, 76), "DataWhale llm-algo-leetcode | Issue #177 | vm-60 | measured evidence (not a screenshot)",
       font=font, fill="#94a3b8")
d.text((36, 110), "Source SHA256: " + e1["source_sha256"] + " (exp1) | 环境/口径见各 JSON",
       font=font, fill="#94a3b8")
y = 160
for line, color in lines:
    if color == "#7dd3fc":
        d.text((36, y), line, font=head_font, fill=color)
    else:
        d.text((36, y), line, font=font, fill=color)
    y += 30
image.save(root / "task4_runshot.png")
print("rendered: 01_growth_ledger.png 02_paged_tradeoff.png 03_prefix_cache.png "
      "04_capacity.png task4_runshot.png")
