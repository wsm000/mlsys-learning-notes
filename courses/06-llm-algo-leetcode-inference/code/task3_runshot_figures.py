# -*- coding: utf-8 -*-
"""生成 Task3 的运行截图：真实 cell 源码 + 真实输出 + 自加实验数据。"""
import json
import os

import nbformat

import task3_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "..", "evidence", "task3")
ENV = ("GitHub ID: wsm000    |    微信昵称: empty    |    "
       "vm-60 · NVIDIA RTX 4090 D (24GB) · torch 2.9.1+cu128 · Python 3.10.12")
LINK = ("教程: datawhalechina/llm-algo-leetcode  |  DataWhale 社区  |  "
        "202609 推理优化 Task3 · KV Cache 状态与生命周期 (Issue #166)")


def load(name):
    return nbformat.read(os.path.join(EV, name), as_version=4)


def cell_by_source(nb, needle):
    for cell in nb.cells:
        if cell.cell_type == "code" and needle in "".join(cell.source):
            return cell
    raise SystemExit("no cell with " + needle)


def out_text(cell, limit=None):
    text = ""
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            text += "".join(out.get("text", []))
        elif out.output_type == "execute_result":
            text += "".join(out.get("data", {}).get("text/plain", []))
        elif out.output_type == "error":
            text += "ERROR " + out.get("ename", "") + ": " + str(out.get("evalue", ""))
    text = text.strip().replace("✅", "[PASS]")
    return text[:limit] if limit else text


def exp(name):
    path = os.path.join(EV, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


exp1, exp2, exp3 = (exp("task3_exp1_paged_fragmentation.json"),
                    exp("task3_exp2_prefix_reuse.json"),
                    exp("task3_exp3_prefix_cache_gpu.json"))

# ---------------------------------------------------------------- 图 1：Part02 · 22 PagedAttention
nb22 = load("22_solved_executed.ipynb")
out22 = out_text(cell_by_source(nb22, "test_paged_attention_manager"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task3 —— Part 02 · 22 vLLM PagedAttention 运行截图",
                 ENV, LINK, nrows=3, ncols=2, figsize=(15.6, 9.6))
C.card(gs[0, 0], "In: 我的实现（TODO 1-7）",
       ["def allocate_for_prefill(req):",
        "    needed = ceil(seq_len / block_size)",
        "    if len(free_blocks) < needed: raise OOM",
        "    req.block_table += free_blocks[:needed]",
        "",
        "def allocate_for_decode(req):",
        "    is_new_block = (seq_len+1) % bs == 1",
        "    # 只在跨过 block 边界时按需 +1 块",
        "",
        "get_physical_cache: 按 block_table 顺序",
        "  拼接物理块并截断到 seq_len"], body_size=8.4)
C.card(gs[0, 1], "Out: 内存管理三段链路测试", out22.splitlines(), body_size=7.6)
if exp1:
    rows = ["block_size  paged_tok  tail_waste  util   blocks",
            "----------------------------------------------"]
    for r in exp1["block_size_scan"]:
        rows.append("%6d %10d %11d %6.1f%% %7d" % (
            r["block_size"], r["paged_tokens"], r["paged_tail_waste_tokens"],
            r["paged_utilization"] * 100, r["paged_blocks"]))
    rows += ["", "8 请求长度 17..2047（共 4080 token），",
             "连续预留 4096 x 8 = 32768 token（利用率 12.5%）"]
    C.card(gs[1, 0], "实测：block_size 扫描 —— 尾块碎片 vs 块表开销", rows, body_size=8.4)
    g = exp1["gpu"]
    C.card(gs[1, 1], "实测：真实 GPU 分配（4090 D，LLaMA-7B 账本 0.5 MiB/token）",
           ["paged(block=16)   %6d tok -> %8.1f MiB" % (g["paged_tokens"], g["paged_peak_mib"]),
            "contiguous(4096)  %6d tok -> %8.1f MiB" % (g["contiguous_tokens"], g["contiguous_peak_mib"]),
            "-----------------------------------------------",
            "节省 %d MiB (%.1f%%)，同等显存下并发 x %.2f" % (
                g["saved_mib"], g["saved_ratio"] * 100, g["concurrency_gain"]),
            "",
            "分页把「按最大长度预留」改成「按实际长度分配」",
            "消除的是预留尾部 + 请求间不连续两块浪费"], body_size=8.4)
C.card(gs[2, 0], "逻辑 token / 物理 block / block table",
       ["逻辑视角:  [t0 t1 ... t4095]  连续一维",
        "                  |",
        " block_table: [5, 0, 9, 3, ...]  (逻辑 i -> 物理)",
        "                  |",
        "物理池:    [b0][b1][b2][b3][b4][b5]...",
        "            ^       ^       ^",
        "          req2    空闲    req1(逻辑第0块)",
        "",
        "逻辑连续 != 物理连续：kernel 按块表寻址，",
        "拼装只在需要时发生（TODO 7）"], body_size=8.4)
C.card(gs[2, 1], "为什么逻辑连续不要求物理连续",
       ["Attention 计算只依赖 token 顺序，",
        "不依赖显存地址：按块表顺序读 K/V 即得",
        "正确结果。物理不连续换来的是：",
        "  * 按需分配（无预留尾部）",
        "  * 请求间共享池（无外部碎片）",
        "  * 前缀复用可按块共享（refcount）",
        "代价：块表寻址、非连续访存、kernel 复杂化"], body_size=8.4)
C.save(fig, os.path.join(EV, "task3_22_paged_attention_runshot.png"))

# ---------------------------------------------------------------- 图 2：Part02 · 24 RadixAttention
nb24 = load("24_solved_executed.ipynb")
out24 = out_text(cell_by_source(nb24, "test_radix_attention"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task3 —— Part 02 · 24 SGLang RadixAttention 运行截图",
                 ENV, LINK, nrows=2, ncols=2, figsize=(15.6, 6.8))
C.card(gs[0, 0], "In: 我的实现（TODO 1-6）",
       ["insert: 按首 token 找 child -> lcp ->",
        "  部分重叠则分裂 shared/old_suffix/new",
        "match_prefix: 只沿完整共享边向下，",
        "  且只有 terminal 节点计入可复用长度",
        "split_prompt: hit_len 之前复用、之后重算"], body_size=8.6)
C.card(gs[0, 1], "Out: 共享边 / 最长命中 / 拆分测试", out24.splitlines(), body_size=8.6)
C.card(gs[1, 0], "Radix Tree 结构（insert [0,1,2,3] [0,1,2,3,4] [9,9,9] 后）",
       ["root",
        " ├─ [0,1,2,3](terminal)",
        " │    └─ [4](terminal)     <- 新后缀挂到共享边下",
        " └─ [9,9,9](terminal)",
        "",
        "公共前缀只保留一条共享边；",
        "部分重叠时旧边分裂，terminal/缓存指针",
        "转移到旧后缀，shared 记录公共段"], body_size=8.6)
C.card(gs[1, 1], "GPU 边界探针（real_gpu，4090 D）",
       ["cached_prefix = [101,102,103,104]",
        "request       = [101,102,103,104,201,202]",
        "hit_len = 4 -> 复用 4 token，suffix 2 token",
        "hit_tensor [4] / suffix_tensor [2]",
        "",
        "树匹配是 CPU 侧索引操作；命中长度决定",
        "GPU 侧只需为 suffix 建输入张量"], body_size=8.6)
C.save(fig, os.path.join(EV, "task3_24_radix_attention_runshot.png"))

# ---------------------------------------------------------------- 图 3：Part02 · 34 Prefix Caching & Chunked Prefill
nb34 = load("34_solved_executed.ipynb")
out34 = out_text(cell_by_source(nb34, "test_prefix_cache_manager"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task3 —— Part 02 · 34 Prefix Caching 与 Chunked Prefill 运行截图",
                 ENV, LINK, nrows=2, ncols=3, figsize=(15.6, 7.0))
C.card(gs[0, 0], "In: 我的实现（TODO 1-8）",
       ["_normalize: 统一 list[int]，拒绝非整数",
        "_chunk_tokens: 每 block_size 切块（尾块可不足）",
        "match_prefix: 只从开头连续命中，取最长",
        "chunked_suffix_prefill_plan:",
        "  命中前缀不再进计划，只对 suffix 分块"], body_size=8.4)
C.card(gs[0, 1], "Out: PrefixCacheManager 测试", out34.splitlines(), body_size=8.8)
C.card(gs[0, 2], "执行账本（block_size=2）",
       ["prompt [1,2,3,9], cached [(1,2,3),(1,2,9)]",
        "hit_tokens      = 3",
        "uncached_tokens = 1",
        "reuse_ratio     = 0.75",
        "",
        "完整命中 -> suffix 为空 -> 无 chunk"], body_size=8.4)
C.card(gs[1, 0], "GPU 分块探针（real_gpu，4090 D）",
       ["suffix 4096 token, hidden 1024, fp16",
        "one_shot peak = 16.0 MiB",
        "chunked(512)  peak =  2.0 MiB   (1/8)",
        "",
        "分块把单次工作集切成 8 段，",
        "峰值分配随之下降（合成探针）"], body_size=8.4)
C.card(gs[1, 1], "Chunked Prefill 调度效果",
       ["无分块:  [---- 4096 token 一次 prefill ----]",
        "          ^ 独占计算，decode 排队等",
        "分块:    [512][512][512][512][512][512][512][512]",
        "          ^ 每个 chunk 后可插入 decode 步",
        "",
        "TTFT 不变（总算量相同），但 TPOT 抖动下降，",
        "长 prompt 不再阻塞已生成请求"], body_size=8.2)
C.card(gs[1, 2], "与 Decode 的调度关系",
       ["prefill 是 compute-bound（并行 token）",
        "decode 是 memory-bound（读 KV+权重）",
        "chunk 化后两类任务可混排进同一 batch，",
        "提高 GPU 利用率；代价是调度复杂度与",
        "块边界效率损失（尾块不满）"], body_size=8.4)
C.save(fig, os.path.join(EV, "task3_34_prefix_chunked_prefill_runshot.png"))

# ---------------------------------------------------------------- 图 4：Part02 · 69 Prefix Caching Benchmark
nb69 = load("69_solved_executed.ipynb")
out69 = out_text(cell_by_source(nb69, "test_prefix_cache_benchmark_template"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task3 —— Part 02 · 69 Prefix Caching Benchmark 运行截图",
                 ENV, LINK, nrows=2, ncols=3, figsize=(15.6, 7.0))
C.card(gs[0, 0], "In: 我的实现（TODO 1-4）",
       ["simulate: key=(block_index, block)",
        "  只复用完整 block，LRU 驱逐",
        "summarize: run_count/avg/best",
        "compare: 同 workload_id 才比较",
        "recommend: 三个门槛全过才 accept"], body_size=8.4)
C.card(gs[0, 1], "Out: CPU 缓存模拟与测试", out69.splitlines()[:14], body_size=7.4)
C.card(gs[0, 2], "决策逻辑（TODO 4）",
       ["hit_rate_ok  = gain >= min_hit_rate_gain",
        "ttft_ok      = delta < 0",
        "overhead_ok  = delta <= max_overhead",
        "三者全过 -> accept",
        "命中+延迟过、开销超 -> tune",
        "否则 -> reject",
        "命中率不够时不能仅因 TTFT 改善就 accept"], body_size=8.2)
C.card(gs[1, 0], "G0/G1 对照要同时记录",
       ["复用: hit_rate / reused_tokens /",
        "      prefill_work_reduction",
        "性能: TTFT / TPOT / E2E / P99 / 吞吐",
        "资源: peak_memory / cache_blocks /",
        "      eviction_count",
        "质量: status / quality / OOM"], body_size=8.4)
C.card(gs[1, 1], "证据边界",
       ["CPU 只验证 block 命中/驱逐逻辑；",
        "真实命中率必须来自 backend metrics",
        "或日志，不能只用 TTFT/吞吐反推；",
        "vLLM 未安装 -> RUN_REAL_BACKEND=False，",
        "真实 backend 对照留待 serving 环境"], body_size=8.4)
C.card(gs[1, 2], "与 66 baseline 的对照方式",
       ["66: 固定 workload 的 G0/G1 单变量对照",
        "69: 唯一变量 = cache policy，",
        "    模型/backend/并发/生成长度全固定",
        "输出 accept/tune/reject + next_action"], body_size=8.4)
C.save(fig, os.path.join(EV, "task3_69_prefix_cache_benchmark_runshot.png"))

# ---------------------------------------------------------------- 图 5：Part02 · 66 推理性能比较（Task2 已解答，Task3 复跑）
nb66 = load("66_solved_executed.ipynb")
out66 = out_text(cell_by_source(nb66, "test_inference_project_template"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task3 —— Part 02 · 66 推理性能比较 运行截图（Task2 解答版复跑）",
                 ENV, LINK, nrows=2, ncols=3, figsize=(15.6, 7.0))
C.card(gs[0, 0], "Out: 66 模板测试（vm-60 复跑通过）", out66.splitlines(), body_size=8.8)
C.card(gs[0, 1], "五个指标各反映什么",
       ["TTFT   : 排队+prefill，交互体感第一关",
        "TPOT   : 解码稳态每 token 时间",
        "吞吐   : 系统级 token/s（随 batch 升）",
        "峰值显存: 容量边界（权重+KV+激活）",
        "P99    : 长尾延迟，SLO  Usually 看它"], body_size=8.4)
C.card(gs[0, 2], "指标冲突",
       ["batch 升 -> 吞吐升、TTFT 升、显存升",
        "prefill 升 -> TTFT 升、TPOT 基本不变",
        "上下文升 -> TTFT/TPOT/显存都升",
        "单指标最优通常不是全局最优"], body_size=8.4)
C.card(gs[1, 0], "bottleneck 分类（66 CPU 模拟）",
       ["prefill_share vs decode_share",
        "decode-bound -> 看 KV 读写/调度/投机解码",
        "prefill-bound -> 看 attention kernel/分块",
        "memory-bound -> 看 KV 量化/淘汰/并发"], body_size=8.4)
C.card(gs[1, 1], "本机环境（vm-60）",
       ["GPU: RTX 4090 D (sm89, 24GB)",
        "torch 2.9.1+cu128, cuda 12.8",
        "bf16 supported: True",
        "vllm on kernel: False",
        "-> 真实 backend 对照需另装环境"], body_size=8.4)
C.card(gs[1, 2], "与 69 的衔接",
       ["66 建立 G0/G1 对照与指标口径；",
        "69 把唯一变量换成 cache policy，",
        "用同一套指标判断 prefix cache",
        "是否值得上线（accept/tune/reject）"], body_size=8.4)
C.save(fig, os.path.join(EV, "task3_66_inference_metrics_runshot.png"))

# ---------------------------------------------------------------- 图 6：自加实验
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task3 —— 自加实验：前缀复用命中 / 容量治理 / 真实模型 Prefix Cache",
                 ENV, LINK, nrows=3, ncols=2, figsize=(15.6, 9.6))
if exp2:
    r = exp2["radix"]
    C.card(gs[0, 0], "实验2a: Radix 前缀复用（146 请求，共享 256 token system prompt）",
           ["avg_prompt_len        = %.1f" % r["avg_prompt_len"],
            "avg_hit_len           = %.1f" % r["avg_hit_len"],
            "prefill_work_reduction= %.1f%%" % (r["prefill_work_reduction"] * 100),
            "requests_with_hit     = %d / %d" % (r["requests_with_hit"], exp2["workload"]["requests"]),
            "max_hit_len           = %d（多轮历史逐轮变长）" % r["max_hit_len"],
            "",
            "多轮对话命中最长；独立请求只有落在",
            "已登记完整路径上才能命中（terminal 口径）"], body_size=8.2)
    rows = ["cap_blocks cap_tok  hit_rate  tok_reuse  evictions",
            "-------------------------------------------------"]
    for row in exp2["lru_sweep"]:
        rows.append("%8d %8d %8.1f%% %9.1f%% %9d" % (
            row["capacity_blocks"], row["capacity_tokens"],
            row["hit_rate"] * 100, row["token_reuse_rate"] * 100, row["eviction_count"]))
    C.card(gs[0, 1], "实验2b: LRU 容量扫描（block=16 token）",
           rows + ["", "容量 < 工作集(512 tok) -> 命中率崩塌；",
                   "容量足够后命中率封顶，但驱逐仍有维护成本"], body_size=8.2)
if exp3:
    rows = ["system  g0_ttft  g1_ttft  speedup  tok_saved",
            "----------------------------------------------"]
    for row in exp3["sequential_sweep"]:
        rows.append("%6d %8.2f %8.2f %7.2fx %8.1f%%" % (
            row["system_len"], row["g0_ttft_ms"], row["g1_ttft_ms"],
            row["ttft_speedup"], row["prefill_token_reduction"] * 100))
    rows += ["", "gpt2 124M fp16, 8 请求 x (system+200) token,",
             "串行逐请求；prefill 计算量降 44~70%，",
             "但小模型下 TTFT 被固定开销摊薄"]
    C.card(gs[1, 0], "实验3a: 真实模型 Prefix Cache（串行，4090 D）", rows, body_size=8.2)
    b = exp3["batched"]
    C.card(gs[1, 1], "实验3b: 批量 8 请求一次前向（system=800）",
           ["G0 full prefill : ttft %6.2f ms  peak %7.1f MiB" % (b["g0_ttft_ms"], b["g0_peak_mib"]),
            "G1 prefix cache : ttft %6.2f ms  peak %7.1f MiB" % (b["g1_ttft_ms"], b["g1_peak_mib"]),
            "--------------------------------------------------",
            "TTFT speedup x %.2f，prefill token -%.0f%%" % (b["ttft_speedup"], b["prefill_token_reduction"] * 100),
            "冷启动(首次前缀) %.2f ms 由后续请求摊销" % b["g1_cold_ttft_ms"],
            "",
            "批量摊薄固定开销后，prefill 计算占比上升，",
            "prefix cache 的 TTFT 收益直接显形"], body_size=8.2)
    r0 = exp3["sequential_sweep"][-1]
    C.card(gs[2, 0], "正确性：复用不改变输出",
           ["max |logit_G1 - logit_G0| = %.3f" % r0["max_logit_abs_diff"],
            "G0 top1-top2 最小边距       = %.3f" % r0["min_top2_margin_g0"],
            "top1 一致率 = %.0f%%（8 请求）" % (r0["top1_agreement"] * 100),
            "",
            "logit 差异同量级于近 tie 边距：",
            "个别 argmax 翻转是 fp 近 tie 噪声，",
            "不是复用逻辑错误"], body_size=8.2)
C.card(gs[2, 1], "实验结论",
       ["1. 分页: 省预留尾部+外部碎片（实测 x7.9 并发）",
        "2. 前缀复用: 多轮/共享 prompt 命中最长",
        "3. 容量: 缓存必须大于工作集，否则驱逐吃掉收益",
        "4. TTFT 收益随 batch/长度放大；小请求被固定",
        "   开销摊薄——不能只用 TTFT 反推命中率",
        "5. 峰值显存节省立竿见影（KV 常驻的代价）"], body_size=8.4)
C.save(fig, os.path.join(EV, "task3_exp_prefix_reuse_runshot.png"))
print("ALL FIGURES DONE")
