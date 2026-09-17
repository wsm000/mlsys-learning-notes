# -*- coding: utf-8 -*-
"""生成 Task2 的运行截图：真实 cell 源码 + 真实输出 + 自加实验数据。"""
import json
import os

import nbformat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import task2_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = ("GitHub ID: wsm000    |    微信昵称: empty    |    "
       "vm-60 · NVIDIA RTX 4090 D (24GB) · torch 2.9.1+cu128 · Python 3.10.12")
LINK = ("教程: datawhalechina/llm-algo-leetcode  |  DataWhale 社区  |  "
        "202609 推理优化 Task2 · 单请求 Decode 与生成策略 (Issue #165)")


def load(name):
    return nbformat.read(os.path.join(HERE, name), as_version=4)


def cell_by_source(nb, needle):
    for cell in nb.cells:
        if cell.cell_type == "code" and needle in "".join(cell.source):
            return cell
    raise SystemExit("no cell with " + needle)


def out_text(cell, limit=None):
    text = ""
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            text += "".join(out.get("text", ""))
        elif out.output_type == "execute_result":
            text += "".join(out.get("data", {}).get("text/plain", ""))
        elif out.output_type == "error":
            text += "ERROR " + out.get("ename", "") + ": " + str(out.get("evalue", ""))
    text = text.strip().replace("✅", "[PASS]")
    return text[:limit] if limit else text


def body_after(cell, marker, maxlen=1700):
    src = "".join(cell.source)
    idx = src.find(marker)
    src = src[idx:] if idx >= 0 else src
    lines = [ln.rstrip() for ln in src.splitlines() if ln.strip()]
    return "\n".join(lines)[:maxlen]


def exp(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


exp1, exp2, exp3 = exp("task2_exp1_kv_ledger.json"), exp("task2_exp2_sampling.json"), exp("task2_exp3_metrics.json")

# ---------------------------------------------------------------- 图 1：Part01 · 11
nb11 = load("11_KV_Cache_and_Memory_Growth_solved_executed.ipynb")
out_growth = out_text(cell_by_source(nb11, "def kv_cache_bytes"))
out_heads = out_text(cell_by_source(nb11, "def kv_cache_gb"))
out_paged = out_text(cell_by_source(nb11, "def paged_attention_pages"))

fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task2 —— Part 01 · 11 KV Cache 与显存增长 运行截图",
                 ENV, LINK, nrows=2, ncols=3, figsize=(15.6, 8.8))
C.card(gs[0, 0], "In: 账本公式（cell 4）\nKV Bytes ~ 2 x L x B x H_kv x D x S x dtype",
       ["def kv_cache_bytes(seq_len, num_layers, num_kv_heads,", "                 head_dim, batch_size=1, dtype_bytes=2):",
        "    ...", "    return 2 * seq_len * num_layers * num_kv_heads \\", "           * head_dim * batch_size * dtype_bytes",
        "", "S=1024/2048/4096, L=32, H_kv=32, D=128, fp16"], body_size=8.4)
C.card(gs[0, 1], "Out: 序列翻倍 -> 缓存翻倍", out_growth.splitlines(), body_size=9.4)
C.card(gs[0, 2], "Out: MHA / GQA / MQA 只改 H_kv", out_heads.splitlines(), body_size=9.4)
C.card(gs[1, 0], "Out: 组织 vs 表示 vs 复用", out_paged.splitlines(), body_size=8.4)
if exp1:
    hc = exp1["head_configs"]
    C.card(gs[1, 1], "实测（同骨架只改 num_key_value_heads）",
           ["配置      H_kv  实测 cache  账本    并发数(20GiB)",
            "------------------------------------------------",
            "MHA        8    %6.1f MiB %6.1f MiB  %d" % (hc["MHA"]["cache_mib_at_S4096_B1"], hc["MHA"]["ledger_mib"], hc["MHA"]["concurrent_requests_in_budget"]),
            "GQA        2    %6.1f MiB %6.1f MiB  %d" % (hc["GQA"]["cache_mib_at_S4096_B1"], hc["GQA"]["ledger_mib"], hc["GQA"]["concurrent_requests_in_budget"]),
            "MQA        1    %6.1f MiB %6.1f MiB  %d" % (hc["MQA"]["cache_mib_at_S4096_B1"], hc["MQA"]["ledger_mib"], hc["MQA"]["concurrent_requests_in_budget"]),
            "",
            "S=4096, B=1, 12 层, 8 query heads, head_dim=128",
            "实测字节与 2·L·B·H_kv·D·S·2 完全一致",
            "（误差 0，说明缓存就是这些张量本身）"], body_size=8.6)
    seq_rows = ["seq_len   实测 MiB   账本 MiB   每 token 字节", "-----------------------------------------------"]
    for row in exp1["seq_scan"]:
        seq_rows.append("%6d   %8.1f  %8.1f   %9d" % (row["seq_len"], row["measured_mib"], row["ledger_mib"], row["bytes_per_token"]))
    C.card(gs[1, 2], "序列线性扫描（GQA, B=1）\n每 token 成本恒定 = 线性增长，不是平方",
           seq_rows, title_color="#1e8449", body_size=8.8)
C.save(fig, os.path.join(HERE, "task2_11_kv_cache_runshot.png"))

# ---------------------------------------------------------------- 图 2：Part02 · 21
nb21 = load("21_Decoding_Strategies_solved_executed.ipynb")
c21 = cell_by_source(nb21, "def test_decoding")
out21 = out_text(c21)
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task2 —— Part 02 · 21 解码策略 运行截图",
                 ENV, LINK, nrows=2, ncols=2, figsize=(14.2, 9.6))
C.card(gs[0, 0], "In: 已补全的三个过滤函数",
       [body_after(cell_by_source(nb21, "def apply_temperature"), "def apply_temperature", 1000)], body_size=7.8)
C.card(gs[0, 1], "Out: 题目自带测试全通过", out21.splitlines(), title_color="#1e8449", body_size=9.0)
C.card(gs[1, 0], "候选数与熵：三种策略改变的是候选集合",
       ["greedy : candidates=10  entropy=1.3310",
        "top_k=3: candidates= 3  entropy=0.8889",
        "top_p=0.8: candidates=3  entropy=0.8889",
        "",
        "断言覆盖：",
        "  · temperature 缩放倍数与排序不变",
        "  · top_k 阈值语义（ties 可保留 > k 个）",
        "  · top_p 首次达到阈值并恢复原词表顺序",
        "  · greedy / batch 输入 / 随机种子可复现",
        "  · 非法 temperature 与 top_p 明确报错",
        "  · 最小自回归循环 3 步输出 [3,3,3]"], body_size=8.6)
C.card(gs[1, 1], "本节口径边界",
       ["· 只验证「选择规则」，不测量真实模型质量",
        "· 熵/候选数是分布形状的代理，不是质量指标",
        "· 质量、重复率、长度需要真实模型实验（见 exp2）",
        "",
        "补题时发现的一处上游问题：",
        "  题目区 decode_next_token 中有一行多余的",
        "  三引号，使该 cell 无法解析；解答版删除了",
        "  这一行，函数体与参考实现一致。"], title_color="#b9770e", body_size=8.6)
C.save(fig, os.path.join(HERE, "task2_21_decoding_runshot.png"))

# ---------------------------------------------------------------- 图 3：Part02 · 23
nb23 = load("23_Speculative_Decoding_solved_executed.ipynb")
out23 = out_text(cell_by_source(nb23, "def test_speculative_decoding"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task2 —— Part 02 · 23 投机解码 运行截图",
                 ENV, LINK, nrows=2, ncols=2, figsize=(14.2, 9.2))
C.card(gs[0, 0], "In: 已补全的 speculative_decode_step",
       [body_after(cell_by_source(nb23, "def speculative_decode_step"), "for i in range(K)", 1500)], body_size=8.0)
C.card(gs[0, 1], "Out: 题目自带测试全通过", out23.splitlines(), title_color="#1e8449", body_size=9.0)
C.card(gs[1, 0], "断言覆盖的四种路径",
       ["输入契约：非负 + 每行归一化，否则 ValueError",
        "接受：p >= q 必接受；p < q 按 p/q 掷硬币",
        "拒绝：从 residual=max(target-draft,0) 归一化后采样",
        "全部接受：追加 target_probs[K] 的 bonus token",
        "",
        "边界：",
        "  · q=0 且 p>0：直接接受（避免除零后误拒）",
        "  · q=0 且 p=0：进入拒绝分支",
        "  · residual 质量为 0：明确报错而不是静默采样"], body_size=8.6)
C.card(gs[1, 1], "为什么这样能保持目标分布",
       ["接受-拒绝采样把 q 的候选按 min(1,p/q)",
        "校正成 p 的样本；被拒时用 residual",
        "  p'(x) ∝ max(p(x)-q(x), 0)",
        "补齐剩余质量，因此最终分布仍等于 p。",
        "",
        "draft 的职责：便宜地提出候选",
        "target 的职责：集中验证 + 兜底修正/bonus",
        "收益上限 ≈ 平均接受长度 / 一次验证成本",
        "（68 节用 acceptance rate 与 verify cost 判断）"], title_color="#2471a3", body_size=8.6)
C.save(fig, os.path.join(HERE, "task2_23_speculative_runshot.png"))

# ---------------------------------------------------------------- 图 4：Part02 · 35
nb35 = load("35_Multi_Token_Decoding_solved_executed.ipynb")
out35 = out_text(cell_by_source(nb35, "def test_multi_token_decoder"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task2 —— Part 02 · 35 多 Token 解码 运行截图",
                 ENV, LINK, nrows=2, ncols=2, figsize=(14.2, 9.0))
C.card(gs[0, 0], "In: 已补全的 MultiTokenDecoderSim",
       [body_after(cell_by_source(nb35, "class MultiTokenDecoderSim"), "def propose", 900),
        "", body_after(cell_by_source(nb35, "class MultiTokenDecoderSim"), "def verify", 700)], body_size=7.6)
C.card(gs[0, 1], "Out: 题目自带测试全通过", out35.splitlines(), title_color="#1e8449", body_size=9.0)
C.card(gs[1, 0], "一轮的状态关系",
       ["propose   : 截取本轮候选前缀（<= max_proposal_len）",
        "verify    : 从左到右逐个验证，首次拒绝即停止",
        "accept    : 已接受前缀 -> 本轮真正推进的 token",
        "reject    : rejected_at 位置起为回退后缀",
        "decode    : 汇总 accepted_len 与 progress_per_round",
        "",
        "progress_per_round = 接受数 / 提议数（= 2/3 等）",
        "全部接受时 progress = 1.0，回退后缀为空"], body_size=8.6)
C.card(gs[1, 1], "与投机解码的差别",
       ["候选生成：",
        "  投机解码 = 独立 draft 模型的分布采样",
        "  多 token 解码 = 同一解码步内尝试推进多个候选",
        "目标验证：",
        "  投机解码用 min(1,p/q) 校正，保证分布不变",
        "  本节用阈值规则（target >= draft * ratio），",
        "  只观察控制流，不保证严格分布等价",
        "共同难点：接受率、验证成本、首次拒绝后的回退",
        "此处 progress_per_round 不是吞吐收益"], title_color="#b9770e", body_size=8.4)
C.save(fig, os.path.join(HERE, "task2_35_multi_token_runshot.png"))
print("FIG 1-4 done", flush=True)


# ================= 第二部分：66 / 68 与自加实验汇总 =================

exp3b = exp("task2_exp3b_kv_reuse.json")

exp1 = exp("task2_exp1_kv_ledger.json")
exp2 = exp("task2_exp2_sampling.json")
exp3 = exp("task2_exp3_metrics.json")
exp3b = exp("task2_exp3b_kv_reuse.json")

# ---------------------------------------------------------------- 图 5：Part02 · 66
nb66 = load("66_Inference_Performance_Comparison_solved_executed.ipynb")
out66_test = out_text(cell_by_source(nb66, "def test_inference_project_template"))
out66_demo = out_text(cell_by_source(nb66, "# TODO 1: 模拟请求执行"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task2 —— Part 02 · 66 推理性能比较 运行截图",
                 ENV, LINK, nrows=2, ncols=2, figsize=(15.0, 10.2))
C.card(gs[0, 0], "In: 已补全的请求模拟（离散事件 + 指标口径）",
       [body_after(cell_by_source(nb66, "def simulate_inference_requests"), "request_results = []", 1100)], body_size=7.6)
C.card(gs[0, 1], "Out: 题目自带测试全通过", out66_test.splitlines(), title_color="#1e8449", body_size=9.0)
C.card(gs[1, 0], "Out: baseline vs candidate 与选型结论", out66_demo.splitlines(), body_size=7.4)
if exp3:
    rows = ["prompt batch  TTFT(ms) TPOT(ms) e2e(ms)  tok/s   峰值显存(MiB)",
            "-------------------------------------------------------------"]
    for r in exp3["prompt_sweep"]:
        rows.append("%6d %5d  %8.2f %8.3f %7.1f %7.1f %10.1f" % (
            r["prompt_len"], r["batch"], r["ttft_ms"], r["tpot_ms"], r["e2e_ms"],
            r["throughput_tok_s"], r["peak_allocated_mib"]))
    rows.append("  ...")
    for r in exp3["batch_sweep"]:
        rows.append("%6d %5d  %8.2f %8.3f %7.1f %7.1f %10.1f" % (
            r["prompt_len"], r["batch"], r["ttft_ms"], r["tpot_ms"], r["e2e_ms"],
            r["throughput_tok_s"], r["peak_allocated_mib"]))
    rows += ["", "prompt 变长：TTFT 涨、TPOT 基本不变",
             "batch 变大：吞吐近线性涨、TTFT 同步涨、显存涨"]
    if exp3b:
        rows.append("")
        rows.append("KV Cache 对照（合成 LLaMA，B=16 S=4096）：")
        r = exp3b["rows"][-1]
        rows.append("  带缓存 TPOT=%.1f ms  无缓存 %.1f ms（%.0fx）" % (
            r["cached"]["tpot_ms"], r["recompute"]["tpot_ms"], r["tpot_speedup"]))
    C.card(gs[1, 1], "自加实验：五个指标必须同时记录", rows, title_color="#2471a3", body_size=7.6)
else:
    C.card(gs[1, 1], "自加实验数据", ["exp3 尚未生成"], title_color="#c0392b")
C.save(fig, os.path.join(HERE, "task2_66_inference_metrics_runshot.png"))

# ---------------------------------------------------------------- 图 6：Part02 · 68
nb68 = load("68_Speculative_Decoding_Benchmark_solved_executed.ipynb")
out68_test = out_text(cell_by_source(nb68, "def test_speculative_benchmark_template"))
out68_plan = out_text(cell_by_source(nb68, "project_config = shared_project_config"))
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task2 —— Part 02 · 68 投机解码基准 运行截图",
                 ENV, LINK, nrows=2, ncols=2, figsize=(15.0, 9.6))
C.card(gs[0, 0], "In: 已补全的 simulate_speculative_decode",
       [body_after(cell_by_source(nb68, "def simulate_speculative_decode"), "for round_index, item in enumerate(rounds)", 1000)], body_size=7.6)
C.card(gs[0, 1], "Out: 题目自带测试全通过", out68_test.splitlines(), title_color="#1e8449", body_size=9.0)
C.card(gs[1, 0], "Out: G0/G1/G2 实验计划（真实 backend 入口）",
       [ln for ln in out68_plan.splitlines() if ln.strip()][:7] +
       ["", "本机未安装 vLLM，RUN_BACKEND_SMOKE=False，",
        "保持 CPU-first：只输出统一实验计划与字段。"], body_size=7.4)
C.card(gs[1, 1], "baseline vs speculative candidate（题目给定口径）",
       ["baseline : ttft=120 ms  throughput=100  acc=0.00  verify=40 ms",
        "candidate: ttft=110 ms  throughput=135  acc=0.72  verify=48 ms",
        "",
        "ttft_delta_ms      = -10    （候选 TTFT 更快）",
        "throughput_gain    = +35    （差值，不是比例）",
        "throughput_speedup = 1.35   （比值）",
        "verify_cost_delta  = +8     （验证更贵）",
        "decision = accept -> promote_to_serving_eval",
        "",
        "判定顺序：quality_ok -> acceptance_rate ->",
        "          throughput_gain -> verify_cost_delta",
        "quality_ok=False 时优先 reject。"], title_color="#b9770e", body_size=8.2)
C.save(fig, os.path.join(HERE, "task2_68_speculative_benchmark_runshot.png"))

# ---------------------------------------------------------------- 图 7：自加实验汇总
fig, gs = C.page("llm-algo-leetcode 推理优化 | 202609 Task2 —— 自加实验：KV 账本 · 生成策略 · 指标口径",
                 ENV, LINK, nrows=3, ncols=2, figsize=(15.6, 14.4))
if exp1:
    hc = exp1["head_configs"]
    rows = ["配置  H_kv  实测cache 账本    20GiB并发  每token",
            "-------------------------------------------------"]
    for name in ("MHA", "GQA", "MQA"):
        d = hc[name]
        rows.append("%-4s  %4d  %7.1f  %7.1f  %8d  %6.1f KiB" % (
            name, d["num_kv_heads"], d["cache_mib_at_S4096_B1"], d["ledger_mib"],
            d["concurrent_requests_in_budget"], d["cache_kib_per_token"]))
    rows += ["", "decode 每步(B=8,S=1024)："]
    for name in ("MHA", "GQA", "MQA"):
        d = hc[name]
        rows.append("  %-4s %6.2f ms/step  %6.1f tok/s" % (
            name, d["decode_ms_per_step_B8_S1024"], d["decode_throughput_tok_s_B8_S1024"]))
    rows += ["", "→ 相同骨架只改 H_kv：容量 8x 差距,",
             "  短上下文下每步耗时几乎不变。"]
    C.card(gs[0, 0], "实验 1：KV head 配置（MHA/GQA/MQA）", rows, title_color="#1e8449", body_size=8.4)
    rows3 = ["长上下文 decode（B=16）",
             "配置    S     峰值显存(MiB)  ms/step  tok/s",
             "-----------------------------------------------"]
    for d in exp1["long_context_decode"]:
        rows3.append("%-5s %6d %11.1f %9.3f %7.1f" % (
            d["name"], d["seq_len"], d["peak_mib"], d["decode_ms_per_step"], d["throughput_tok_s"]))
    rows3 += ["", "S=8192 时 MHA cache 是 MQA 的 8 倍，",
              "步时也变成 2 倍（22.6 vs 11.3 ms）——",
              "长上下文下 KV 读取才成为瓶颈；",
              "短上下文下只是容量差异。"]
    C.card(gs[0, 1], "实验 1 续：容量差何时变成延迟差", rows3, title_color="#b9770e", body_size=8.4)
if exp2:
    rows2 = ["策略             长度  rep2   distinct2  PPL    一致性  tok/s",
             "----------------------------------------------------------------"]
    for name, d in exp2["strategies"].items():
        rows2.append("%-16s %5.1f  %.3f  %8.3f  %6.2f  %5.2f  %6.1f" % (
            name, d["mean_length"], d["mean_rep2"], d["distinct_2"],
            d["mean_self_ppl"], d["mean_agreement_with_greedy"], d["decode_tok_s"]))
    rows2 += ["",
              "温度升高：重复率单调下降（0.68->0.02）",
              "          distinct-2 单调上升（0.10->0.92）",
              "          自困惑度变差（2.2->33.9）",
              "          跨 seed 波动变大（0.57->7.11）",
              "greedy 的 PPL 最低恰恰因为它在重复自己"]
    C.card(gs[1, 0], "实验 2：采样策略（gpt2, 6 prompt x 3 seed）", rows2, title_color="#2471a3", body_size=7.6)
    rows4 = ["固定条件：同一模型 / 同一 prompt 集 / 同一 max_new",
             "指标口径：",
             "  长度     生成 token 数（EOS 提前结束）",
             "  rep2     重复 bigram 占比（越高越啰嗦）",
             "  distinct2 跨样本 bigram 去重比例（越高越多样）",
             "  PPL      同一模型对生成段的自困惑度",
             "  一致性   与 greedy 的逐 token 相同比例",
             "  tok/s    解码速度（采样开销可忽略）",
             "",
             "结论：采样只改「选择规则」，不改模型计算；",
             "      多样性与稳定性是一对反向指标，",
             "      必须和重复率一起读。"]
    C.card(gs[1, 1], "实验 2 的口径与边界", rows4, title_color="#7d3c98", body_size=8.2)
if exp3b:
    rows5 = ["同一模型、同一 batch/prompt，只切换是否复用 KV：",
             "B   S     带缓存TPOT  无缓存TPOT  倍数   峰值显存(MiB)",
             "-------------------------------------------------------"]
    for r in exp3b["rows"]:
        rows5.append("%-3d %5d %9.1f %10.1f %6.1fx  %6.0f->%6.0f" % (
            r["batch"], r["prompt_len"], r["cached"]["tpot_ms"], r["recompute"]["tpot_ms"],
            r["tpot_speedup"], r["recompute"]["peak_mib"], r["cached"]["peak_mib"]))
    rows5 += ["",
              "序列越长，重算的代价按 O(S) 放大：",
              "  S=512  -> 1.5x",
              "  S=2048 -> 6.6x",
              "  S=4096 -> 34x（e2e 0.5s vs 6.0s）",
              "这正是「KV Cache 是历史状态成本」的收益面"]
    C.card(gs[2, 0], "实验 3b：KV Cache 收益边界（合成 LLaMA）", rows5, title_color="#c0392b", body_size=8.2)
    C.card(gs[2, 1], "三个实验各自回答什么",
           ["实验 1（4.3-1）：不同 KV head 配置下的",
             "  显存占用 / 缓存利用率 / 生成性能",
             "  -> 容量与并发先变，长上下文才轮到延迟",
             "",
             "实验 2（4.3-2）：不同采样策略对",
             "  质量 / 重复率 / 生成长度的影响",
             "  -> 只改选择规则，指标必须成组读",
             "",
             "实验 3/3b（4.3-3）：TTFT / TPOT / 吞吐 /",
             "  峰值显存 / 端到端延迟同时记录",
             "  -> 交互体验、稳态成本、服务容量、",
             "     资源上限、用户感知是五个不同问题"], title_color="#16264a", body_size=8.4)
C.save(fig, os.path.join(HERE, "task2_exp_kv_sampling_runshot.png"))
print("FIG 5-7 done", flush=True)
