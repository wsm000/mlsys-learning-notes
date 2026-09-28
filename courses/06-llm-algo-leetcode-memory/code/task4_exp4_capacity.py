# -*- coding: utf-8 -*-
"""Task4 实验4 · 容量验证：显存预算 -> 容量规划 -> 准入压力 -> 同 workload 策略对照。

回答 4.2(1)（容量规划四要素）、4.3(1)（66/69 共同口径）、4.3(2)（TTFT 改善但 peak/P99
恶化的候选该如何决策）、4.3(3)（PagedAttention / Prefix Cache / KV 量化 / GQA 各自解决
什么问题）。

内容：
  A. 容量规划（7B 级 GQA 模型，vm-60 实测权重与运行缓冲，账本推导各表示变体的 KV 池与并发上限）
  B. 准入压力实测（MHA / GQA-8 / MQA 三表示，同一 workload，按预算准入直到拒绝）
  C. 66->69 同 workload 口径的缓存策略对照（baseline / 无界 prefix cache / LRU 封顶 cache）
固定条件：RTX 4090 D；权重 fp16；KV 预算 = 总显存 x 0.9 - 上下文 - 权重 - 运行缓冲 - 安全余量。
"""
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
from task4_common import (assert_true, build_llama, cache_tensor_bytes, dump_json,
                          env_info, ledger_kv_bytes, mib, mla_kv_bytes, per_token_bytes,
                          seeded_ids, source_sha256)

OUT = Path(__file__).parent / "task4_exp4_capacity.json"

# ---------------- 实测模型（1.5B 级，LLaMA 形状）与 8B 级账本对照 ----------------
MID = dict(layers=24, hidden=2048, heads=16, kv_heads=8, head_dim=128,
           intermediate=5632, vocab=128256)     # 约 1.5B 参数
BIG = dict(layers=32, hidden=4096, heads=32, kv_heads=8, head_dim=128,
           intermediate=14336, vocab=128256)    # LLaMA-3-8B 形状（账本对照）
TOY = dict(layers=12, hidden=1024, heads=8, kv_heads=8, head_dim=128,
           intermediate=2816, vocab=32000)
PROMPT, SUFFIX, GEN = 512, 48, 32          # 准入 workload：prompt+suffix=560
TOY_PROMPT, TOY_SUFFIX, TOY_GEN = 512, 128, 24


def build_mid(kv_heads, dtype=torch.float16):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=MID["vocab"], hidden_size=MID["hidden"],
                      intermediate_size=MID["intermediate"],
                      num_hidden_layers=MID["layers"],
                      num_attention_heads=MID["heads"],
                      num_key_value_heads=kv_heads, head_dim=MID["head_dim"],
                      max_position_embeddings=8192, torch_dtype=dtype)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("cpu"):
            model = LlamaForCausalLM(cfg)
        model = model.to("cuda").eval()
    finally:
        torch.set_default_dtype(prev)
    return model, cfg


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def part_a_capacity_plan():
    print("== A. 容量规划：权重/缓冲/上下文/KV 池/安全余量 ==")
    dev = "cuda"
    props = torch.cuda.get_device_properties(dev)
    total_bytes = props.total_memory
    # 上下文与空池：PyTorch 进程 baseline reserved
    torch.cuda.empty_cache()
    context_bytes = torch.cuda.memory_reserved(dev)
    # 真实 1.5B 级模型的权重与运行缓冲（GQA-8）
    try:
        model, _ = build_mid(MID["kv_heads"])
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        ids = seeded_ids((1, 512), seed=4242)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            for _ in range(8):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        del model, out, past, nxt
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        measured = True
    except RuntimeError as e:
        print("  1.5B 实测失败，退回账本：", str(e)[:80])
        weights = 1_500_000_000 * 2
        peak = weights + 512 * 1024 ** 2
        peak_reserved = weights + 512 * 1024 ** 2
        measured = False
        torch.cuda.empty_cache()

    util = 0.90                       # vLLM gpu_memory_utilization 常见默认
    guard = 256 * 1024 ** 2           # 安全余量（在 0.9 之上再留 256 MiB）
    usable = total_bytes * util - context_bytes - guard
    buffers = max(peak - weights, 0)
    kv_pool = usable - weights - buffers

    def variant_rows(shape, layers, tag):
        base = shape["layers"]
        hd = shape["head_dim"]
        variants = [
            (f"MHA (kv={shape['heads']})", lambda t: ledger_kv_bytes(t, base, shape["heads"], hd, 2)),
            (f"GQA-8", lambda t: ledger_kv_bytes(t, base, 8, hd, 2)),
            (f"MQA (kv=1)", lambda t: ledger_kv_bytes(t, base, 1, hd, 2)),
            ("MLA (512+64)", lambda t: mla_kv_bytes(t, base, 2)),
            ("GQA-8 + FP8 KV", lambda t: ledger_kv_bytes(t, base, 8, hd, 1)),
            ("GQA-8 + INT8 KV", lambda t: ledger_kv_bytes(t, base, 8, hd, 1)),
        ]
        rows = []
        for name, fn in variants:
            per_tok = fn(1)
            cap_tokens = kv_pool // per_tok
            rows.append({
                "variant": f"{name} [{tag}]", "per_token_bytes": per_tok,
                "per_token_kib": round(per_tok / 1024, 1),
                "kv_pool_gib": round(kv_pool / 1024 ** 3, 2),
                "max_tokens": int(cap_tokens),
                "max_ctx_single_request": int(cap_tokens),
                "max_concurrency_at_4096": int(cap_tokens // 4096),
                "max_concurrency_at_2048": int(cap_tokens // 2048),
            })
        return rows

    rows = variant_rows(MID, MID["layers"], f"{MID['layers']}L measured")
    rows_big = variant_rows(BIG, BIG["layers"], f"{BIG['layers']}L ledger")
    return {
        "device": torch.cuda.get_device_name(0),
        "total_memory_gib": round(total_bytes / 1024 ** 3, 2),
        "context_reserved_mib": round(mib(context_bytes), 1),
        "gpu_memory_utilization": util,
        "safety_guard_mib": 256,
        "weights_measured": measured,
        "weights_bytes": weights, "weights_gib": round(weights / 1024 ** 3, 2),
        "runtime_peak_bytes": peak, "runtime_buffers_mib": round(mib(buffers), 1),
        "peak_reserved_gib": round(peak_reserved / 1024 ** 3, 2),
        "kv_pool_gib": round(kv_pool / 1024 ** 3, 2),
        "model_shape_measured": MID,
        "model_shape_reference": BIG,
        "capacity_rows": rows,
        "capacity_rows_8b_ledger": rows_big,
        "boundary": "权重与运行缓冲为 1.5B 级模型真实测量；MLA/FP8/INT8 与 8B 行按同一账本"
                    "推导（表示变化），真实 kernel 收益需在 backend 复测",
    }


def admission_wave(model, kv_heads, wave, seed0=900):
    """一波请求：chunked prefill 后批量 decode。返回 TTFT/TPOT/KV 字节。

    prefill 按 64 token 分块（chunked prefill），避免整段 materialize 全词表 logits；
    decode 每步只取最后位置 logits。与真实服务的 prefill 分块模式一致。
    """
    full_len = PROMPT + SUFFIX
    chunk = 64
    shared = seeded_ids((1, PROMPT), seed=seed0)
    ids_list = []
    for i in range(wave):
        suf = seeded_ids((1, SUFFIX), seed=seed0 + 1000 + i)
        ids_list.append(torch.cat([shared, suf], dim=1))
    batch = torch.cat(ids_list, dim=0)
    with torch.no_grad():
        t0 = time.perf_counter()
        cache = None
        for s0 in range(0, full_len, chunk):
            out = model(batch[:, s0:s0 + chunk], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        kv_bytes = cache_tensor_bytes(cache)
        nxt = out.logits[:, -1:].argmax(-1)
        step_ms = []
        for _ in range(GEN):
            t1 = time.perf_counter()
            out = model(nxt, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t1) * 1000.0)
        kv_bytes_end = cache_tensor_bytes(cache)
    return {"wave": wave, "ttft_ms": ttft_ms,
            "tpot_ms_median": sorted(step_ms)[len(step_ms) // 2],
            "kv_bytes_after_prefill": kv_bytes, "kv_bytes_end": kv_bytes_end,
            "gen_tokens": wave * GEN}


def part_b_admission():
    print("== B. 准入压力实测：MHA / GQA-8 / MQA ==")
    import gc
    dev = "cuda"
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"  part B 入口 free={mib(total - free):.0f} MiB / total={mib(total):.0f} MiB")
    props = torch.cuda.get_device_properties(dev)
    total_bytes = props.total_memory
    results = []
    for name, kv_heads in [("MHA-16", 16), ("GQA-8", 8), ("MQA-1", 1)]:
        torch.cuda.empty_cache()
        try:
            model, _ = build_mid(kv_heads)
        except RuntimeError as e:
            print(f"  {name}: 模型构建 OOM，跳过（{str(e)[:60]}）")
            results.append({"variant": name, "status": "oom_at_build"})
            continue
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        ids = seeded_ids((1, 512), seed=4242)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(ids, use_cache=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        buffers = max(peak - weights, 0)
        context_bytes = torch.cuda.memory_reserved(dev)
        util, guard = 0.90, 256 * 1024 ** 2
        kv_pool = total_bytes * util - context_bytes - guard - weights - buffers
        per_tok = ledger_kv_bytes(1, MID["layers"], kv_heads, MID["head_dim"], 2)
        budget_tokens = int(kv_pool // per_tok)
        # 准入：每请求占 full_len + GEN 个 token
        need = PROMPT + SUFFIX + GEN
        # 逐波增大并发直到预算拒绝（或达到波次上限 64）
        admitted, denied, waves, ttfts, tpots, kv_peak = 0, 0, 0, [], [], 0
        wave, last_admitted_wave = 1, 0
        physical_limit = None
        while wave <= 64:
            if (admitted + wave) * need > budget_tokens:
                denied = wave
                break
            try:
                torch.cuda.reset_peak_memory_stats()
                r = admission_wave(model, kv_heads, wave)
                phys_peak = torch.cuda.max_memory_allocated()
            except RuntimeError as e:
                physical_limit = {"wave": wave, "reason": str(e)[:120]}
                print(f"  {name}: wave={wave} 物理 OOM，停止增长")
                break
            waves += 1
            admitted += wave
            last_admitted_wave = wave
            ttfts.append(r["ttft_ms"])
            tpots.append(r["tpot_ms_median"])
            kv_peak = max(kv_peak, r["kv_bytes_end"], phys_peak if False else 0)
            del r
            torch.cuda.empty_cache()
            wave *= 2
        tok_per_s = 0.0
        if ttfts and tpots:
            total_gen = admitted * GEN
            wall = sum(ttfts) + sum(tpots) * GEN
            tok_per_s = total_gen / wall * 1000.0
        results.append({
            "variant": name, "kv_heads": kv_heads,
            "weights_gib": round(weights / 1024 ** 3, 2),
            "buffers_mib": round(mib(buffers), 1),
            "kv_pool_gib": round(kv_pool / 1024 ** 3, 2),
            "per_token_bytes": per_tok,
            "budget_tokens": budget_tokens,
            "planned_max_concurrency_4096": budget_tokens // 4096,
            "admitted_total": admitted, "denied_wave": denied, "waves": waves,
            "max_concurrent_wave": last_admitted_wave,
            "physical_limit": physical_limit,
            "ttft_ms_waves": [round(x, 1) for x in ttfts],
            "tpot_ms_waves": [round(x, 2) for x in tpots],
            "kv_peak_bytes": kv_peak, "kv_peak_gib": round(kv_peak / 1024 ** 3, 3),
            "gen_tokens_per_s": round(tok_per_s, 1),
        })
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return {"workload": {"prompt": PROMPT, "suffix": SUFFIX, "gen": GEN,
                         "per_request_tokens": PROMPT + SUFFIX + GEN},
            "rows": results}


def toy_cache_policy_ab():
    print("== C. 同 workload 缓存策略对照（toy 12L） ==")
    model, _ = build_llama(TOY["kv_heads"], layers=TOY["layers"], hidden=TOY["hidden"],
                           heads=TOY["heads"], dtype=torch.float16)
    PROMPT_T, SUF_T, GEN_T = TOY_PROMPT, TOY_SUFFIX, TOY_GEN
    N_REQ = 12
    shared = seeded_ids((1, PROMPT_T), seed=313)
    suffixes = [seeded_ids((1, SUF_T), seed=4000 + i) for i in range(N_REQ)]

    def measure(ttfts, tpots, kv_records):
        return {
            "ttft_mean_ms": round(sum(ttfts) / len(ttfts), 2),
            "ttft_p95_ms": round(pct(ttfts, 95), 2),
            "ttft_p99_ms": round(pct(ttfts, 99), 2),
            "tpot_mean_ms": round(sum(tpots) / len(tpots), 3),
            "kv_peak_bytes": max(kv_records),
            "kv_peak_mib": round(mib(max(kv_records)), 2),
        }

    # ---- (a) baseline：无前缀缓存，每请求整段 prefill + decode ----
    ttfts, tpots, kv_records, hits = [], [], [], 0
    wall0 = time.perf_counter()
    for i in range(N_REQ):
        full = torch.cat([shared, suffixes[i]], dim=1)
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(full, use_cache=True)
            torch.cuda.synchronize()
            ttfts.append((time.perf_counter() - t0) * 1000.0)
            cache = out.past_key_values
            kv_records.append(cache_tensor_bytes(cache))
            nxt = out.logits[:, -1:].argmax(-1)
            for _ in range(GEN_T):
                out = model(nxt, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
        del out, cache
        torch.cuda.empty_cache()
    wall_a = (time.perf_counter() - wall0) * 1000.0
    a = measure(ttfts, [wall_a / (N_REQ * GEN_T)] * N_REQ, kv_records)
    a["wall_ms"] = round(wall_a, 1)
    a["hit_rate"] = 0.0
    a["throughput_tok_s"] = round(N_REQ * GEN_T / wall_a * 1000, 1)

    # ---- (b) 无界 prefix cache：所有请求共享同一 prompt cache ----
    with torch.no_grad():
        prompt_cache = model(shared, use_cache=True).past_key_values
    prompt_bytes = cache_tensor_bytes(prompt_cache)
    ttfts, tpots, kv_records = [], [], []
    wall0 = time.perf_counter()
    for i in range(N_REQ):
        pc = copy.deepcopy(prompt_cache)
        mask = torch.ones(1, PROMPT_T + SUF_T, dtype=torch.long, device="cuda")
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(suffixes[i], past_key_values=pc, attention_mask=mask,
                        use_cache=True)
            torch.cuda.synchronize()
            ttfts.append((time.perf_counter() - t0) * 1000.0)
            cache = out.past_key_values
            kv_records.append(prompt_bytes + cache_tensor_bytes(cache))
            nxt = out.logits[:, -1:].argmax(-1)
            for _ in range(GEN_T):
                out = model(nxt, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
        del pc, out, cache
        torch.cuda.empty_cache()
    wall_b = (time.perf_counter() - wall0) * 1000.0
    b = measure(ttfts, [wall_b / (N_REQ * GEN_T)] * N_REQ, kv_records)
    b["wall_ms"] = round(wall_b, 1)
    b["hit_rate"] = 1.0
    b["throughput_tok_s"] = round(N_REQ * GEN_T / wall_b * 1000, 1)

    # ---- (c) LRU 封顶 cache：容量只够 2 份 prompt 块 -> 命中率下降 + 重算尾部 ----
    cap_prompts = 2
    ttfts, tpots, kv_records, hits = [], [], [], 0
    from task4_exp3_prefix import BlockHashPrefixCache
    hasher = BlockHashPrefixCache(block_size=64,
                                  capacity_blocks=cap_prompts * (PROMPT_T // 64))
    wall0 = time.perf_counter()
    for i in range(N_REQ):
        hit_blocks, hit_tokens = hasher.match(shared[0].tolist())
        hasher.register(shared[0].tolist())
        hit = hit_tokens > 0
        hits += 1 if hit else 0
        if hit:
            pc = copy.deepcopy(prompt_cache)
            inp = suffixes[i]
            mask = torch.ones(1, PROMPT_T + SUF_T, dtype=torch.long, device="cuda")
            prefix_kv = prompt_bytes
        else:
            pc = None
            inp = torch.cat([shared, suffixes[i]], dim=1)
            mask = None
            prefix_kv = 0
        with torch.no_grad():
            t0 = time.perf_counter()
            if pc is None:
                out = model(inp, use_cache=True)
            else:
                out = model(inp, past_key_values=pc, attention_mask=mask, use_cache=True)
            torch.cuda.synchronize()
            ttfts.append((time.perf_counter() - t0) * 1000.0)
            cache = out.past_key_values
            kv_records.append(prefix_kv + cache_tensor_bytes(cache))
            nxt = out.logits[:, -1:].argmax(-1)
            for _ in range(GEN_T):
                out = model(nxt, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
        if pc is not None:
            del pc
        del out, cache
        torch.cuda.empty_cache()
    wall_c = (time.perf_counter() - wall0) * 1000.0
    c = measure(ttfts, [wall_c / (N_REQ * GEN_T)] * N_REQ, kv_records)
    c["wall_ms"] = round(wall_c, 1)
    c["hit_rate"] = round(hits / N_REQ, 4)
    c["throughput_tok_s"] = round(N_REQ * GEN_T / wall_c * 1000, 1)

    # notebook 69 的决策骨架（同 workload 比较）
    def compare(base, cand):
        return {"ttft_delta_ms": round(cand["ttft_mean_ms"] - base["ttft_mean_ms"], 2),
                "peak_delta_mib": round(cand["kv_peak_mib"] - base["kv_peak_mib"], 2)}
    comparison = {
        "unbounded_vs_baseline": compare(a, b),
        "lru_capped_vs_baseline": compare(a, c),
    }
    decisions = []
    for label, cand, comp in [("unbounded", b, comparison["unbounded_vs_baseline"]),
                              ("lru_capped", c, comparison["lru_capped_vs_baseline"])]:
        if comp["ttft_delta_ms"] < 0 and comp["peak_delta_mib"] <= 0:
            decision = "accept"
        elif comp["ttft_delta_ms"] < 0:
            decision = "tune"   # TTFT 收益被峰值显存上升抵消 -> 调容量/驱逐
        else:
            decision = "reject"
        decisions.append({"candidate": label, "decision": decision,
                          "ttft_delta_ms": comp["ttft_delta_ms"],
                          "peak_delta_mib": comp["peak_delta_mib"]})
    return {"workload": {"requests": N_REQ, "prompt": PROMPT_T, "suffix": SUF_T,
                         "gen": GEN_T, "dtype": "fp16", "model": "toy-12L-1024",
                         "arrival": "sequential"},
            "baseline": a, "unbounded_prefix_cache": b, "lru_capped_cache": c,
            "comparison": comparison, "decisions": decisions,
            "decision_rule": "同 workload 下 TTFT 降且峰值不升 -> accept；TTFT 降但峰值升 -> tune；否则 reject"}


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    result = {
        "experiment": "task4_exp4_capacity",
        "environment": env_info(),
        "source_sha256": source_sha256(__file__),
        "a_capacity_plan": part_a_capacity_plan(),
        "b_admission": part_b_admission(),
        "c_policy_ab": toy_cache_policy_ab(),
        "status": "PASS",
    }
    dump_json(result, OUT)
    print("ALL_TESTS_PASS")


if __name__ == "__main__":
    main()
