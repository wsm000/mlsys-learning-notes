# -*- coding: utf-8 -*-
"""Task3 实验 3：真实模型上的 Prefix Cache 对照——prefill 计算量 / TTFT / 峰值显存。

问题：Prefix Cache 命中后哪些计算可以跳过？对 TTFT 的真实改善有多少？
- workload：8 个请求共享一段 system prompt，各带 200 token 独特后缀，生成长度 8。
- G0（cache off）：每个请求都完整 prefill system+200 token。
- G1（prefix cache）：共享前缀只 prefill 一次并常驻缓存，之后每个请求只 prefill
  自己的 200 token 后缀（past_key_values 续接，位置自动接续）；请求结束后把
  后缀+生成部分的 K/V crop 掉、只保留共享前缀——与真实 prefix cache 一致。
- Part A：串行逐请求，扫描 system 长度 {200, 400, 800}，看 TTFT 比值随规模的变化。
- Part B：批量（8 请求一次前向），模拟 serving 的 continuous batching 场景。
- 记录 prefill 实际处理的 token 数（计算量代理）、TTFT、峰值显存；并校验 G0/G1
  首 token 一致（缓存复用不改变输出）。
环境：vm-60 · RTX 4090 D · torch 2.9.1+cu128 · transformers 5.16.1 · gpt2 fp16
"""
import json
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT = "task3_exp3_prefix_cache_gpu.json"
DEV = "cuda"
MODEL_PATH = "/home/simin/local_models/gpt2"
SUFFIX_LEN = 200
GEN_LEN = 8
N_REQ = 8


def build_workload(tok, system_len):
    base = tok("You are a helpful assistant. " * 400, return_tensors="pt").input_ids[0]
    system = base[:system_len].tolist()
    rng = torch.Generator().manual_seed(7)
    reqs = []
    for i in range(N_REQ):
        suffix = torch.randint(50000, 50257, (SUFFIX_LEN,), generator=rng).tolist()
        reqs.append(system + suffix)
    return system, reqs


@torch.no_grad()
def prefill(model, ids, past=None):
    out = model(ids, past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1:]


@torch.no_grad()
def decode(model, first_token, past, gen_len):
    cur, cache = first_token, past
    for _ in range(gen_len - 1):
        out = model(cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
    return int(cur)


def timed_prefill(model, ids, past=None):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    past, logits = prefill(model, ids, past)
    torch.cuda.synchronize()
    return past, logits, (time.perf_counter() - t0) * 1000


def run_sequential(model, reqs, system, use_prefix_cache):
    """串行逐请求；use_prefix_cache=True 时共享前缀只算一次。"""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    prefill_tokens, firsts, ttfts, logits_rows = 0, [], [], []
    system_past = None
    if use_prefix_cache:
        system_past, _, cold = timed_prefill(model, torch.tensor([system], device=DEV))
        prefill_tokens += len(system)
    else:
        cold = None
    system_len_cached = system_past.get_seq_length() if system_past is not None else None

    # 未计量热身请求：消除冷启动/时钟爬升伪影
    if use_prefix_cache:
        warm_suffix = torch.tensor([reqs[0][len(system):]], device=DEV)
        _, w_logits, _ = timed_prefill(model, warm_suffix, system_past)
        decode(model, w_logits.argmax(-1), system_past, GEN_LEN)
        extra = system_past.get_seq_length() - system_len_cached
        if extra > 0:
            system_past.crop(-extra)
    else:
        _, w_logits, _ = timed_prefill(model, torch.tensor([reqs[0]], device=DEV))
        decode(model, w_logits.argmax(-1), system_past, GEN_LEN)

    for ids in reqs:
        if use_prefix_cache:
            suffix = ids[len(system):]
            past, logits, ms = timed_prefill(
                model, torch.tensor([suffix], device=DEV), system_past)
            prefill_tokens += len(suffix)
            ttfts.append(ms)
            firsts.append(decode(model, logits.argmax(-1), past, GEN_LEN))
            logits_rows.append(logits[0, -1].float().cpu())
            extra = system_past.get_seq_length() - system_len_cached
            if extra > 0:
                system_past.crop(-extra)  # 只保留共享前缀
        else:
            past, logits, ms = timed_prefill(model, torch.tensor([ids], device=DEV))
            prefill_tokens += len(ids)
            ttfts.append(ms)
            firsts.append(decode(model, logits.argmax(-1), past, GEN_LEN))
            logits_rows.append(logits[0, -1].float().cpu())
    return {"ttft_ms": sum(ttfts) / len(ttfts), "cold_ttft_ms": cold,
            "prefill_tokens": prefill_tokens,
            "peak_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "first_tokens": firsts, "logits": logits_rows}


def run_batched(model, reqs, system, use_prefix_cache):
    """8 个请求合并成一次前向（continuous batching 的最小模型）。"""
    # 未计量热身
    run_batched_once(model, reqs, system, use_prefix_cache)
    return run_batched_once(model, reqs, system, use_prefix_cache)


def run_batched_once(model, reqs, system, use_prefix_cache):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    batch = torch.tensor(reqs, device=DEV)  # [8, 1000]
    prefill_tokens, cold = 0, None
    if use_prefix_cache:
        system_past, _, cold = timed_prefill(model, torch.tensor([system], device=DEV))
        system_past.batch_repeat_interleave(N_REQ)  # 前缀 KV 广播到整个 batch
        suffix = batch[:, len(system):]
        past, logits, ms = timed_prefill(model, suffix, system_past)
        prefill_tokens += len(system) + suffix.numel()
    else:
        past, logits, ms = timed_prefill(model, batch)
        prefill_tokens += batch.numel()
    firsts = decode_batch(model, logits.argmax(-1), past, GEN_LEN)
    return {"ttft_ms": ms, "cold_ttft_ms": cold, "prefill_tokens": prefill_tokens,
            "peak_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "first_tokens": firsts, "logits": logits[:, -1].float().cpu()}


@torch.no_grad()
def decode_batch(model, first_tokens, past, gen_len):
    cur, cache = first_tokens, past
    for _ in range(gen_len - 1):
        out = model(cur, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
    return [int(t) for t in cur[:, 0]]


@torch.no_grad()
def warmup(model):
    """消除 lazy init / autotune 的冷启动伪影。"""
    ids = torch.randint(0, 50000, (1, 32), device=DEV)
    for _ in range(3):
        past, logits = prefill(model, ids)
        decode(model, logits.argmax(-1), past, 4)
    torch.cuda.synchronize()


def main():
    tok = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEV).eval()
    warmup(model)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    report = {"model": "gpt2 (124M, fp16)", "device": torch.cuda.get_device_name(0),
              "suffix_len": SUFFIX_LEN, "gen_len": GEN_LEN, "requests": N_REQ,
              "sequential_sweep": [], "batched": {}}

    # Part A：串行 + system 长度扫描
    for system_len in (200, 400, 800):
        system, reqs = build_workload(tok, system_len)
        g0 = run_sequential(model, reqs, system, use_prefix_cache=False)
        g1 = run_sequential(model, reqs, system, use_prefix_cache=True)
        row = {
            "system_len": system_len,
            "g0_ttft_ms": round(g0["ttft_ms"], 3),
            "g1_ttft_ms": round(g1["ttft_ms"], 3),
            "g1_cold_ttft_ms": round(g1["cold_ttft_ms"], 3),
            "ttft_speedup": round(g0["ttft_ms"] / g1["ttft_ms"], 3),
            "prefill_token_reduction": round(1 - g1["prefill_tokens"] / g0["prefill_tokens"], 4),
            "g0_peak_mib": round(g0["peak_mib"], 2),
            "g1_peak_mib": round(g1["peak_mib"], 2),
            "top1_agreement": sum(int(a == b) for a, b in zip(g0["first_tokens"], g1["first_tokens"])) / N_REQ,
            "max_logit_abs_diff": round(max((a - b).abs().max().item() for a, b in zip(g0["logits"], g1["logits"])), 5),
            "min_top2_margin_g0": round(min((row.topk(2).values[0] - row.topk(2).values[1]).item() for row in g0["logits"]), 4),
        }
        report["sequential_sweep"].append(row)
        print("seq system", system_len, json.dumps(row, ensure_ascii=False), flush=True)

    # Part B：批量
    system, reqs = build_workload(tok, 800)
    g0b = run_batched(model, reqs, system, use_prefix_cache=False)
    g1b = run_batched(model, reqs, system, use_prefix_cache=True)
    report["batched"] = {
        "system_len": 800,
        "g0_ttft_ms": round(g0b["ttft_ms"], 3),
        "g1_ttft_ms": round(g1b["ttft_ms"], 3),
        "g1_cold_ttft_ms": round(g1b["cold_ttft_ms"], 3),
        "ttft_speedup": round(g0b["ttft_ms"] / g1b["ttft_ms"], 3),
        "prefill_token_reduction": round(1 - g1b["prefill_tokens"] / g0b["prefill_tokens"], 4),
        "g0_peak_mib": round(g0b["peak_mib"], 2),
        "g1_peak_mib": round(g1b["peak_mib"], 2),
        "top1_agreement": sum(int(a == b) for a, b in zip(g0b["first_tokens"], g1b["first_tokens"])) / N_REQ,
        "max_logit_abs_diff": round((g0b["logits"] - g1b["logits"]).abs().max().item(), 5),
        "min_top2_margin_g0": round(min((row.topk(2).values[0] - row.topk(2).values[1]).item() for row in g0b["logits"]), 4),
    }
    print("batched:", json.dumps(report["batched"], ensure_ascii=False), flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
