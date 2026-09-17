# -*- coding: utf-8 -*-
"""Task2 实验 1：KV Cache 账本与 KV head 配置（MHA / GQA / MQA）对照。

回答 4.1(1)(2) 与 4.3(1)：不同 KV head 配置下的
  - 显存占用（实测 cache 张量字节 vs 账本公式）
  - 缓存利用率（20 GiB 显存预算下能容纳多少并发请求 / 每 token 成本）
  - 生成性能（prefill 延迟、decode 每 token 延迟、吞吐）
固定条件：同一模型骨架（层数/隐藏维/head_dim 相同），只改变 num_key_value_heads。
"""
import json
import time

import torch
from transformers import LlamaConfig, LlamaForCausalLM

OUT = "task2_exp1_kv_ledger.json"
DEV = "cuda"
DTYPE = torch.float16
BUDGET_MIB = 20 * 1024  # 20 GiB 预算（24 GiB 卡留出安全余量）


def collect_tensors(obj, out, depth=0):
    """把 cache 对象里所有张量收集起来（兼容 DynamicCache 的各种实现）。"""
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif depth < 5:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                collect_tensors(item, out, depth + 1)
        elif hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                collect_tensors(value, out, depth + 1)


def cache_bytes(cache):
    tensors = []
    collect_tensors(cache, tensors)
    seen, total = set(), 0
    for t in tensors:
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


def build_model(num_kv_heads, layers=12, hidden=1024, heads=8):
    config = LlamaConfig(
        vocab_size=32000, hidden_size=hidden, intermediate_size=hidden * 3,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=num_kv_heads, max_position_embeddings=8192,
        torch_dtype=DTYPE,
    )
    model = LlamaForCausalLM(config).to(DEV, dtype=DTYPE).eval()
    return model, config


def ledger_bytes(seq_len, layers, batch, kv_heads, head_dim, dtype_bytes=2):
    """账本公式：2(K,V) x L x B x H_kv x D x S x dtype_bytes。"""
    return 2 * layers * batch * kv_heads * head_dim * seq_len * dtype_bytes


def measure_cache(model, batch, seq_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, 1000, (batch, seq_len), device=DEV)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache_b = cache_bytes(out.past_key_values)
    del out
    torch.cuda.empty_cache()
    return cache_b


def measure_decode(model, batch, seq_len, steps=24, warmup=4):
    ids = torch.randint(0, 1000, (batch, seq_len), device=DEV)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        # 预热
        for _ in range(warmup):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed / steps * 1000.0, batch * steps / elapsed


def measure_prefill(model, batch, seq_len, iters=5):
    ids = torch.randint(0, 1000, (batch, seq_len), device=DEV)
    with torch.no_grad():
        model(ids, use_cache=True)  # warmup
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(ids, use_cache=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main():
    print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    props = torch.cuda.get_device_properties(0)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "total_memory_mib": round(props.total_memory / 1024 ** 2, 1),
        "dtype": "float16",
        "budget_mib": BUDGET_MIB,
        "ledger_formula": "2 * L * B * H_kv * D * S * dtype_bytes",
        "head_configs": {},
        "seq_scan": [],
        "batch_scan": [],
        "theoretical_70b_style": [],
    }
    head_configs = [("MHA", 8), ("GQA", 2), ("MQA", 1)]
    models = {}
    for name, kv_heads in head_configs:
        model, config = build_model(kv_heads)
        models[name] = (model, config)
        head_dim = config.hidden_size // config.num_attention_heads
        params = sum(p.numel() for p in model.parameters())
        # 1) head 配置对照：固定 S=4096, B=1
        measured = measure_cache(model, 1, 4096)
        expected = ledger_bytes(4096, config.num_hidden_layers, 1, kv_heads, head_dim)
        per_token = measured / 4096
        simultaneous = int(BUDGET_MIB * 1024 ** 2 // measured) if measured else 0
        report["head_configs"][name] = {
            "num_kv_heads": kv_heads,
            "num_query_heads": config.num_attention_heads,
            "head_dim": head_dim,
            "layers": config.num_hidden_layers,
            "params_million": round(params / 1e6, 1),
            "cache_mib_at_S4096_B1": round(measured / 1024 ** 2, 2),
            "ledger_mib": round(expected / 1024 ** 2, 2),
            "ledger_match": abs(measured - expected) / expected < 0.02,
            "cache_kib_per_token": round(per_token / 1024, 2),
            "concurrent_requests_in_budget": simultaneous,
            "budget_utilization_ratio": round(measured / (BUDGET_MIB * 1024 ** 2), 4),
        }
        print(name, report["head_configs"][name], flush=True)

    # 2) 序列扫描（GQA 配置）：验证线性增长
    model, config = models["GQA"]
    head_dim = config.hidden_size // config.num_attention_heads
    for seq in (512, 1024, 2048, 4096, 8192):
        measured = measure_cache(model, 1, seq)
        expected = ledger_bytes(seq, config.num_hidden_layers, 1, config.num_key_value_heads, head_dim)
        report["seq_scan"].append({
            "seq_len": seq,
            "measured_mib": round(measured / 1024 ** 2, 3),
            "ledger_mib": round(expected / 1024 ** 2, 3),
            "bytes_per_token": round(measured / seq, 1),
        })
        print("seq", seq, report["seq_scan"][-1], flush=True)

    # 3) batch 扫描（GQA 配置）：验证随并发线性增长
    for batch in (1, 2, 4, 8):
        measured = measure_cache(model, batch, 2048)
        expected = ledger_bytes(2048, config.num_hidden_layers, batch, config.num_key_value_heads, head_dim)
        report["batch_scan"].append({
            "batch_size": batch,
            "measured_mib": round(measured / 1024 ** 2, 3),
            "ledger_mib": round(expected / 1024 ** 2, 3),
        })
        print("batch", batch, report["batch_scan"][-1], flush=True)

    # 4) 三个 head 配置在固定 workload 下的性能
    for name, _ in head_configs:
        model, config = models[name]
        prefill_ms = measure_prefill(model, batch=8, seq_len=1024)
        per_token_ms, throughput = measure_decode(model, batch=8, seq_len=1024)
        report["head_configs"][name].update({
            "prefill_ms_B8_S1024": round(prefill_ms, 3),
            "decode_ms_per_step_B8_S1024": round(per_token_ms, 4),
            "decode_throughput_tok_s_B8_S1024": round(throughput, 2),
            "decode_ms_per_token_per_request": round(per_token_ms / 8, 4),
        })
        print(name, "perf", report["head_configs"][name], flush=True)

    # 5) 70B 级配置的纯账本（不实测）
    for name, kv_heads in (("MHA", 64), ("GQA", 8), ("MQA", 1), ("MLA-latent-512", None)):
        if kv_heads is None:
            # MLA 近似：每 token 每层 latent 维度 512（K/V 合一的压缩表示）
            bytes_per_token = 80 * 512 * 2
        else:
            bytes_per_token = 2 * 80 * kv_heads * 128 * 2
        report["theoretical_70b_style"].append({
            "name": name,
            "bytes_per_token_per_layer_all_layers": bytes_per_token,
            "kib_per_token": round(bytes_per_token / 1024, 2),
            "gib_at_32k_context_batch1": round(bytes_per_token * 32768 / 1024 ** 3, 3),
            "gib_at_32k_context_batch32": round(bytes_per_token * 32768 * 32 / 1024 ** 3, 3),
        })

    # 6) 长上下文 + 大 batch：让 decode 的 KV 读取量真正拉开差距
    for name, _ in head_configs:
        model, config = models[name]
        for seq in (2048, 8192):
            per_token_ms, throughput = measure_decode(model, batch=16, seq_len=seq, steps=12, warmup=3)
            torch.cuda.reset_peak_memory_stats()
            _ = measure_cache(model, 16, seq)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            report.setdefault("long_context_decode", []).append({
                "name": name, "seq_len": seq, "batch_size": 16,
                "decode_ms_per_step": round(per_token_ms, 4),
                "throughput_tok_s": round(throughput, 2),
                "peak_mib": round(peak, 2),
            })
            print("long", name, seq, report["long_context_decode"][-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
