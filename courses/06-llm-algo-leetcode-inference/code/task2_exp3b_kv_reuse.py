# -*- coding: utf-8 -*-
"""Task2 实验 3b：KV Cache 的收益边界——缓存复用 vs 每步重算整段上下文。

用与实验 1 相同的合成 LLaMA 骨架（12 层 / hidden 1024 / GQA 2 个 KV 头），
在长上下文下比较「带缓存 decode」与「不带缓存、每步重算」的 TPOT 与峰值显存。
"""
import json
import time

import torch
from transformers import LlamaConfig, LlamaForCausalLM

OUT = "task2_exp3b_kv_reuse.json"
DEV = "cuda"
DTYPE = torch.float16


def build_model(num_kv_heads=2, layers=12, hidden=1024, heads=8):
    config = LlamaConfig(
        vocab_size=32000, hidden_size=hidden, intermediate_size=hidden * 3,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=num_kv_heads, max_position_embeddings=16384,
        torch_dtype=DTYPE)
    return LlamaForCausalLM(config).to(DEV, dtype=DTYPE).eval(), config


@torch.no_grad()
def run(model, batch, prompt_len, gen_len, use_cache, warmup=1, repeats=2):
    ids = torch.randint(100, 30000, (batch, prompt_len), device=DEV)
    stats = []
    for _ in range(warmup + repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_cache:
            out = model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
        else:
            full = ids
            out = model(full, use_cache=False)
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(gen_len - 1):
            if use_cache:
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
            else:
                full = torch.cat([full, nxt], dim=1)
                out = model(full, use_cache=False)
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        stats.append({
            "ttft_ms": (t1 - t0) * 1000,
            "tpot_ms": (t2 - t1) * 1000 / max(1, gen_len - 1),
            "e2e_ms": (t2 - t0) * 1000,
            "peak_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "tok_s": batch * gen_len / (t2 - t0),
        })
        del out
    best = sorted(stats[warmup:], key=lambda r: r["e2e_ms"])[len(stats[warmup:]) // 2]
    return {k: round(v, 3) for k, v in best.items()}


def main():
    model, config = build_model()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    report = {"model": "synthetic LLaMA 12L/hidden1024/8 heads/GQA-2 KV heads",
              "dtype": "float16", "gen_len": 16, "rows": []}
    for batch, prompt_len in ((8, 512), (8, 2048), (16, 4096)):
        cached = run(model, batch, prompt_len, 16, use_cache=True)
        recompute = run(model, batch, prompt_len, 16, use_cache=False)
        row = {
            "batch": batch, "prompt_len": prompt_len,
            "cached": cached, "recompute": recompute,
            "tpot_speedup": round(recompute["tpot_ms"] / cached["tpot_ms"], 2),
            "e2e_speedup": round(recompute["e2e_ms"] / cached["e2e_ms"], 2),
            "peak_mem_delta_mib": round(cached["peak_mib"] - recompute["peak_mib"], 2),
        }
        report["rows"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
