# -*- coding: utf-8 -*-
"""Task2 实验 3：TTFT / TPOT / 吞吐 / 峰值显存 / 端到端延迟的统一口径（4.3(3)）。

同一模型（gpt2）、同一 dtype、同一 warmup、同一 workload 固定项，
只改变 prompt 长度与 batch size，并额外给出「有/无 KV Cache」对照。
每个指标都记录测量边界（prefill 段 / decode 段 / 全流程）。
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.expanduser("~/local_models/gpt2")
OUT = "task2_exp3_metrics.json"
DEV = "cuda"


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    return tokenizer, model.to(DEV).eval()


@torch.no_grad()
def run_workload(model, batch, prompt_len, gen_len, use_cache=True, warmup=2, repeats=3):
    """一次完整请求：prefill -> 逐 token decode。返回分段计时与峰值显存。"""
    ids = torch.randint(100, 20000, (batch, prompt_len), device=DEV)
    timings = []
    for _ in range(warmup + repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        if use_cache:
            out = model(ids, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t_first = time.perf_counter()
            for _ in range(gen_len - 1):
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t_end = time.perf_counter()
        else:
            full = ids
            out = model(full, use_cache=False)
            nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t_first = time.perf_counter()
            for _ in range(gen_len - 1):
                full = torch.cat([full, nxt], dim=1)  # 无缓存：每步重算整段上下文
                out = model(full, use_cache=False)
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t_end = time.perf_counter()
        timings.append({
            "ttft_ms": (t_first - t_start) * 1000.0,
            "decode_ms": (t_end - t_first) * 1000.0,
            "e2e_ms": (t_end - t_start) * 1000.0,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        })
        del out
    best = sorted(timings[warmup:], key=lambda r: r["e2e_ms"])[len(timings[warmup:]) // 2]
    best["tpot_ms"] = best["decode_ms"] / max(1, gen_len - 1)
    best["output_tokens"] = batch * gen_len
    best["throughput_tok_s"] = batch * gen_len / (best["e2e_ms"] / 1000.0)
    best["decode_throughput_tok_s"] = batch * (gen_len - 1) / (best["decode_ms"] / 1000.0)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}


def main():
    tokenizer, model = load()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    report = {"gpu": torch.cuda.get_device_name(0), "model": "gpt2 (124M)", "dtype": "float16",
              "gen_len": 32, "prompt_sweep": [], "batch_sweep": [], "cache_off": []}

    # gpt2 的 max_position_embeddings=1024，prefill + decode 必须留出 decode 的位置
    for prompt_len in (32, 128, 512, 768):
        r = run_workload(model, batch=1, prompt_len=prompt_len, gen_len=32)
        r["prompt_len"] = prompt_len
        r["batch"] = 1
        report["prompt_sweep"].append(r)
        print("prompt", prompt_len, r, flush=True)

    for batch in (1, 4, 16, 32):
        r = run_workload(model, batch=batch, prompt_len=256, gen_len=32)
        r["prompt_len"] = 256
        r["batch"] = batch
        report["batch_sweep"].append(r)
        print("batch", batch, r, flush=True)

    for prompt_len in (256, 512):
        r = run_workload(model, batch=1, prompt_len=prompt_len, gen_len=32, use_cache=False)
        r["prompt_len"] = prompt_len
        r["batch"] = 1
        r["note"] = "无 KV Cache：每步重算整段上下文"
        report["cache_off"].append(r)
        print("nocache", prompt_len, r, flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
