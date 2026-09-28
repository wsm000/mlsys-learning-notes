# -*- coding: utf-8 -*-
"""Task4 公共模块：KV 账本公式、模型配置、计时与证据输出。

所有实验共享同一份账本定义与同一套固定 workload 字段，保证 4.3 的"共同口径"。
账本公式（与教程 Part01·11 一致）：

    KV bytes = 2(K,V) * layers * kv_heads * head_dim * tokens * dtype_bytes

MLA proxy（latent 表示，DeepSeek 风格）每 token 每层存 (kv_lora_rank + qk_rope) 个元素。
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def ledger_kv_bytes(tokens, layers, kv_heads, head_dim, dtype_bytes, batch=1):
    """标准 MHA/GQA/MQA 的 KV Cache 字节账本。"""
    return 2 * layers * kv_heads * head_dim * tokens * dtype_bytes * batch


def mla_kv_bytes(tokens, layers, dtype_bytes, kv_lora_rank=512, qk_rope=64, batch=1):
    """MLA latent 表示的 KV 字节账本（proxy，不等于真实 MLA kernel 收益）。"""
    return (kv_lora_rank + qk_rope) * tokens * layers * dtype_bytes * batch


def per_token_bytes(layers, kv_heads, head_dim, dtype_bytes):
    return ledger_kv_bytes(1, layers, kv_heads, head_dim, dtype_bytes)


def mib(x):
    return x / 1024 ** 2


def gib(x):
    return x / 1024 ** 3


def env_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "total_memory_mib": round(props.total_memory / 1024 ** 2, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[json] {path}")


def sync_time_ms(fn, warmup=2, iters=5):
    """CUDA 边界计时：warmup + synchronize + 多次取中位数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def collect_cache_tensors(obj, out, depth=0):
    """收集 HF Cache 对象中的全部张量（按 data_ptr 去重）。"""
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif depth < 6:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                collect_cache_tensors(item, out, depth + 1)
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                collect_cache_tensors(value, out, depth + 1)


def cache_tensor_bytes(cache):
    tensors = []
    collect_cache_tensors(cache, tensors)
    seen, total = set(), 0
    for t in tensors:
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total


def build_llama(num_kv_heads, layers=12, hidden=1024, heads=8, vocab=32000,
                intermediate=2816, dtype=torch.float16, device="cuda",
                max_position=8192):
    """构造与 Task2/推理线一致的 LLaMA 玩具模型；只改 num_key_value_heads。

    以 dtype 为默认 dtype 在 CPU 上构建后再搬到 device：避免 CPU fp32 副本触发
    系统 OOM（7B 级 fp32 约 29GB，超过 vm-60 内存）。
    """
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=num_kv_heads, max_position_embeddings=max_position,
        torch_dtype=dtype,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("cpu"):
            model = LlamaForCausalLM(config)
        model = model.to(device).eval()
    finally:
        torch.set_default_dtype(prev)
    return model, config


def seeded_ids(shape, seed, low=0, high=32000, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(low, high, shape, generator=g)
    return ids.to(device)


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [ok] {msg}")
