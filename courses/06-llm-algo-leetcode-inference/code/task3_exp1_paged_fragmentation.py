# -*- coding: utf-8 -*-
"""Task3 实验 1：PagedAttention 分页分配 vs 连续预留——碎片账本 + 真实 GPU 分配。

问题：把连续 KV Cache 拆成固定大小 block，解决了连续显存分配中的什么问题？
- CPU 账本：同一请求长度分布下，比较「按最大长度连续预留」与「按 block 分页按需分配」
  的 token 容量、尾块碎片与利用率；并扫描 block_size 看碎片与块表开销的权衡。
- 真实 GPU：把两种策略翻译成同样账本的 fp16 张量分配，用 peak allocated 直接测量。
  账本按 LLaMA-7B 量级（32 层 / 32 KV 头 / head_dim 128 / fp16）折算每 token KV 字节。
"""
import json

import torch

OUT = "task3_exp1_paged_fragmentation.json"
DEV = "cuda"

# LLaMA-7B 量级的每 token KV 字节数：2(K/V) * 32 层 * 32 KV 头 * 128 head_dim * 2 字节
BYTES_PER_TOKEN = 2 * 32 * 32 * 128 * 2
REQUEST_LENGTHS = [17, 33, 65, 129, 255, 511, 1023, 2047]
MAX_RESERVED = 4096  # 连续预留策略：每个请求都按 4096 预留


def cpu_ledger(lengths, block_size, max_reserved):
    paged_tokens = sum((L + block_size - 1) // block_size * block_size for L in lengths)
    contiguous_tokens = len(lengths) * max_reserved
    used = sum(lengths)
    return {
        "requests": len(lengths),
        "used_tokens": used,
        "paged_tokens": paged_tokens,
        "paged_tail_waste_tokens": paged_tokens - used,
        "paged_utilization": round(used / paged_tokens, 4),
        "contiguous_tokens": contiguous_tokens,
        "contiguous_waste_tokens": contiguous_tokens - used,
        "contiguous_utilization": round(used / contiguous_tokens, 4),
        "capacity_saved_tokens": contiguous_tokens - paged_tokens,
        "capacity_saved_ratio": round(1 - paged_tokens / contiguous_tokens, 4),
        "paged_blocks": sum((L + block_size - 1) // block_size for L in lengths),
    }


def gpu_alloc(tokens):
    """按每 token KV 字节数分配 fp16 张量，返回峰值 allocated MiB。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kv = torch.empty(int(tokens * BYTES_PER_TOKEN / 2), dtype=torch.float16, device=DEV)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    del kv
    torch.cuda.empty_cache()
    return round(peak, 2)


def main():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("bytes per KV token:", BYTES_PER_TOKEN, flush=True)
    report = {"bytes_per_token": BYTES_PER_TOKEN,
              "request_lengths": REQUEST_LENGTHS, "max_reserved": MAX_RESERVED,
              "block_size_scan": [], "gpu": {}}

    # 1) block_size 扫描：尾块碎片 vs 块表条目数
    for bs in (8, 16, 32, 64, 128):
        row = cpu_ledger(REQUEST_LENGTHS, bs, MAX_RESERVED)
        row["block_size"] = bs
        report["block_size_scan"].append(row)
        print("block_size", bs, json.dumps(row, ensure_ascii=False), flush=True)

    # 2) 真实 GPU 分配：block_size=16 分页 vs 连续预留
    ledger16 = cpu_ledger(REQUEST_LENGTHS, 16, MAX_RESERVED)
    paged_mib = gpu_alloc(ledger16["paged_tokens"])
    contig_mib = gpu_alloc(ledger16["contiguous_tokens"])
    report["gpu"] = {
        "paged_tokens": ledger16["paged_tokens"], "paged_peak_mib": paged_mib,
        "contiguous_tokens": ledger16["contiguous_tokens"], "contiguous_peak_mib": contig_mib,
        "saved_mib": round(contig_mib - paged_mib, 2),
        "saved_ratio": round(1 - paged_mib / contig_mib, 4),
        "concurrency_gain": round(contig_mib / paged_mib, 2),
    }
    print("GPU alloc:", json.dumps(report["gpu"], ensure_ascii=False), flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
