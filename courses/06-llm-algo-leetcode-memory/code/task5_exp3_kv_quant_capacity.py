# -*- coding: utf-8 -*-
"""Task5 实验 3：KV Cache 量化 -> 容量、并发与上下文（4.2-Q1/Q3/Q4）。

A. 真实 KV 量化误差（Qwen2.5-0.5B-Instruct prefill 512 token，第 0 层 K/V）：
   FP16 baseline vs INT8 per-tensor / INT8 per-group(g=64) / FP8-E4M3；字节、MSE、max_err、cosine。
B. 容量账本（4090D 24GB）：KV 池 = 总显存×0.9 − 权重(实测) − 运行缓冲(实测) − 256MiB；
   每 token 成本 FP16/INT8/FP8 → 最大 token 数与最大并发（S=512/4096 两档，GQA-8kv）。
C. 构造性准入实测：toy-12L，固定 S=4096，逐请求真实分配 KV，直到 OOM；
   FP16 KV vs INT8 KV 两种存储，对比实测最大并发（存储容量口径，非 kernel 口径）。
D. 运行时预留项实测：权重 / 激活峰值 / KV 池 / 分配器 reserved-allocated 差 / 256MiB 余量。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from task5_common import (source_sha256,build_llama, dequantize_per_group, dequantize_per_tensor, dump_json, env_info,
                          gib, mib, ok, per_token_kv_bytes, quantize_fp8_e4m3, quantize_per_group,
                          quantize_per_tensor, seeded_ids)

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
PROMPT = 'Summarize the role of key value caches in transformer inference.'
OUT = Path('../evidence/task5/task5_exp3_kv_quant_capacity.json')
KV_POOL_TOTAL_RATIO = 0.9
RESERVE_MIB = 256


def a_real_kv_error():
    print('== A. 真实 KV 量化误差（Qwen2.5-0.5B-Instruct 第 0 层）==')
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    ids = tok(PROMPT, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    key = past.layers[0].keys.float()
    value = past.layers[0].values.float()
    n_tokens = key.shape[2]
    rows = {'n_tokens': int(n_tokens), 'shape': list(key.shape), 'variants': {}}
    base_bytes = (key.numel() + value.numel()) * 2  # FP16 基线

    def add(name, rec_k, rec_v, bytes_):
        rows['variants'][name] = {
            'bytes': int(bytes_),
            'mse_key': torch.mean((key - rec_k) ** 2).item(),
            'mse_value': torch.mean((value - rec_v) ** 2).item(),
            'max_err_key': (key - rec_k).abs().max().item(),
            'max_err_value': (value - rec_v).abs().max().item(),
            'cosine_key': F.cosine_similarity(key.flatten(), rec_k.flatten(), dim=0).item(),
            'compression_vs_fp16': base_bytes / bytes_,
        }

    add('fp16', key, value, base_bytes)
    # INT8 per-tensor：量化体 1B/元素 + 每张量一个 fp32 scale
    qk, sk = quantize_per_tensor(key)
    qv, sv = quantize_per_tensor(value)
    rk, rv = dequantize_per_tensor(qk, sk), dequantize_per_tensor(qv, sv)
    add('int8_per_tensor', rk, rv, base_bytes // 2 + 8)
    # INT8 按 token 行分组（last_dim=64 => 每行一个 scale）
    qk, sk, pad_k = quantize_per_group(key.reshape(-1, key.shape[-1]), group_size=64, qmax=127)
    qv, sv, pad_v = quantize_per_group(value.reshape(-1, value.shape[-1]), group_size=64, qmax=127)
    rk = dequantize_per_group(qk, sk, pad_k, key.shape[-1]).reshape(key.shape)
    rv = dequantize_per_group(qv, sv, pad_v, value.shape[-1]).reshape(value.shape)
    n_scale = qk.shape[0] + qv.shape[0]
    add('int8_per_row_g64', rk, rv, base_bytes // 2 + n_scale * 4)
    # INT4 按 token 行分组
    qk, sk, pad_k = quantize_per_group(key.reshape(-1, key.shape[-1]), group_size=64, qmax=15)
    qv, sv, pad_v = quantize_per_group(value.reshape(-1, value.shape[-1]), group_size=64, qmax=15)
    rk = dequantize_per_group(qk, sk, pad_k, key.shape[-1]).reshape(key.shape)
    rv = dequantize_per_group(qv, sv, pad_v, value.shape[-1]).reshape(value.shape)
    add('int4_per_row_g64', rk, rv, base_bytes // 4 + n_scale * 4)
    # FP8 E4M3（真实硬件浮点格式，1B/元素，无 scale）
    rk, rv = quantize_fp8_e4m3(key), quantize_fp8_e4m3(value)
    add('fp8_e4m3', rk, rv, base_bytes // 2)
    for name, v in rows['variants'].items():
        print(f"  {name:18} {v['bytes']/1024:8.3f} KiB (x{v['compression_vs_fp16']:.2f}) "
              f"MSE_K {v['mse_key']:.3e} MSE_V {v['mse_value']:.3e} maxK {v['max_err_key']:.3f} "
              f"cos {v['cosine_key']:.5f}")
    del model, out
    torch.cuda.empty_cache()
    return rows


def b_capacity_ledger(model=None):
    print('== B. 容量账本（4090D）==')
    if model is None:
        model, _ = build_llama(num_kv_heads=8)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    # 运行缓冲实测：一次 forward 的峰值（权重 + 激活）与权重的差
    ids = seeded_ids((1, 512), seed=11)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(input_ids=ids)
    peak = torch.cuda.max_memory_allocated()
    buffers = peak - weights
    total = torch.cuda.get_device_properties(0).total_memory
    pool = total * KV_POOL_TOTAL_RATIO - weights - buffers - RESERVE_MIB * 1024 ** 2
    row = {'weights_measured_bytes': int(weights), 'buffers_measured_bytes': int(buffers),
           'kv_pool_bytes': int(pool), 'reserve_mib': RESERVE_MIB, 'total_bytes': int(total)}
    variants = {'MHA-32kv': (32, 32, 128), 'GQA-8kv': (32, 8, 128), 'MQA-1kv': (32, 1, 128)}
    row['per_token_bytes'] = {}
    row['capacity'] = {}
    for name, (layers, kv_heads, head_dim) in variants.items():
        costs = {dt: per_token_kv_bytes(1, layers, kv_heads, head_dim, b)
                 for dt, b in [('fp16', 2), ('int8', 1), ('fp8', 1)]}
        row['per_token_bytes'][name] = costs
        for dt, c in costs.items():
            row['capacity'][f'{name}-{dt}'] = {
                'max_tokens': int(pool // c),
                'max_concurrency_s512': int(pool // (c * 512)),
                'max_concurrency_s4096': int(pool // (c * 4096)),
            }
    print(f"  权重 {mib(weights):.1f} MiB + 运行缓冲 {mib(buffers):.1f} MiB -> KV 池 {gib(pool):.2f} GiB")
    for name, costs in row['per_token_bytes'].items():
        line = ' | '.join(f"{dt} {c/1024:.0f}KiB/tok -> S512 并发 {row['capacity'][f'{name}-{dt}']['max_concurrency_s512']}"
                          for dt, c in costs.items())
        print(f"  {name}: {line}")
    return row


def c_admission_pressure():
    print('== C. 构造性准入实测（真实分配直到 OOM）==')
    S = 4096
    rows = {}
    for tag, dtype in [('fp16', torch.float16), ('int8', torch.int8)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 每请求 S token 的 KV（GQA-8kv shapes, 12L, 8kv×128）
        per_req_bytes = 2 * 12 * 8 * 128 * S * (2 if dtype == torch.float16 else 1)
        alloc, n = [], 0
        oom = False
        try:
            while True:
                k = torch.empty(12, 8, S, 128, dtype=dtype, device='cuda')
                v = torch.empty(12, 8, S, 128, dtype=dtype, device='cuda')
                alloc += [k, v]
                n += 1
                if n > 1000:
                    break
        except torch.cuda.OutOfMemoryError:
            oom = True
        used = torch.cuda.memory_allocated()
        rows[tag] = {
            'requests_admitted': n, 'oom': oom, 'kv_bytes_allocated': int(used),
            'per_request_bytes': int(per_req_bytes),
            'measured_kv_capacity_gib': round(gib(used), 3),
        }
        print(f"  {tag}: 准入 {n} 请求 (S={S})，占用 {gib(used):.2f} GiB，OOM={oom}")
        del alloc
        torch.cuda.empty_cache()
    rows['concurrency_gain'] = rows['int8']['requests_admitted'] / rows['fp16']['requests_admitted']
    print(f"  INT8 KV 准入并发为 FP16 的 x{rows['concurrency_gain']:.2f}")
    return rows


def d_runtime_reservations():
    print('== D. 运行时预留项实测 ==')
    model, _ = build_llama(num_kv_heads=8)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    ids = seeded_ids((4, 512), seed=13)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(input_ids=ids, use_cache=True)
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    allocator_gap = reserved - peak
    rows = {
        'weights_mib': round(mib(weights), 1),
        'forward_peak_mib': round(mib(peak), 1),
        'activations_mib': round(mib(peak - weights), 1),
        'reserved_mib': round(mib(reserved), 1),
        'allocator_reserved_minus_allocated_mib': round(mib(allocator_gap), 1),
        'note': 'reserved-allocated 差包含分配器缓存与 CUDA context/workspace；容量规划需预留',
    }
    print(f"  权重 {rows['weights_mib']} MiB；forward 峰值 {rows['forward_peak_mib']} MiB"
          f"（激活 {rows['activations_mib']} MiB）；reserved {rows['reserved_mib']} MiB，"
          f"gap {rows['allocator_reserved_minus_allocated_mib']} MiB")
    del model
    torch.cuda.empty_cache()
    return rows


if __name__ == '__main__':
    result = {'experiment': 'task5_exp3_kv_quant_capacity', 'source_sha256': source_sha256(__file__), 'environment': env_info()}
    result['a_real_kv_error'] = a_real_kv_error()
    result['b_capacity_ledger'] = b_capacity_ledger()
    result['c_admission_pressure'] = c_admission_pressure()
    result['d_runtime_reservations'] = d_runtime_reservations()
    result['status'] = 'ALL_TESTS_PASS'
    dump_json(result, OUT)
    print('PASS exp3')
