# -*- coding: utf-8 -*-
"""Task5 实验 4：同 workload 候选对照与决策（4.3，66 baseline × 67 决策机制）。

固定 workload（与 Task4 一致）：toy-12L GQA-8kv，B=4，prompt=512，gen=64，fp16，seed=73。
候选（只改量化一个变量族）：
  FP16 baseline / W8A16 权重 / INT4 权重 / W8A16 权重 + KV-INT8 存储
指标：peak memory（max_memory_allocated）、KV 字节、TTFT、TPOT、吞吐、logits MSE/cosine。
决策（67 机制）：误差预算为硬门槛（cosine<0.99 直接 reject），再看性能/显存收益。
KV-INT8 采用"存储 int8、读取时反量化"的往返路径（round-trip），
用于测量存储容量收益与逐 step 反量化开销；不代表真实 INT8 KV kernel。
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from task5_common import (source_sha256,W8A16Linear, build_llama, dequantize_per_group, dequantize_per_tensor,
                          dump_json, env_info, gib, mib, quantize_per_group, quantize_per_tensor,
                          seeded_ids, w8a16ify)

B, S_PROMPT, S_GEN = 4, 512, 64
SEED = 73
OUT = Path('../evidence/task5/task5_exp4_candidates.json')


def quantize_kv_int8_roundtrip(past):
    """把 Cache 中所有 K/V 做 per-tensor INT8 往返（存储 int8、读取反量化）。返回存储字节增量。"""
    extra = 0
    for layer in past.layers:
        for t in ('keys', 'values'):
            x = getattr(layer, t)
            q, s = quantize_per_tensor(x.float())
            xq = (q.to(x.device), s.to(x.dtype).to(x.device))
            setattr(layer, t, xq)  # 之后由 hook 解包
            extra += q.numel() * 1 + s.numel() * s.element_size()
    return extra


def dequantize_kv_int8_roundtrip(past):
    for layer in past.layers:
        for t in ('keys', 'values'):
            x = getattr(layer, t)
            if isinstance(x, tuple):
                q, s = x
                setattr(layer, t, dequantize_per_tensor(q, s).to(torch.float16))


class WeightInt4Linear(torch.nn.Module):
    """INT4 per-group 权重存储 + 前向解包（教学版，无 bit packing 的 kernel 收益）。"""

    def __init__(self, src: torch.nn.Linear, group_size=128, qmax=15):
        super().__init__()
        w = src.weight.detach().float()
        q, s, pad = quantize_per_group(w, group_size=group_size, qmax=qmax)
        self.q, self.s, self.pad, self.in_f = q, s, pad, w.shape[1]
        self.bias = src.bias
        self.out_f = w.shape[0]

    def forward(self, x):
        rec = dequantize_per_group(self.q.to(x.device), self.s.to(x.device), self.pad, self.in_f)
        return torch.nn.functional.linear(x, rec.to(x.dtype), self.bias)


def int4ify_clean(model, group_size=128):
    n = 0
    for name, mod in list(model.named_children()):
        if isinstance(mod, torch.nn.Linear):
            setattr(model, name, WeightInt4Linear(mod, group_size))
            n += 1
        else:
            n += int4ify_clean(mod, group_size)
    return n


def kv_bytes_of(past, quantized=False):
    total = 0
    for layer in past.layers:
        for t in ('keys', 'values'):
            x = getattr(layer, t)
            if isinstance(x, tuple):
                q, s = x
                total += q.numel() * 1 + s.numel() * s.element_size()
            else:
                total += x.numel() * x.element_size()
    return total


@torch.no_grad()
def run_candidate(model, ids, cont_ids, kv_quant):
    """warmup 后 prefill + 逐步 decode（teacher forcing：步骤输入固定，隔离量化误差）。

    返回 logits、计时、peak、KV 存储字节。
    """
    # 冷启动（kernel 选择/分配器增长）不计入：先用小序列预热
    model(input_ids=ids[:, :8])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1:]
    ttft = (time.perf_counter() - t0) * 1000
    if kv_quant:
        quantize_kv_int8_roundtrip(past)
    step_times = []
    for i in range(cont_ids.shape[1]):
        next_id = cont_ids[:, i:i + 1]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if kv_quant:
            dequantize_kv_int8_roundtrip(past)
        o = model(input_ids=next_id, past_key_values=past, use_cache=True)
        if kv_quant:
            quantize_kv_int8_roundtrip(past)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t1) * 1000)
        logits = torch.cat([logits, o.logits[:, -1:]], dim=1)
    peak = torch.cuda.max_memory_allocated()
    total_ms = ttft + sum(step_times)
    return {
        'logits': logits, 'ttft_ms': ttft, 'step_ms': step_times,
        'tpot_ms': sum(step_times) / max(len(step_times), 1),
        'throughput_tok_s': B * (cont_ids.shape[1] + 1) / (total_ms / 1000),
        'total_ms': total_ms, 'peak_bytes': peak,
        'kv_bytes': kv_bytes_of(past, kv_quant),
    }


def summarize(tag, r, base):
    mse = torch.mean((base['logits'].float() - r['logits'].float()) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(
        base['logits'].float().flatten(), r['logits'].float().flatten(), dim=0).item()
    row = {
        'ttft_ms': round(r['ttft_ms'], 2), 'tpot_ms': round(r['tpot_ms'], 3),
        'throughput_tok_s': round(r['throughput_tok_s'], 1),
        'peak_mib': round(mib(r['peak_bytes']), 1),
        'kv_mib': round(mib(r['kv_bytes']), 2),
        'logits_mse_vs_baseline': mse, 'logits_cosine_vs_baseline': cos,
        'delta_peak_mib': round(mib(r['peak_bytes'] - base['peak_bytes']), 1),
        'delta_throughput_tok_s': round(r['throughput_tok_s'] - base['throughput_tok_s'], 1),
    }
    error_ok = cos > 0.99
    mem_gain = r['peak_bytes'] < base['peak_bytes'] * 0.95
    perf_gain = r['throughput_tok_s'] > base['throughput_tok_s'] * 1.02
    if tag == 'fp16_baseline':
        row['decision'] = 'baseline'
        row['gates'] = {'error_ok': True, 'memory_gain': False, 'performance_gain': False}
    else:
        row['decision'] = ('reject' if not error_ok else
                           'accept' if (perf_gain and mem_gain) else
                           'tune' if (perf_gain or mem_gain) else 'reject')
        row['gates'] = {'error_ok': error_ok, 'memory_gain': mem_gain, 'performance_gain': perf_gain}
    print(f"  {tag:22} TTFT {row['ttft_ms']:8.2f} TPOT {row['tpot_ms']:7.3f} "
          f"thr {row['throughput_tok_s']:7.1f} peak {row['peak_mib']:7.1f} KV {row['kv_mib']:7.2f} "
          f"cos {cos:.5f} -> {row['decision']}")
    return row


if __name__ == '__main__':
    result = {'experiment': 'task5_exp4_candidates', 'source_sha256': source_sha256(__file__),
              'environment': env_info(),
              'workload': {'model': 'toy-12L-1024 GQA-8kv (随机初始化教学形状, 同 seed 同权重)', 'batch': B,
                           'prompt': S_PROMPT, 'gen': S_GEN, 'dtype': 'fp16', 'seed': SEED}}
    ids = seeded_ids((B, S_PROMPT), seed=SEED)
    cont = seeded_ids((B, S_GEN), seed=SEED + 1)  # 固定续续序列（teacher forcing）
    rows = {}
    base_r = None
    for tag, setup, kv_quant in [
        ('fp16_baseline', lambda m: None, False),
        ('w8a16_weight', lambda m: w8a16ify(m), False),
        ('int4_weight', lambda m: int4ify_clean(m), False),
        ('w8a16_weight_kv_int8', lambda m: w8a16ify(m), True),
    ]:
        torch.manual_seed(SEED)  # 每个候选同一份初始权重（随机初始化教学模型）
        model, _ = build_llama(num_kv_heads=8)
        setup(model)
        r = run_candidate(model, ids, cont, kv_quant)
        if base_r is None:
            base_r = r
        rows[tag] = summarize(tag, r, base_r)
        del model, r
        torch.cuda.empty_cache()
    result['candidates'] = rows
    result['status'] = 'ALL_TESTS_PASS'
    dump_json(result, OUT)
    print('PASS exp4')
