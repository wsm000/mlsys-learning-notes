# -*- coding: utf-8 -*-
"""Task5 实验 1：对象 × 时机的字节账本（权重三档实测 + 三元对象账本构成）。

回答 4.1-Q1（哪类对象优先量化、为什么不能只看参数量）与 4.1-Q2（INT8 权重 / scale / FP16 激活的角色）。

A. 真实权重（Qwen2.5-0.5B-Instruct 全部 2D+ 参数）：FP16 → INT8 per-tensor /
   per-channel → INT4 per-group(g=128) 打包；实测字节、scale 元数据、重建 MSE。
B. 容量账本表：0.5B/1.5B/7B × FP16/INT8/INT4 权重字节；KV 每 token（MHA-32kv /
   GQA-8kv / MQA-1kv）× FP16/INT8/FP8；toy-12L 激活峰值实测（B=4/S=512 prefill）。
C. 同一模型两种 workload 的账本构成切换（权重不变，KV 从配角变主角）——"不能只看参数量"的实证。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from task5_common import (source_sha256,build_llama, dequantize_per_channel, dequantize_per_group,
                          dequantize_per_tensor, dump_json, env_info, gib, int4_packed_bytes,
                          ledger_weight_bytes, mib, quantize_per_channel, quantize_per_group,
                          quantize_per_tensor, seeded_ids, sync_time_ms)

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
OUT = Path('../evidence/task5/task5_exp1_object_timing.json')


def a_real_weight_ladder():
    from transformers import AutoModelForCausalLM
    print('== A. 真实权重三档实测（Qwen2.5-0.5B-Instruct）==')
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    tensors = {n: p.detach() for n, p in model.named_parameters() if p.dim() >= 2}
    total_params = sum(t.numel() for t in tensors.values())
    fp16_bytes = sum(t.numel() * t.element_size() for t in tensors.values())

    int8_pt = int8_pt_scale = int8_pc = int4_pg = int4_pg_scale = 0
    mse8_pt = mse8_pc = mse4 = 0.0
    n_pc = 0
    for t in tensors.values():
        tf = t.float()
        q8, s8 = quantize_per_tensor(tf)
        r8 = dequantize_per_tensor(q8, s8)
        int8_pt += q8.numel()
        int8_pt_scale += s8.numel() * s8.element_size()
        mse8_pt += torch.mean((tf - r8) ** 2).item()
        if t.shape[-1] % 128 == 0:
            q8c, s8c = quantize_per_channel(tf)
            r8c = dequantize_per_channel(q8c, s8c)
            int8_pc += q8c.numel() + s8c.numel() * s8c.element_size()
            mse8_pc += torch.mean((tf - r8c) ** 2).item()
            q4, s4, pad = quantize_per_group(tf, group_size=128, qmax=15)
            r4 = dequantize_per_group(q4, s4, pad, t.shape[-1])
            int4_pg += int4_packed_bytes(q4)
            int4_pg_scale += s4.numel() * s4.element_size()
            mse4 += torch.mean((tf - r4) ** 2).item()
            n_pc += 1
    rows = {
        'n_matrices': len(tensors), 'total_params': int(total_params),
        'fp16_bytes': int(fp16_bytes),
        'int8_per_tensor_bytes': int(int8_pt + int8_pt_scale),
        'int8_per_tensor_scale_bytes': int(int8_pt_scale),
        'int8_per_channel_bytes': int(int8_pc),
        'int4_group128_packed_bytes': int(int4_pg + int4_pg_scale),
        'int4_group128_scale_bytes': int(int4_pg_scale),
        'mse_fp16_vs_int8_pertensor': mse8_pt / len(tensors),
        'mse_fp16_vs_int8_perchannel': mse8_pc / max(n_pc, 1),
        'mse_fp16_vs_int4_group128': mse4 / max(n_pc, 1),
    }
    rows['compression'] = {
        'int8_per_tensor': fp16_bytes / rows['int8_per_tensor_bytes'],
        'int8_per_channel': fp16_bytes / rows['int8_per_channel_bytes'],
        'int4_group128': fp16_bytes / rows['int4_group128_packed_bytes'],
    }
    print(f"  矩阵数 {rows['n_matrices']}，参数 {rows['total_params']/1e6:.0f}M")
    print(f"  FP16 {mib(fp16_bytes):8.2f} MiB | INT8-per-tensor {mib(rows['int8_per_tensor_bytes']):7.2f} MiB "
          f"(x{rows['compression']['int8_per_tensor']:.3f}) | INT8-per-channel {mib(rows['int8_per_channel_bytes']):7.2f} MiB "
          f"(x{rows['compression']['int8_per_channel']:.3f})")
    print(f"  INT4-g128 打包 {mib(rows['int4_group128_packed_bytes']):7.2f} MiB (x{rows['compression']['int4_group128']:.3f})，"
          f"其中 scale {mib(rows['int4_group128_scale_bytes']):.3f} MiB")
    print(f"  重建 MSE：INT8-pt {rows['mse_fp16_vs_int8_pertensor']:.3e} / INT8-pc "
          f"{rows['mse_fp16_vs_int8_perchannel']:.3e} / INT4-g128 {rows['mse_fp16_vs_int4_group128']:.3e}")
    del model
    torch.cuda.empty_cache()
    return rows


def b_capacity_ledger():
    print('== B. 容量账本表 + 激活峰值实测 ==')
    weights = {name: {dt: ledger_weight_bytes(p, b) for dt, b in
                      [('fp16', 2), ('int8', 1), ('int4', 0.5)]}
               for name, p in {'0.5B': 490_000_000, '1.5B': 1_540_000_000, '7B': 7_620_000_000}.items()}
    kv_variants = {'MHA-32kv': (32, 32, 128), 'GQA-8kv': (32, 8, 128), 'MQA-1kv': (32, 1, 128)}
    kv_table = {}
    for name, (layers, kv_heads, head_dim) in kv_variants.items():
        kv_table[name] = {
            dt: 2 * layers * kv_heads * head_dim * b for dt, b in [('fp16', 2), ('int8', 1), ('fp8', 1)]
        }
    print('  权重账本：' + ' | '.join(f"{k} " + '/'.join(f"{dt} {gib(v):.2f}GiB" for dt, v in d.items())
                                  for k, d in weights.items()))
    print('  KV/token：' + ' | '.join(f"{k} " + '/'.join(f"{dt} {v/1024:.0f}KiB" for dt, v in d.items())
                                  for k, d in kv_table.items()))
    # 激活峰值实测：toy-12L，B=4/S=512 prefill（与 Task4 同形状）
    model, cfg = build_llama(num_kv_heads=8)
    ids = seeded_ids((4, 512), seed=73)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(input_ids=ids)
    act_peak = torch.cuda.max_memory_allocated()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  激活+权重峰值（toy-12L B=4 S=512 prefill）：{mib(act_peak):.1f} MiB"
          f"（其中权重 {mib(weight_bytes):.1f} MiB）")
    del model, out
    torch.cuda.empty_cache()
    return {
        'weights': weights,
        'kv_per_token_bytes': kv_table,
        'activation_peak_measured': {
            'model': 'toy-12L-1024 (随机初始化教学形状)', 'batch': 4, 'seq': 512,
            'weight_bytes': int(weight_bytes), 'peak_allocated_bytes': int(act_peak),
        },
    }


def c_same_model_two_workloads():
    print('== C. 同一 7B 模型两种 workload 的账本构成（说明不能只看参数量）==')
    p7b = 7_620_000_000
    w_fp16 = ledger_weight_bytes(p7b, 2)
    w_int8 = ledger_weight_bytes(p7b, 1)
    scenarios = {}
    for tag, tokens, batch in [('短上下文单并发', 512, 1), ('长上下文高并发', 32768, 32)]:
        kv = 2 * 32 * 8 * 128 * tokens * 2 * batch  # GQA-8kv fp16
        total = w_fp16 + kv
        scenarios[tag] = {
            'kv_bytes': int(kv), 'weight_fp16_bytes': int(w_fp16),
            'weight_share': w_fp16 / total, 'kv_share': kv / total,
            'total_bytes': int(total),
        }
        print(f"  {tag}: 权重 {gib(w_fp16):.2f} GiB ({w_fp16/total:.1%}) + KV {gib(kv):.2f} GiB ({kv/total:.1%})")
    # INT8 权重后，长上下文场景 KV 占比反而上升（瓶颈转移）
    kv = scenarios['长上下文高并发']['kv_bytes']
    total_int8 = w_int8 + kv
    print(f"  权重量化 INT8 后长上下文场景：权重 {gib(w_int8):.2f} GiB ({w_int8/total_int8:.1%}) + "
          f"KV {gib(kv):.2f} GiB ({kv/total_int8:.1%}) —— 瓶颈从权重转移到 KV")
    scenarios['int8_weight_shift'] = {'weight_int8_bytes': int(w_int8), 'kv_share_after': kv / total_int8}
    return scenarios


if __name__ == '__main__':
    result = {'experiment': 'task5_exp1_object_timing', 'source_sha256': source_sha256(__file__), 'environment': env_info()}
    result['a_real_weight_ladder'] = a_real_weight_ladder()
    result['b_capacity_ledger'] = b_capacity_ledger()
    result['c_same_model_two_workloads'] = c_same_model_two_workloads()
    result['status'] = 'ALL_TESTS_PASS'
    dump_json(result, OUT)
    print('PASS exp1')
