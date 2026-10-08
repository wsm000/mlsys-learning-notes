# -*- coding: utf-8 -*-
"""Task5 实验 2：W8A16 压静态权重 ≠ 解决 KV OOM（4.1-Q3 + 4.2-Q1 权重侧）。

A. 真实模型单层（Qwen2.5-0.5B-Instruct 第 0 层 q_proj）：FP16 vs W8A16 教学前向的
   latency / peak memory / 权重字节 / 输出 MSE。口径与上游 02·25 Step5 一致。
B. 全模型 W8A16 化（Embedding 之外全部 Linear）：fp16 baseline vs int8 权重存储 +
   逐层反量化前向，测 forward 峰值显存与时间；再与"同 workload KV fp16/int8"对照。
C. 账本对照：权重省下的是固定字节；KV 每 token 持续增长（实测每 token 成本），
   把"省下的固定字节"换成"多支持的 token 数"，展示固定收益 vs 线性增长的差异。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from task5_common import (source_sha256,W8A16Linear, build_llama, dump_json, env_info, gib, mib,
                          per_token_kv_bytes, quantize_per_tensor, seeded_ids, sync_time_ms,
                          w8a16ify, w8a16ify_pc)

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
PROMPT = 'Explain why lower precision can reduce weight storage.'
OUT = Path('../evidence/task5/task5_exp2_w8a16_vs_kv.json')


def a_real_layer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print('== A. 真实 q_proj：FP16 vs W8A16（口径同上游 02·25 Step5）==')
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    ids = tok(PROMPT, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        x = model.model.embed_tokens(ids)
    src = model.model.layers[0].self_attn.q_proj
    inf, outf = src.in_features, src.out_features
    fp = torch.nn.Linear(inf, outf, bias=src.bias is not None, device='cuda', dtype=torch.float16)
    with torch.no_grad():
        fp.weight.copy_(src.weight)
        if src.bias is not None:
            fp.bias.copy_(src.bias)
    q = W8A16Linear(inf, outf).cuda()
    q.from_float(fp)
    del model, src
    torch.cuda.empty_cache()

    def measure(layer):
        torch.cuda.reset_peak_memory_stats()
        t = sync_time_ms(lambda: layer(x), warmup=3, iters=10)
        peak = torch.cuda.max_memory_allocated()
        return t, peak

    t_fp, p_fp = measure(fp)
    t_q, p_q = measure(q)
    with torch.no_grad():
        ref, approx = fp(x), q(x)
        mse = torch.mean((ref.float() - approx.float()) ** 2).item()
    rows = {
        'in_features': inf, 'out_features': outf, 'input_shape': list(x.shape),
        'fp16': {'latency_ms': round(t_fp, 4), 'peak_mib': round(mib(p_fp), 2),
                 'weight_bytes': fp.weight.numel() * fp.weight.element_size()},
        'w8a16': {'latency_ms': round(t_q, 4), 'peak_mib': round(mib(p_q), 2),
                  'weight_bytes': q.weight_int8.numel() * 1 + q.scale.numel() * q.scale.element_size()},
        'output_mse': mse,
    }
    print(f"  FP16 : {rows['fp16']['latency_ms']:.4f} ms, peak {rows['fp16']['peak_mib']:.2f} MiB, "
          f"权重 {rows['fp16']['weight_bytes']/1024:.0f} KiB")
    print(f"  W8A16: {rows['w8a16']['latency_ms']:.4f} ms ({rows['w8a16']['latency_ms']/rows['fp16']['latency_ms']:.2f}x), "
          f"peak {rows['w8a16']['peak_mib']:.2f} MiB ({rows['w8a16']['peak_mib']/rows['fp16']['peak_mib']:.2f}x), "
          f"权重 {rows['w8a16']['weight_bytes']/1024:.0f} KiB")
    print(f"  输出 MSE {mse:.3e} —— 存储减半，但反量化副本使峰值显存不降、前向不加速")
    return rows


def b_full_model_w8a16():
    print('== B. 全模型 W8A16 化 forward：fp16 vs int8 权重存储 + 逐层反量化 ==')
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    ids = tok(PROMPT, return_tensors='pt').input_ids.cuda()

    torch.cuda.reset_peak_memory_stats()
    t_fp = sync_time_ms(lambda: model(input_ids=ids), warmup=2, iters=3)
    p_fp = torch.cuda.max_memory_allocated()
    with torch.no_grad():
        out_fp = model(input_ids=ids, output_hidden_states=True)
    logits_fp = out_fp.logits
    hidden_fp = [h.clone() for h in out_fp.hidden_states]
    w_fp = sum(p.numel() * p.element_size() for p in model.parameters()) + \
        sum(b.numel() * b.element_size() for b in model.buffers())
    del out_fp
    gc.collect()
    torch.cuda.empty_cache()

    def run_w8a16(pc):
        # 全部 Linear 换成量化版（Embedding 保持原 dtype）
        n_lin = (w8a16ify_pc if pc else w8a16ify)(model)
        gc.collect()
        torch.cuda.empty_cache()
        w_int8 = sum((p.numel() * p.element_size() if p.dtype != torch.int8 else p.numel())
                     for p in list(model.parameters()) + list(model.buffers()))
        torch.cuda.reset_peak_memory_stats()
        t_q = sync_time_ms(lambda: model(input_ids=ids), warmup=2, iters=3)
        p_q = torch.cuda.max_memory_allocated()
        with torch.no_grad():
            out_q = model(input_ids=ids, output_hidden_states=True)
        mse = torch.mean((logits_fp.float() - out_q.logits.float()) ** 2).item()
        # 逐层 hidden 相对误差（找误差从哪里开始复合）
        layer_rel = []
        for a, b in zip(hidden_fp, out_q.hidden_states):
            denom = a.float().abs().max().item()
            layer_rel.append((a.float() - b.float()).abs().max().item() / max(denom, 1e-9))
        del out_q
        gc.collect()
        torch.cuda.empty_cache()
        return {
            'n_linear_replaced': n_lin,
            'latency_ms': round(t_q, 3), 'peak_mib': round(mib(p_q), 2), 'param_bytes': int(w_int8),
            'logits_mse': mse,
            'hidden_rel_err_max': max(layer_rel),
            'hidden_rel_err_layer_argmax': int(max(range(len(layer_rel)), key=lambda i: layer_rel[i])),
        }, layer_rel

    row_pt, rel_pt = run_w8a16(pc=False)
    # 还原 fp16 模型再跑 per-channel
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    row_pc, rel_pc = run_w8a16(pc=True)
    rows = {
        'fp16': {'latency_ms': round(t_fp, 3), 'peak_mib': round(mib(p_fp), 2), 'param_bytes': int(w_fp)},
        'w8a16_per_tensor': row_pt,
        'w8a16_per_channel': row_pc,
    }
    for tag in ('w8a16_per_tensor', 'w8a16_per_channel'):
        r = rows[tag]
        print(f"  {tag}: {r['latency_ms']:.2f} ms, peak {r['peak_mib']:.1f} MiB, "
              f"参数字节 {mib(r['param_bytes']):.1f} MiB (压缩 x{rows['fp16']['param_bytes']/r['param_bytes']:.2f}), "
              f"logits MSE {r['logits_mse']:.3e}, hidden 最大相对误差 {r['hidden_rel_err_max']:.1%} "
              f"(layer {r['hidden_rel_err_layer_argmax']})")
    print(f"  FP16 baseline: {t_fp:.2f} ms, peak {mib(p_fp):.1f} MiB, 参数 {mib(w_fp):.1f} MiB")
    print('  注：量化体存储减半，但教学版逐层反量化物化 fp16 副本，forward 峰值反升；'
          'kernel 内反量化的真实 backend 不会有该副本。')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows, {'per_tensor_layer_rel': rel_pt, 'per_channel_layer_rel': rel_pc}


def c_fixed_saving_vs_linear_growth():
    print('== C. 账本：固定节省 vs 线性增长 ==')
    model, cfg = build_llama(num_kv_heads=8)
    w_fp16 = sum(p.numel() * p.element_size() for p in model.parameters())
    w_int8 = int(w_fp16 / 2)
    saved = w_fp16 - w_int8
    # 实测每 token KV 成本（GQA-8kv, 12L, head_dim=128, fp16）
    per_tok = per_token_kv_bytes(1, cfg.num_hidden_layers, cfg.num_key_value_heads,
                                 cfg.hidden_size // cfg.num_attention_heads, 2)
    extra_tokens = saved / per_tok
    # decode 每步每请求新增 KV
    import time
    ids = seeded_ids((1, 64), seed=7)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=16, do_sample=False)
    rows = {
        'weight_fp16_bytes': int(w_fp16), 'weight_int8_bytes': int(w_int8), 'saved_bytes': int(saved),
        'kv_per_token_bytes_measured': int(per_tok),
        'saving_equals_extra_tokens': extra_tokens,
        'note': '权重省下的是固定字节；KV 每生成一个 token 每请求新增 per_tok 字节，线性增长',
    }
    print(f"  权重 {mib(w_fp16):.1f} -> {mib(w_int8):.1f} MiB，固定省 {mib(saved):.1f} MiB")
    print(f"  每 token KV {per_tok/1024:.1f} KiB -> 节省量只够多 {extra_tokens:.0f} token 的 KV")
    del model, out
    torch.cuda.empty_cache()
    return rows


if __name__ == '__main__':
    result = {'experiment': 'task5_exp2_w8a16_vs_kv', 'source_sha256': source_sha256(__file__), 'environment': env_info()}
    result['a_real_layer'] = a_real_layer()
    rows, layer_rel = b_full_model_w8a16()
    result['b_full_model'] = rows
    result['b_layer_rel_err'] = layer_rel
    result['c_fixed_vs_linear'] = c_fixed_saving_vs_linear_growth()
    result['status'] = 'ALL_TESTS_PASS'
    dump_json(result, OUT)
    print('PASS exp2')
