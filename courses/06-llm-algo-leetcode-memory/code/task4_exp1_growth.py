# -*- coding: utf-8 -*-
"""Task4 实验1 · 增长：KV Cache 账本的逐维度验证与真实 GPU 实测。

回答 4.1(1)（增长关系与线性维度）与 4.3(3)（表示变化：MHA/GQA/MQA/MLA/量化）。

内容：
  A. 账本公式 vs 真实张量分配（每维逐一验证，精确相等）
  B. 逐维度线性增长（倍乘关系，log2-log2 斜率恰为 1）
  C. 真实 LLaMA 玩具模型：MHA/GQA/MQA 的 cache 字节、decode 每步增长、每 token 成本
  D. MLA latent 表示的显式缓冲账本
  E. FP8 / INT8 KV 的字节节省与数值误差（真实 K/V 张量）
固定条件：vm-60 / RTX 4090 D / fp16；同一模型骨架只改 num_key_value_heads。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
from task4_common import (assert_true, build_llama, cache_tensor_bytes, dump_json,
                          env_info, gib, ledger_kv_bytes, mib, mla_kv_bytes,
                          per_token_bytes, seeded_ids, source_sha256, sync_time_ms)

OUT = Path(__file__).parent / "task4_exp1_growth.json"
L, H, D = 12, 8, 128          # layers / attention heads / head_dim
HIDDEN = 1024
DT = torch.float16
DTB = 2                       # dtype bytes


def part_a_formula_vs_allocation():
    print("== A. 账本公式 vs 真实张量分配 ==")
    rows = []
    cases = [
        ("B=1,S=1024", 1, 1024, 32, 128, 2),
        ("B=4,S=512", 4, 512, 8, 128, 2),
        ("B=2,S=2048", 2, 2048, 4, 64, 2),
        ("B=1,S=256,fp32", 1, 256, 32, 128, 4),
    ]
    for name, B, S, Hkv, d, dtb in cases:
        dtype = torch.float32 if dtb == 4 else torch.float16
        # 真实分配：K 与 V 两份，形状 [B, H_kv, S, head_dim]
        k = torch.empty(B, Hkv, S, d, dtype=dtype, device="cuda")
        v = torch.empty(B, Hkv, S, d, dtype=dtype, device="cuda")
        real = k.numel() * k.element_size() + v.numel() * v.element_size()
        ledger = ledger_kv_bytes(S, 1, Hkv, d, dtb, batch=B)  # 单层
        ledger_all = ledger * 1  # 单层即可验证乘性
        assert_true(real == ledger_all, f"{name}: 单层分配 {real} == 账本 {ledger_all}")
        rows.append({"case": name, "bytes": real, "mib": round(mib(real), 3)})
        del k, v
    torch.cuda.empty_cache()
    return rows


def part_b_linearity():
    print("== B. 逐维度线性增长（倍增斜率） ==")
    base = dict(tokens=1024, layers=32, kv_heads=8, head_dim=128, dtype_bytes=2, batch=1)
    base_bytes = ledger_kv_bytes(**base)
    dims = {}
    for dim in ("tokens", "layers", "kv_heads", "head_dim", "dtype_bytes", "batch"):
        doubled = dict(base)
        doubled[dim] = doubled[dim] * 2
        ratio = ledger_kv_bytes(**doubled) / base_bytes
        assert_true(ratio == 2.0, f"{dim} 翻倍 -> KV 字节翻倍（比率 {ratio}）")
        dims[dim] = {"double_ratio": ratio, "linear": True}
    return {"base_bytes": base_bytes, "dimensions": dims,
            "note": "全部六个维度严格一阶线性；无平方项（tokens 不是 tokens^2）"}


def part_c_real_model():
    print("== C. 真实 LLaMA 玩具模型：MHA/GQA/MQA 实测 ==")
    B, S, GEN = 4, 512, 32
    results = []
    for name, kv_heads in [("MHA", 8), ("GQA-4", 4), ("GQA-2", 2), ("MQA", 1)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, cfg = build_llama(kv_heads, layers=L, hidden=HIDDEN, heads=H, dtype=DT)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        ids = seeded_ids((B, S), seed=20260926)
        with torch.no_grad():
            out = model(ids, use_cache=True)
            cache = out.past_key_values
            prefill_bytes = cache_tensor_bytes(cache)
            # decode：每步追加 1 token，看 cache 增长与步时
            past = cache
            nxt = out.logits[:, -1:].argmax(-1)
            step_bytes, step_ms = [], []
            for step in range(GEN):
                before = cache_tensor_bytes(past)
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
                t1.record()
                torch.cuda.synchronize()
                step_ms.append(t0.elapsed_time(t1))
                step_bytes.append(cache_tensor_bytes(past) - before)
        ledger_pf = ledger_kv_bytes(S, L, kv_heads, D, DTB, batch=B)
        per_tok = per_token_bytes(L, kv_heads, D, DTB)
        results.append({
            "config": name, "kv_heads": kv_heads,
            "prefill_cache_bytes": prefill_bytes, "prefill_ledger_bytes": ledger_pf,
            "ledger_match": prefill_bytes == ledger_pf,
            "per_token_bytes": per_tok, "per_token_kib": round(per_tok / 1024, 1),
            "decode_step_bytes_median": sorted(step_bytes)[len(step_bytes) // 2],
            "decode_step_ms_median": round(sorted(step_ms)[len(step_ms) // 2], 3),
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "param_mib": round(mib(param_bytes), 1),
        })
        assert_true(prefill_bytes == ledger_pf,
                    f"{name}: prefill cache {prefill_bytes} == 账本 {ledger_pf}")
        assert_true(sorted(step_bytes)[len(step_bytes) // 2] == per_tok * B,
                    f"{name}: decode 每步新增 {per_tok * B} B == per_token*B")
        del model, out, cache, past
        torch.cuda.empty_cache()
    return {"batch": B, "prefill_seq": S, "gen_tokens": GEN, "rows": results}


def part_d_mla():
    print("== D. MLA latent 表示的显式缓冲账本 ==")
    B, S = 1, 4096
    # 显式分配 latent 缓冲：每层每 token 存 (kv_lora_rank + qk_rope) 个 fp16 元素
    kv_lora, rope = 512, 64
    latent = torch.empty(B, S, L * (kv_lora + rope), dtype=DT, device="cuda")
    real = latent.numel() * latent.element_size()
    ledger = mla_kv_bytes(S, L, DTB, kv_lora_rank=kv_lora, qk_rope=rope, batch=B)
    assert_true(real == ledger, f"MLA latent 缓冲 {real} == 账本 {ledger}")
    mha = ledger_kv_bytes(S, L, H, D, DTB, batch=B)
    gqa8 = ledger_kv_bytes(S, L, 8, D, DTB, batch=B)
    del latent
    torch.cuda.empty_cache()
    return {"seq": S, "latent_elements_per_token_per_layer": kv_lora + rope,
            "mla_bytes": ledger, "mha_bytes_H8": mha, "gqa8_bytes": gqa8,
            "mla_vs_mha_ratio": round(ledger / mha, 4),
            "mla_vs_gqa8_ratio": round(ledger / gqa8, 4),
            "mla_mib": round(mib(ledger), 2), "mha_mib": round(mib(mha), 2),
            "gqa8_mib": round(mib(gqa8), 2)}


def part_e_kv_precision():
    print("== E. FP8 / INT8 KV：字节与数值误差 ==")
    torch.manual_seed(7)
    # 用真实 attention 投影后的 K/V 统计：取 GQA-4 模型第一层输出
    model, _ = build_llama(4, layers=L, hidden=HIDDEN, heads=H, dtype=DT)
    ids = seeded_ids((1, 256), seed=11)
    captured = {}

    def hook(mod, inp, out):
        captured["k"] = out.detach()  # k_proj 输出即缓存前的 K 投影张量

    handle = model.model.layers[0].self_attn.k_proj.register_forward_hook(hook)
    # k_proj 输出是 K；v 需要单独抓
    captured_v = {}

    def hook_v(mod, inp, out):
        captured_v["v"] = out.detach()
    hv = model.model.layers[0].self_attn.v_proj.register_forward_hook(hook_v)
    with torch.no_grad():
        model(ids)
    handle.remove()
    hv.remove()
    K, V = captured["k"].float(), captured_v["v"].float()
    del model
    torch.cuda.empty_cache()

    def report(name, kq, vq, bytes_fn):
        k_err = (kq.float() - K).abs().max().item()
        v_err = (vq.float() - V).abs().max().item()
        rel = (kq.float() - K).norm() / K.norm()
        b = bytes_fn()
        print(f"  {name}: bytes={b} max_abs_err(K)={k_err:.4f} rel_l2={rel:.4f}")
        return {"name": name, "bytes": b, "k_max_abs_err": round(k_err, 5),
                "v_max_abs_err": round(v_err, 5), "k_rel_l2": round(rel.item(), 5)}

    rows = []
    fp16_bytes = K.numel() * 2 + V.numel() * 2
    rows.append({"name": "fp16", "bytes": fp16_bytes, "k_max_abs_err": 0.0,
                 "v_max_abs_err": 0.0, "k_rel_l2": 0.0})
    # FP8 E4M3（torch 原生 float8 转换）
    k_fp8 = K.to(torch.float8_e4m3fn).to(torch.float16)
    v_fp8 = V.to(torch.float8_e4m3fn).to(torch.float16)
    rows.append(report("fp8_e4m3", k_fp8, v_fp8, lambda: K.numel() + V.numel()))
    # INT8 对称逐头缩放
    k_scale = K.abs().amax(dim=-1, keepdim=True) / 127.0
    v_scale = V.abs().amax(dim=-1, keepdim=True) / 127.0
    k_i8 = (K / k_scale).round().clamp(-127, 127).to(torch.int8)
    v_i8 = (V / v_scale).round().clamp(-127, 127).to(torch.int8)
    rows.append(report("int8_symmetric", (k_i8.float() * k_scale), (v_i8.float() * v_scale),
                       lambda: K.numel() + V.numel()))
    return {"tensor_shape": list(K.shape), "rows": rows,
            "fp16_vs_fp8_bytes_ratio": 2.0, "fp16_vs_int8_bytes_ratio": 2.0}


def main():
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    result = {
        "experiment": "task4_exp1_growth",
        "environment": env_info(),
        "source_sha256": source_sha256(__file__),
        "model_shape": {"layers": L, "heads": H, "head_dim": D, "hidden": HIDDEN,
                        "dtype": "fp16"},
        "formula": "KV bytes = 2 * layers * kv_heads * head_dim * tokens * dtype_bytes * batch",
        "a_allocation_match": part_a_formula_vs_allocation(),
        "b_linearity": part_b_linearity(),
        "c_real_model": part_c_real_model(),
        "d_mla": part_d_mla(),
        "e_kv_precision": part_e_kv_precision(),
        "status": "PASS",
    }
    dump_json(result, OUT)
    print("ALL_TESTS_PASS")


if __name__ == "__main__":
    main()
