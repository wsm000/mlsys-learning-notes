#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task6 LoRA/QLoRA Lab | CPU-first, stdlib only.
覆盖: LoRA原理账本 + LoRA vs QLoRA显存换算 + NF4量化演示 + DeepSeek式调参扫描(mock,已标注) + LLaMAFactory YAML生成.
本地Windows无torch可直接跑; vm-60有GPU时看 task6_vm60_lora_sweep.py 做真训.
"""
import json, platform, sys, time
from pathlib import Path

# ---------- 1. LoRA 参数账本 ----------
def lora_trainable_params(in_dim, out_dim, rank):
    return rank * (in_dim + out_dim)

def full_linear_params(in_dim, out_dim):
    return in_dim * out_dim

def lora_param_ratio(in_dim, out_dim, rank):
    return lora_trainable_params(in_dim, out_dim, rank) / full_linear_params(in_dim, out_dim)

def adapter_params_for_model(n_layers, n_targets_per_layer, hidden, rank):
    return n_layers * n_targets_per_layer * rank * 2 * hidden

# ---------- 2. 显存估算(估计值,非实测;单位MB) ----------
MB = 1024 * 1024
def estimate_full_ft_mem_mb(model_params, bytes_w=2.0, bytes_g=2.0, bytes_opt=8.0, overhead_mb=800.0):
    return model_params * (bytes_w + bytes_g + bytes_opt) / MB + overhead_mb

def estimate_lora_mem_mb(model_params, adapter_params, base_bytes=2.0, adapter_bytes=12.0, overhead_mb=800.0):
    return (model_params * base_bytes + adapter_params * adapter_bytes) / MB + overhead_mb

def estimate_qlora_mem_mb(model_params, adapter_params, base_bytes=0.5, adapter_bytes=12.0, quant_overhead_mb=150.0, overhead_mb=800.0):
    return (model_params * base_bytes + adapter_params * adapter_bytes) / MB + quant_overhead_mb + overhead_mb

def compare_table(model_params_B, hidden=4096, n_layers=32, n_targets=2, rank=8):
    P = int(model_params_B * 1e9)
    A = adapter_params_for_model(n_layers, n_targets, hidden, rank)
    return {
        "model_params_B": model_params_B, "adapter_params": A,
        "full_bf16_mb": round(estimate_full_ft_mem_mb(P), 1),
        "lora_bf16_mb": round(estimate_lora_mem_mb(P, A), 1),
        "qlora_nf4_mb": round(estimate_qlora_mem_mb(P, A), 1),
    }

# ---------- 3. NF4 演示(QLoRA论文表1,归一化到[-1,1]) ----------
NF4_LEVELS = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

def _nearest(levels, x):
    best, bd = levels[0], abs(x - levels[0])
    for v in levels[1:]:
        d = abs(x - v)
        if d < bd: best, bd = v, d
    return best

def quant_demo(values, absmax):
    uni = [-1.0 + i * (2.0 / 15) for i in range(16)]
    se_nf4, se_uni = 0.0, 0.0
    for v in values:
        x = max(-1.0, min(1.0, v / absmax))
        dn = _nearest(NF4_LEVELS, x) * absmax
        du = _nearest(uni, x) * absmax
        se_nf4 += (v - dn) ** 2; se_uni += (v - du) ** 2
    n = len(values)
    return {"n": n, "mse_nf4": round(se_nf4 / n, 7), "mse_uniform_int4": round(se_uni / n, 7)}

# 正态集中型小向量(靠近0的值多,NF4占优);absmax=0.65
DEMO_VEC = [-0.42, -0.31, -0.22, -0.15, -0.09, -0.04, -0.01, 0.0,
    0.02, 0.06, 0.11, 0.17, 0.24, 0.33, 0.45, 0.62]

# ---------- 4. DeepSeek式调参扫描(纯mock,说明相对趋势,不代表真训) ----------
RANK_GAIN = {4: 0.030, 8: 0.020, 16: 0.010, 32: 0.005, 64: 0.000}
TARGET_GAIN = {"q_v": 0.0, "q_k_v_o": -0.004, "all_linear": -0.010}
TARGET_MULT = {"q_v": 1.0, "q_k_v_o": 2.0, "all_linear": 3.5}
DROPOUT_PENALTY = {0.0: 0.005, 0.05: 0.0, 0.1: 0.005, 0.2: 0.015}
BASE_VAL = 0.52

def mock_predict_val(rank, alpha, target, dropout, use_qlora):
    g = RANK_GAIN[rank] + TARGET_GAIN[target] + DROPOUT_PENALTY[dropout]
    g += min(0.03, 0.01 * abs(alpha / rank - 2.0))
    if use_qlora: g += 0.005
    return round(BASE_VAL + g, 4)

def sweep_configs(hidden=4096, n_layers=32):
    base_n_targets = 2  # q_v
    cands = [
        ("A r8/a16 q_v d0.05 bf16", 8, 16, "q_v", 0.05, False),
        ("B r16/a32 q_v d0.05 bf16", 16, 32, "q_v", 0.05, False),
        ("C r32/a64 q_v d0.05 bf16", 32, 64, "q_v", 0.05, False),
        ("D r16/a32 all d0.05 bf16", 16, 32, "all_linear", 0.05, False),
        ("E r8/a8 q_v d0.05 bf16", 8, 8, "q_v", 0.05, False),
        ("F r8/a16 q_v d0.10 bf16", 8, 16, "q_v", 0.10, False),
        ("G r8/a16 q_v d0.05 nf4", 8, 16, "q_v", 0.05, True),
        ("H r16/a32 q_v d0.05 nf4", 16, 32, "q_v", 0.05, True),
        ("I r32/a64 q_v d0.05 nf4", 32, 64, "q_v", 0.05, True),
        ("J r16/a16 q_v d0.00 bf16", 16, 16, "q_v", 0.0, False),
        ("K r64/a128 q_v d0.05 bf16", 64, 128, "q_v", 0.05, False),
        ("L r16/a32 qkv_o d0.05 bf16", 16, 32, "q_k_v_o", 0.05, False),
    ]
    rows = []
    for name, rk, al, tg, do, q in cands:
        base_adapter = adapter_params_for_model(n_layers, base_n_targets, hidden, rk)
        adapter = int(base_adapter * TARGET_MULT[tg])
        rows.append({"name": name, "rank": rk, "alpha": al, "scaling": round(al / rk, 2),
            "target": tg, "dropout": do, "quant": ("nf4" if q else "bf16"),
            "adapter_params": adapter, "mock_val": mock_predict_val(rk, al, tg, do, q)})
    rows.sort(key=lambda r: r["mock_val"])
    return rows

def recommend(rows, adapter_budget=20_000_000, qlora_tolerance=0.01):
    feasible = [r for r in rows if r["adapter_params"] <= adapter_budget]
    if not feasible:
        return {"decision": "reject", "pick": None, "reason": "无配置满足adapter预算,先缩小rank/目标范围"}
    best = feasible[0]
    qlora_cands = [r for r in feasible if r["quant"] == "nf4"]
    alt = qlora_cands[0] if qlora_cands else None
    if alt and alt["mock_val"] - best["mock_val"] <= qlora_tolerance and alt["name"] != best["name"]:
        return {"decision": "accept", "pick": best["name"], "qlora_alt": alt["name"],
            "reason": f"最优为{best['name']}(mock val {best['mock_val']});{alt['name']}仅差{round(alt['mock_val']-best['mock_val'],4)},显存少约10GB,24GB卡优先选QLoRA"}
    return {"decision": "accept", "pick": best["name"], "qlora_alt": None,
        "reason": f"预算内最优为{best['name']}(mock val {best['mock_val']})"}

# ---------- 5. LLaMAFactory YAML 生成 ----------
def build_yaml(model_id, quant_bit, rank, alpha, dropout, targets, lr=2e-4, max_len=1024, epochs=3):
    t = "\n".join([f"  - {x}" for x in targets])
    qsec = "quantization_bit: 4\nquantization_method: bitsandbytes\n" if quant_bit == 4 else "quantization_bit: null\n"
    return f"""# LLaMAFactory {'QLoRA(NF4 4bit)' if quant_bit==4 else 'LoRA(bf16)'} | vm-60 24GB示例(按需改model/dataset路径)
model_name_or_path: {model_id}
trust_remote_code: true
finetuning_type: lora
{qsec}lora_rank: {rank}
lora_alpha: {alpha}
lora_dropout: {dropout}
lora_target:
{t}
learning_rate: {lr}
num_train_epochs: {epochs}
cutoff_len: {max_len}
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
lr_scheduler_type: cosine
warming_steps: 20
bf16: true
flash_attn: auto
logging_steps: 10
save_steps: 100
output_dir: outputs/{'qlora' if quant_bit==4 else 'lora'}-rank{rank}
report_to: swanlab
"""

def test_lab():
    assert lora_trainable_params(8, 8, 2) == 32
    assert full_linear_params(8, 8) == 64
    assert abs(lora_param_ratio(4096, 4096, 8) - 0.00390625) < 1e-12
    assert adapter_params_for_model(32, 2, 4096, 8) == 4194304
    t = compare_table(7)
    assert t["full_bf16_mb"] > t["lora_bf16_mb"] > t["qlora_nf4_mb"]
    assert 12000 < t["lora_bf16_mb"] < 17000 and 3000 < t["qlora_nf4_mb"] < 6000
    d = quant_demo(DEMO_VEC, 0.65)
    assert d["mse_nf4"] < d["mse_uniform_int4"], d
    rows = sweep_configs()
    assert len(rows) == 12 and rows[0]["mock_val"] <= rows[-1]["mock_val"]
    rec = recommend(rows)
    assert rec["decision"] == "accept" and rec["pick"] is not None
    y = build_yaml("Qwen/Qwen2.5-0.5B-Instruct", 4, 16, 32, 0.05, ["q_proj", "v_proj"])
    assert "quantization_bit: 4" in y and "lora_rank: 16" in y and "swanlab" in y
    print("PASS test_lab")

if __name__ == "__main__":
    t0 = time.perf_counter()
    print("=== Task6 LoRA/QLoRA Lab | CPU-first stdlib ===")
    print(f"host={platform.node()} {platform.platform()} python={sys.version.split()[0]}")
    test_lab()
    for hs, rk in [(4096, 8), (4096, 16), (4096, 32)]:
        tr = lora_trainable_params(hs, hs, rk)
        print(f"hidden={hs} rank={rk} -> adapter/层={tr:,} 全层32x2={tr*64:,} 占比={lora_param_ratio(hs,hs,rk):.4%}")
    for B in (1.5, 7, 8):
        print(f"{B}B:", compare_table(B))
    print("NF4 demo:", quant_demo(DEMO_VEC, 0.65))
    rows = sweep_configs()
    for r in rows[:6]:
        print(f"  {r['name']}: val={r['mock_val']} adapter={r['adapter_params']:,} scaling={r['scaling']}")
    print("recommend:", recommend(rows))
    outdir = Path("打卡材料/task6_qlora"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "lab_report.json").write_text(json.dumps(
        {"tables_7B": compare_table(7), "nf4": quant_demo(DEMO_VEC, 0.65),
         "sweep_mock": rows, "recommend": recommend(rows),
         "note": "显存为公式估算;mock_val仅用于讲清相对趋势,不是真训结论.真训见task6_vm60_lora_sweep.py在vm-60上的实测"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "llamafactory_qlora_qwen25_05b_vm60.yaml").write_text(
        build_yaml("Qwen/Qwen2.5-0.5B-Instruct", 4, 16, 32, 0.05, ["q_proj", "v_proj"]), encoding="utf-8")
    (outdir / "llamafactory_lora_qwen25_05b_vm60.yaml").write_text(
        build_yaml("Qwen/Qwen2.5-0.5B-Instruct", 0, 16, 32, 0.05, ["q_proj", "v_proj"]), encoding="utf-8")
    print(f"saved to {outdir} total_wall={(time.perf_counter()-t0)*1000:.1f}ms")
    print("DONE lab")
