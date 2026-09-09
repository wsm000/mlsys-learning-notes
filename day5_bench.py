#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day5 推理加速对比实验 (vm-60 自动化)
统一计时口径: import 三个版本的 generate(), 同 seed 同参数, 只测纯生成段.
实验矩阵:
  A. 单分子短序列: max_new_tokens=100, repeats=20  -> 验证"短序列看不出提升"
  B. 长序列:       max_new_tokens=500, repeats=10  -> 验证 O(N^2) vs O(N)
每个 (版本, 参数) 组合: 先 2 次预热(不计时), 再计时 repeats 次, 报告 中位数/均值.
"""
import importlib.util
import time
import statistics
import json
import gc
import torch
import pickle

HERE = "/home/simin/day5_llm"
DEVICE = torch.device("cuda")


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # 顶层只做定义, main 受 __main__ 保护, 无副作用
    return m


def main():
    torch.manual_seed(42)
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    mods = {}
    for name in ["molecular_gpt", "molecular_gpt_cuDNN", "molecular_gpt_KVCache"]:
        mods[name] = load(name)
        print(f"loaded: {name}")

    # 三个文件模型结构等价, 用各自模块里的类加载同一份 state_dict
    ckpt = torch.load(f"{HERE}/checkpoints/best_model.pt",
                      map_location=DEVICE, weights_only=False)
    margs = ckpt["args"]
    with open(f"{HERE}/checkpoints/best_model.pt.vocab", "rb") as f:
        vocab = pickle.load(f)
    char2idx = vocab["char2idx"]
    idx2char = vocab["idx2char"]
    vocab_size = vocab["vocab_size"]

    def build(version):
        m = mods[version].MolecularGPT(
            vocab_size=vocab_size,
            d_model=margs.d_model,
            n_heads=margs.n_heads,
            n_layers=margs.n_layers,
            ff_dim=margs.ff_dim,
            max_len=margs.max_len,
            dropout=margs.dropout,
        ).to(DEVICE)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        return m

    def bench(version, max_new_tokens, repeats, temperature=1.0):
        model = build(version)
        gen = mods[version].generate
        frag = "c1cc"
        # 预热 2 次 (CUDA kernel / autotune 等一次性开销, 不计入)
        for _ in range(2):
            torch.manual_seed(42)
            gen(model, frag, char2idx, idx2char, DEVICE,
                max_new_tokens=max_new_tokens, temperature=temperature)
        torch.cuda.synchronize()
        times = []
        n_toks = []
        for _ in range(repeats):
            torch.manual_seed(42)  # 同 seed 保证三版本采样序列可比
            t0 = time.perf_counter()
            smi = gen(model, frag, char2idx, idx2char, DEVICE,
                      max_new_tokens=max_new_tokens, temperature=temperature)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            n_toks.append(len(smi))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "median_s": round(statistics.median(times), 4),
            "mean_s": round(statistics.mean(times), 4),
            "min_s": round(min(times), 4),
            "repeats": repeats,
            "smiles_example": smi,
            "smiles_len": len(smi),
        }

    results = {}
    # 实验A: 短序列
    results["A_100tok"] = {
        v: bench(v, 100, 20) for v in mods
    }
    # 实验B: 长序列
    results["B_500tok"] = {
        v: bench(v, 500, 10) for v in mods
    }

    print("\n===== RESULTS =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n===== 加速比汇总 (中位数) =====")
    base_a = results["A_100tok"]["molecular_gpt"]["median_s"]
    base_b = results["B_500tok"]["molecular_gpt"]["median_s"]
    print(f"{'版本':<24} {'100tok (s)':<12} {'vs base':<10} {'500tok (s)':<12} {'vs base':<10}")
    for v in mods:
        a = results["A_100tok"][v]["median_s"]
        b = results["B_500tok"][v]["median_s"]
        print(f"{v:<24} {a:<12.4f} {base_a/a:<10.2f}x {b:<12.4f} {base_b/b:<10.2f}x")

    with open("/home/simin/day5_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved: /home/simin/day5_results.json")


if __name__ == "__main__":
    main()
