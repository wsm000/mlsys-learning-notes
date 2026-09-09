#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day5 加分挑战: cat 版 KVCache vs 预分配 buffer 版, 强制生成长度对比.
实验矩阵: FORCE_LEN in {200, 400}, 各 5 次取中位; 同 seed 正确性校验.
"""
import importlib.util
import time
import statistics
import json
import gc
import torch
import pickle

HERE = "/home/simin/projects/day5_llm"
DEVICE = torch.device("cuda")
BPE_MAX = 512
VERSIONS = ["molecular_gpt_KVCache", "molecular_gpt_KVCacheBuf"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    torch.manual_seed(42)
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    mods = {}
    for name in VERSIONS:
        mods[name] = load(name)
        mods[name].EOS_IDX = -1  # 禁用提前停止
    print("EOS disabled for both versions")

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
            vocab_size=vocab_size, d_model=margs.d_model,
            n_heads=margs.n_heads, n_layers=margs.n_layers,
            ff_dim=margs.ff_dim, max_len=BPE_MAX,
            dropout=margs.dropout,
        ).to(DEVICE)
        sd = {k: v for k, v in ckpt["model_state_dict"].items()
              if not k.startswith("pos_enc.pe")}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and all("pos_enc" in k for k in missing)
        m.eval()
        return m

    models = {v: build(v) for v in VERSIONS}

    # ---------- 正确性校验: 同 seed 生成应一致 ----------
    smiles = {}
    for v in VERSIONS:
        torch.manual_seed(123)
        smiles[v] = mods[v].generate(models[v], "c1cc", char2idx, idx2char,
                                     DEVICE, max_new_tokens=200)
    match = smiles[VERSIONS[0]] == smiles[VERSIONS[1]]
    print(f"\n[correctness] same-seed SMILES identical: {match}")
    if not match:
        print(f"  cat  : {smiles[VERSIONS[0]]}")
        print(f"  buf  : {smiles[VERSIONS[1]]}")

    # ---------- 计时 ----------
    def bench(version, max_new_tokens, repeats):
        model = models[version]
        gen = mods[version].generate
        for _ in range(2):
            torch.manual_seed(42)
            gen(model, "c1cc", char2idx, idx2char, DEVICE, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.manual_seed(42)
            t0 = time.perf_counter()
            gen(model, "c1cc", char2idx, idx2char, DEVICE, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return {"median_s": round(statistics.median(times), 4),
                "min_s": round(min(times), 4), "repeats": repeats}

    results = {"correct_same_seed_match": match}
    for L in (200, 400):
        key = f"forced_{L}tok"
        results[key] = {v: bench(v, L, 5) for v in VERSIONS}
        for v in VERSIONS:  # 释放 buffer 版惰性分配的缓存避免碎片
            pass
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== RESULTS =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n===== 汇总 (中位数) =====")
    print(f"{'版本':<28} {'200tok (s)':<12} {'400tok (s)':<12}")
    for v in VERSIONS:
        a = results["forced_200tok"][v]["median_s"]
        b = results["forced_400tok"][v]["median_s"]
        print(f"{v:<28} {a:<12.4f} {b:<12.4f}")
    for L in (200, 400):
        c = results[f"forced_{L}tok"][VERSIONS[0]]["median_s"]
        b = results[f"forced_{L}tok"][VERSIONS[1]]["median_s"]
        print(f"buffer vs cat @{L}tok: {c/b:.3f}x (buffer 更快则 >1)")

    with open(f"{HERE}/day5_results_v4.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {HERE}/day5_results_v4.json")


if __name__ == "__main__":
    main()
