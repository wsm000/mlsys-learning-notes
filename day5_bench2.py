#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day5 推理加速对比实验 v3 (vm-60)
v2 发现: checkpoint args.max_len=64, 位置编码表只有 64 行, 强制 200 token 越界.
v3 改动: 重建模型时 max_len=512, strict=False 加载 (pos_enc.pe 不加载, 只影响采样内容不影响计时),
        并验证 missing keys 恰好只有 pos_enc.pe.
实验矩阵:
  B. 强制 200 token 单分子, 三版本各 5 次   -> 验证 O(N^2) vs O(N)
  C. 批量: KVCache 串行32次 vs 真 batch=32  -> 验证吞吐
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
FORCE_LEN = 200   # 强制生成的 token 数
BPE_MAX = 512     # 重建模型时的位置编码表大小


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{HERE}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    torch.manual_seed(42)
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    mods = {}
    for name in ["molecular_gpt", "molecular_gpt_cuDNN", "molecular_gpt_KVCache"]:
        mods[name] = load(name)
        mods[name].EOS_IDX = -1  # 禁用提前停止, 强制跑满长度
    print("EOS early-stop disabled for all versions")

    ckpt = torch.load(f"{HERE}/checkpoints/best_model.pt",
                      map_location=DEVICE, weights_only=False)
    margs = ckpt["args"]
    print(f"checkpoint args: d_model={margs.d_model} n_layers={margs.n_layers} "
          f"n_heads={margs.n_heads} max_len={margs.max_len}")
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
              if not k.startswith("pos_enc.pe")}  # pe 是 buffer 非权重, 形状不同直接剔除
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all("pos_enc" in k for k in missing), f"unexpected missing: {missing}"
        m.eval()
        return m

    # ---------- 实验 B: 强制 200 token 单分子 ----------
    def bench_single(version, max_new_tokens, repeats):
        model = build(version)
        gen = mods[version].generate
        frag = "c1cc"
        for _ in range(2):  # 预热
            torch.manual_seed(42)
            gen(model, frag, char2idx, idx2char, DEVICE, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.manual_seed(42)
            t0 = time.perf_counter()
            gen(model, frag, char2idx, idx2char, DEVICE, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {"median_s": round(statistics.median(times), 4),
                "min_s": round(min(times), 4), "repeats": repeats}

    results = {}
    results["B_forced200"] = {v: bench_single(v, FORCE_LEN, 5) for v in mods}

    # ---------- 实验 C: 批量 (KVCache 版) ----------
    kvc = mods["molecular_gpt_KVCache"]
    model_kvc = build("molecular_gpt_KVCache")

    @torch.no_grad()
    def batched_generate(model, frag, batch, max_new_tokens):
        """真 batch 生成: B 个相同片段一次 prefill, 逐 token batch decode (EOS 已禁用)."""
        tokens = [kvc.BOS_IDX] + [char2idx.get(c, kvc.UNK_IDX) for c in frag]
        init_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
        init_ids = init_ids.expand(batch, -1).contiguous()
        mask = kvc.create_causal_mask(init_ids.size(1), DEVICE)
        logits, past = model(init_ids, mask=mask, use_cache=True)
        n_steps = max_new_tokens - len(tokens) + 1
        last = logits[:, -1, :]
        for _ in range(n_steps):
            probs = torch.softmax(last, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # (B,1)
            logits, past = model(nxt, past_key_values=past, use_cache=True, mask=None)
            last = logits[:, -1, :]
        return batch

    def bench_serial32(max_new_tokens=FORCE_LEN, repeats=3):
        gen = kvc.generate
        frag = "c1cc"
        for _ in range(1):  # 预热
            torch.manual_seed(42)
            gen(model_kvc, frag, char2idx, idx2char, DEVICE, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.manual_seed(42)
            t0 = time.perf_counter()
            for _ in range(32):
                gen(model_kvc, frag, char2idx, idx2char, DEVICE,
                    max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return times

    def bench_batch32(max_new_tokens=FORCE_LEN, repeats=3):
        for _ in range(1):  # 预热
            torch.manual_seed(42)
            batched_generate(model_kvc, "c1cc", 32, max_new_tokens)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.manual_seed(42)
            t0 = time.perf_counter()
            batched_generate(model_kvc, "c1cc", 32, max_new_tokens)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return times

    ser = bench_serial32()
    bat = bench_batch32()
    results["C_batch"] = {
        "serial_32x": {"median_s": round(statistics.median(ser), 3),
                       "mol_per_s": round(32 / statistics.median(ser), 1)},
        "batch32_1x": {"median_s": round(statistics.median(bat), 3),
                       "mol_per_s": round(32 / statistics.median(bat), 1)},
        "speedup": round(statistics.median(ser) / statistics.median(bat), 2),
    }

    print("\n===== RESULTS =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n===== 加速比汇总 (中位数, 强制200token) =====")
    base = results["B_forced200"]["molecular_gpt"]["median_s"]
    print(f"{'版本':<24} {'200tok (s)':<12} {'vs base':<10}")
    for v in mods:
        b = results["B_forced200"][v]["median_s"]
        print(f"{v:<24} {b:<12.4f} {base/b:<10.2f}x")

    with open("/home/simin/day5_results_v3.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved: /home/simin/day5_results_v3.json")


if __name__ == "__main__":
    main()
