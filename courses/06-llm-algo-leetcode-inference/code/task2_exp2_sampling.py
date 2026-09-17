# -*- coding: utf-8 -*-
"""Task2 实验 2：采样策略对质量 / 重复率 / 生成长度的影响（4.3(2)）。

固定模型（gpt2，124M）、固定 prompt 集合、固定 max_new_tokens、固定种子集合，
只改变解码策略：greedy / temperature / top-k / top-p 及其组合。
记录：生成长度、重复率（重复 n-gram 占比）、distinct-n（多样性）、
      自困惑度（质量代理）、与 greedy 输出的一致性、跨种子稳定性、解码速度。
"""
import json
import math
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.expanduser("~/local_models/gpt2")
OUT = "task2_exp2_sampling.json"
DEV = "cuda"

PROMPTS = [
    "The history of the printing press shows that",
    "In a small coastal town, the lighthouse keeper",
    "Machine learning models are often evaluated by",
    "The recipe begins with three simple ingredients:",
    "When the spacecraft finally reached the planet,",
    "A good way to explain recursion to a beginner is",
]
SEEDS = [11, 22, 33]
MAX_NEW = 100

STRATEGIES = {
    "greedy": dict(do_sample=False),
    "T=0.3": dict(do_sample=True, temperature=0.3),
    "T=0.7": dict(do_sample=True, temperature=0.7),
    "T=1.0": dict(do_sample=True, temperature=1.0),
    "T=1.5": dict(do_sample=True, temperature=1.5),
    "top_k=10": dict(do_sample=True, temperature=1.0, top_k=10),
    "top_p=0.9": dict(do_sample=True, temperature=1.0, top_p=0.9),
    "top_k10+top_p0.9": dict(do_sample=True, temperature=1.0, top_k=10, top_p=0.9),
}


def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    return tokenizer, model.to(DEV).eval()


def repetition_rate(tokens, n):
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(token_lists, n):
    total, uniq = 0, set()
    for tokens in token_lists:
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / total if total else 0.0


@torch.no_grad()
def self_perplexity(model, full_ids):
    """用同一模型给「prompt+生成」打分，取生成段的平均负对数似然换算困惑度。"""
    out = model(full_ids)
    logits = out.logits[:, :-1, :].float()
    targets = full_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return loss.reshape(targets.shape)


@torch.no_grad()
def generate(model, tokenizer, prompt, kwargs, seed):
    torch.manual_seed(seed)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEV)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(
        ids, max_new_tokens=MAX_NEW, pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_ids = out[0, ids.shape[1]:].tolist()
    return new_ids, elapsed, out


def main():
    tokenizer, model = load()
    print("model loaded", flush=True)
    eos = tokenizer.eos_token_id
    report = {"model": MODEL_DIR, "max_new_tokens": MAX_NEW, "seeds": SEEDS,
              "prompt_count": len(PROMPTS), "strategies": {}, "samples": []}

    greedy_map = {}
    for prompt in PROMPTS:
        ids, _, _ = generate(model, tokenizer, prompt, dict(do_sample=False), SEEDS[0])
        greedy_map[prompt] = ids

    for name, kwargs in STRATEGIES.items():
        lengths, rep2, rep3, ppls, agreements, speeds = [], [], [], [], [], []
        all_tokens, samples = [], []
        for prompt in PROMPTS:
            for seed in SEEDS:
                tokens, elapsed, full = generate(model, tokenizer, prompt, kwargs, seed)
                all_tokens.append(tokens)
                lengths.append(len(tokens))
                rep2.append(repetition_rate(tokens, 2))
                rep3.append(repetition_rate(tokens, 3))
                speeds.append(len(tokens) / elapsed)
                loss = self_perplexity(model, full)
                gen_loss = loss[0, -len(tokens):] if len(tokens) else loss[0, :0]
                ppls.append(float(torch.exp(gen_loss.mean())) if len(tokens) else float("nan"))
                ref = greedy_map[prompt]
                match = sum(1 for a, b in zip(tokens, ref) if a == b)
                agreements.append(match / max(1, min(len(tokens), len(ref))))
                if name == "greedy":
                    eos_hit = len(tokens) < MAX_NEW
                else:
                    eos_hit = eos in tokens
                samples.append({
                    "strategy": name, "prompt": prompt[:40], "seed": seed,
                    "n_tokens": len(tokens), "stopped_early": bool(eos_hit),
                    "rep2": round(rep2[-1], 4), "ppl": round(ppls[-1], 3),
                    "agreement_with_greedy": round(agreements[-1], 4),
                    "text": tokenizer.decode(tokens, skip_special_tokens=True)[:200],
                })
        report["strategies"][name] = {
            "kwargs": {k: v for k, v in kwargs.items()},
            "mean_length": round(statistics.mean(lengths), 2),
            "std_length": round(statistics.pstdev(lengths), 2),
            "early_stop_rate": round(sum(1 for s in samples if s["stopped_early"]) / len(samples), 3),
            "mean_rep2": round(statistics.mean(rep2), 4),
            "mean_rep3": round(statistics.mean(rep3), 4),
            "distinct_1": round(distinct_n(all_tokens, 1), 4),
            "distinct_2": round(distinct_n(all_tokens, 2), 4),
            "mean_self_ppl": round(statistics.mean(ppls), 3),
            "std_self_ppl": round(statistics.pstdev(ppls), 3),
            "mean_agreement_with_greedy": round(statistics.mean(agreements), 4),
            "decode_tok_s": round(statistics.mean(speeds), 2),
        }
        report["samples"].extend(samples)
        print(name, report["strategies"][name], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
