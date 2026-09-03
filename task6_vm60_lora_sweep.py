#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task6 vm-60 轻量LoRA调参 | GPT2(124M)+手写LoRA,无peft/bitsandbytes依赖.
- vm-60(4090D 24GB): 自动真训,3个rank配置 x 20步,几分钟跑完,<8GB显存.
- 本地无torch: 自动mock,产出同结构报告(明确标注simulated=True),用于先验证链路.
用法(vm-60):
  python3 task6_vm60_lora_sweep.py --model openai-community/gpt2 --max-steps 20
  python3 task6_vm60_lora_sweep.py --mock   # 只跑模拟,不碰模型
"""
import argparse, json, platform, re, sys, time
from pathlib import Path

TINY_DATA = [
    {"prompt": "问:什么是LoRA?\n答:", "response": "LoRA冻结基座,只训练低秩旁路BA,省显存。"},
    {"prompt": "问:loss口径怎么查?\n答:", "response": "看labels!=-100且mask==1的token才进loss。"},
    {"prompt": "Q: 2+3=?\nA:", "response": "5。先算加法,再输出结果。"},
    {"prompt": "Q: 7*6=?\nA:", "response": "42。七六四十二。"},
    {"prompt": "问:QLoRA省在哪?\n答:", "response": "4bit NF4存基座+分页优化器,只训adapter。"},
    {"prompt": "Q: capital of France?\nA:", "response": "Paris."},
    {"prompt": "问:rank越大越好吗?\n答:", "response": "不是,收益递减,先看val再看显存。"},
    {"prompt": "Q: 15-7=?\nA:", "response": "8。"},
    {"prompt": "问:alpha怎么设?\n答:", "response": "先按alpha=2*rank,再上下浮动一档。"},
    {"prompt": "Q: 9+8=?\nA:", "response": "17。"},
    {"prompt": "问:dropout设多少?\n答:", "response": "小数据先用0.05,0.2一般太大了。"},
    {"prompt": "Q: 100/4=?\nA:", "response": "25。"},
]

def build_sft_ids(tokenizer, prompt, response, max_len=128):
    p = tokenizer.encode(prompt, add_special_tokens=False)
    r = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    if not any(l != -100 for l in labels):
        raise ValueError("no supervised tokens after truncation")
    m = [1] * len(ids)
    return ids, m, labels

def pad_batch(tokenizer, rows):
    L = max(len(r[0]) for r in rows)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": [[*a, *[pad] * (L - len(a))] for a, _, _ in rows],
        "attention_mask": [[*m, *[0] * (L - len(m))] for _, m, _ in rows],
        "labels": [[*l, *[-100] * (L - len(l))] for _, _, l in rows],
    }

def mock_run(args):
    import hashlib
    cfgs = [{"rank": r, "alpha": 2 * r, "dropout": 0.05, "targets": ["c_attn", "c_proj"]} for r in (4, 8, 16)]
    rows = []
    for c in cfgs:
        h = int(hashlib.md5(f"{c['rank']}".encode()).hexdigest()[:4], 16)
        base = 3.2 - c["rank"] * 0.02
        rows.append({**c, "trainable_params": c["rank"] * 2048,
            "train_loss_first": round(base + 0.3, 4), "train_loss_last": round(base, 4),
            "val_loss": round(base + 0.15 - c["rank"] * 0.005, 4),
            "step_time_ms": 120.0 + c["rank"], "peak_mem_mb": 900.0 + c["rank"] * 5,
            "simulated": True})
    rows.sort(key=lambda r: r["val_loss"])
    return {"mode": "mock", "model": args.model, "rows": rows,
        "best": rows[0]["rank"], "note": "本地无torch的模拟结果,仅验证链路;vm-60真训会覆盖为simulated=False"}

def _ensure_transformers_importable():
    """vm-60 transformers/kernels 版本错位时,用直通桩顶掉 hub_kernels(GPT2 用不到它,不改机器环境)."""
    try:
        import transformers.integrations.hub_kernels  # noqa
    except Exception as e:
        import sys, types
        print(f"[vm60] hub_kernels import broken, using passthrough stub: {e}", flush=True)
        stub = types.ModuleType("transformers.integrations.hub_kernels")
        def _factory(*a, **k):
            def deco(fn): return fn
            return deco
        stub.use_kernel_forward_from_hub = _factory
        stub.__getattr__ = lambda name: _factory
        sys.modules["transformers.integrations.hub_kernels"] = stub

def _ensure_torchvision_stub():
    """torchvision 与 torch 2.9.1 错位导致 transformers 建模链 import 失败时,用最小桩顶掉(纯文本训练用不到 vision)."""
    try:
        import torchvision.io  # noqa
        return
    except Exception as e:
        import sys, types
        print(f"[vm60] torchvision import broken, using stub: {e}", flush=True)
        for m in [m for m in list(sys.modules) if m == "torchvision" or m.startswith("torchvision.")]:
            del sys.modules[m]
        tv = types.ModuleType("torchvision")
        tv.__path__ = []
        tv.__version__ = "0.0-stub"
        tio = types.ModuleType("torchvision.io")
        class ImageReadMode:
            UNCHANGED = 0
            GRAY = 1
            GRAY_ALPHA = 2
            RGB = 3
            RGB_ALPHA = 4
        def decode_image(*a, **k):
            raise RuntimeError("torchvision stub: decode_image unavailable")
        tio.ImageReadMode = ImageReadMode
        tio.decode_image = decode_image
        tv.io = tio
        ttrans = types.ModuleType("torchvision.transforms")
        class InterpolationMode:
            NEAREST = 0
            NEAREST_EXACT = 1
            BILINEAR = 2
            BICUBIC = 3
            BOX = 4
            HAMMING = 5
            LANCZOS = 6
        ttrans.InterpolationMode = InterpolationMode
        ttrans.__getattr__ = lambda name: type(name, (), {})
        tfunc = types.ModuleType("torchvision.transforms.functional")
        def _unavailable(*a, **k):
            raise RuntimeError("torchvision stub: unavailable")
        tfunc.__getattr__ = lambda name: _unavailable
        ttrans.functional = tfunc
        tv.transforms = ttrans
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.io"] = tio
        sys.modules["torchvision.transforms"] = ttrans
        sys.modules["torchvision.transforms.functional"] = tfunc

def real_run(args):
    _ensure_transformers_importable()
    _ensure_torchvision_stub()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32  # 4090D用bf16:fp16跑GPT2 logits易溢出nan
    print(f"[vm60] loading {args.model} -> {dev} {dtype}", flush=True)
    def _from_pretrained(cls, **kw):
        try:
            return cls.from_pretrained(args.model, local_files_only=True, trust_remote_code=False, **kw)
        except Exception as e:
            print(f"[vm60] local cache miss, trying online: {e}", flush=True)
            return cls.from_pretrained(args.model, trust_remote_code=False, **kw)
    tok = _from_pretrained(AutoTokenizer)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = _from_pretrained(AutoModelForCausalLM, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(dev); model.train()
    for p in model.parameters(): p.requires_grad = False

    # ---- 手写LoRA包装(同时兼容nn.Linear与GPT2 Conv1D:前向都是 x@W) ----
    import torch.nn as nn
    class LoRAWrapper(nn.Module):
        def __init__(self, orig, in_f, out_f, rank, alpha, dropout):
            super().__init__()
            self.orig = orig
            self.scaling = alpha / rank
            self.drop = nn.Dropout(dropout)
            self.A = nn.Parameter(torch.empty(rank, in_f))
            self.B = nn.Parameter(torch.zeros(out_f, rank))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            # orig参数冻结
            for p in self.orig.parameters(): p.requires_grad = False
        def forward(self, x):
            base = self.orig(x)
            l = (self.drop(x) @ self.A.T @ self.B.T) * self.scaling
            return base + l

    def attach_lora(model, substrs, rank, alpha, dropout):
        handles = []
        for name, mod in list(model.named_modules()):
            if not any(s in name for s in substrs): continue
            if isinstance(mod, nn.Linear):
                in_f, out_f = mod.in_features, mod.out_features
            else:  # GPT2 Conv1D: weight shape (in, out)
                try:
                    w = mod.weight
                    in_f, out_f = int(w.shape[0]), int(w.shape[1])
                except Exception: continue
            wrap = LoRAWrapper(mod, in_f, out_f, rank, alpha, dropout).to(dev, dtype)
            # 把wrapper挂回父模块
            parent, attr = model, name.split(".")
            for a in attr[:-1]: parent = getattr(parent, a)
            setattr(parent, attr[-1], wrap)
            handles.append((name, in_f, out_f))
        return handles

    train_rows = [build_sft_ids(tok, d["prompt"], d["response"]) for d in TINY_DATA[:10]]
    val_rows = [build_sft_ids(tok, d["prompt"], d["response"]) for d in TINY_DATA[10:]]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    cfgs = [{"rank": r, "alpha": 2 * r, "dropout": 0.05, "targets": ["c_attn", "c_proj"]} for r in (4, 8, 16)]
    results = []
    for c in cfgs:
        # 恢复:最简单是重载模型?为轻量起见,每轮新建wrapper会叠加——所以每轮重新load noms?改为每轮只训一次后移除wrapper
        # 做法:记录被替换的orig,训完恢复。
        replaced = {}
        for name, mod in list(model.named_modules()):
            if not any(s in name for s in ["c_attn", "c_proj"]): continue
            import torch.nn as nn2
            if isinstance(mod, LoRAWrapper): continue
            if isinstance(mod, nn2.Linear): in_f, out_f = mod.in_features, mod.out_features
            else:
                try: in_f, out_f = int(mod.weight.shape[0]), int(mod.weight.shape[1])
                except Exception: continue
            wrap = LoRAWrapper(mod, in_f, out_f, c["rank"], c["alpha"], c["dropout"]).to(dev, dtype)
            parent = model
            parts = name.split(".")
            for a in parts[:-1]: parent = getattr(parent, a)
            replaced[name] = mod
            setattr(parent, parts[-1], wrap)
        lora_params = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in lora_params)
        opt = torch.optim.AdamW(lora_params, lr=2e-4)
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter(); first, last = None, None
        model.train()
        for step in range(args.max_steps):
            batch = pad_batch(tok, [train_rows[step % len(train_rows)], train_rows[(step + 1) % len(train_rows)]])
            ids = torch.tensor(batch["input_ids"], device=dev)
            msk = torch.tensor(batch["attention_mask"], device=dev)
            lab = torch.tensor(batch["labels"], device=dev)
            out = model(input_ids=ids, attention_mask=msk)
            shift, shift_lab = out.logits[..., :-1, :].float(), lab[..., 1:].contiguous()
            loss = loss_fn(shift.view(-1, shift.size(-1)), shift_lab.view(-1))
            loss.backward()
            opt.step(); opt.zero_grad()
            if step == 0: first = float(loss.detach())
            last = float(loss.detach())
        wall = (time.perf_counter() - t1) * 1000 / max(1, args.max_steps)
        # val
        model.eval()
        with torch.no_grad():
            vb = pad_batch(tok, val_rows)
            ids = torch.tensor(vb["input_ids"], device=dev)
            msk = torch.tensor(vb["attention_mask"], device=dev)
            lab = torch.tensor(vb["labels"], device=dev)
            out = model(input_ids=ids, attention_mask=msk)
            shift, shift_lab = out.logits[..., :-1, :].float(), lab[..., 1:].contiguous()
            vloss = float(loss_fn(shift.view(-1, shift.size(-1)), shift_lab.view(-1)))
        peak = float(torch.cuda.max_memory_allocated() / 1024 ** 2) if dev == "cuda" else 0.0
        # sanity生成
        with torch.no_grad():
            q = tok(TINY_DATA[0]["prompt"], return_tensors="pt").to(dev)
            g = model.generate(**q, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.decode(g[0], skip_special_tokens=True)[:200]
        results.append({**c, "trainable_params": int(n_train),
            "train_loss_first": round(first, 4), "train_loss_last": round(last, 4),
            "val_loss": round(vloss, 4), "step_time_ms": round(wall, 2),
            "peak_mem_mb": round(peak, 1), "simulated": False, "sanity_gen": gen})
        print(f"[vm60] rank={c['rank']} train {first:.4f}->{last:.4f} val={vloss:.4f} params={n_train} peak={peak:.0f}MB", flush=True)
        # 恢复orig,去掉wrapper
        for name, orig in replaced.items():
            parent = model
            parts = name.split(".")
            for a in parts[:-1]: parent = getattr(parent, a)
            setattr(parent, parts[-1], orig)
    results.sort(key=lambda r: r["val_loss"])
    return {"mode": "real", "model": args.model, "device": dev, "dtype": str(dtype),
        "rows": results, "best": results[0]["rank"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="打卡材料/task6_qlora/vm60_sweep_report.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    print(f"=== Task6 vm-60 LoRA sweep | host={platform.node()} {platform.platform()} ===", flush=True)
    try:
        import torch  # noqa
        from transformers import AutoTokenizer  # noqa
        has = not args.mock
    except Exception as e:
        print(f"torch/transformers不可用,走mock: {e}", flush=True)
        has = False
    rep = real_run(args) if has else mock_run(args)
    rep["environment"] = {"host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:3000], flush=True)
    print(f"saved to {args.out}", flush=True)
    print("PASS vm60_sweep" if rep.get("rows") else "FAIL vm60_sweep", flush=True)

if __name__ == "__main__":
    main()
