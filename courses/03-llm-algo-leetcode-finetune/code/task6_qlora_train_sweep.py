#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task6 QLoRA真训练(GPT2 4bit基座+手写LoRA,vm-60) | 20/100/2000三档 x rank4/8/16.
与 task6_vm60_lora_sweep.py(bf16基座)严格对齐:同数据/同batch/同lr/同loss口径,只差基座精度.
用法(vm-60): python3 ~/task6_qlora_train_sweep.py --model ~/local_models/gpt2 --tiers 20,100,2000 --out /tmp/vm60_qlora_train.json
"""
import argparse, json, platform, sys, time
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

def _stub():
    try:
        import transformers.integrations.hub_kernels  # noqa
    except Exception as e:
        import types
        print(f"[stub] hub_kernels: {e}", flush=True)
        stub = types.ModuleType("transformers.integrations.hub_kernels")
        def _factory(*a, **k):
            def deco(fn): return fn
            return deco
        stub.use_kernel_forward_from_hub = _factory
        def _hg(n):
            if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
            return _factory
        stub.__getattr__ = _hg
        sys.modules["transformers.integrations.hub_kernels"] = stub
    try:
        import torchvision.io  # noqa
    except Exception as e:
        import types
        print(f"[stub] torchvision: {e}", flush=True)
        for m in [m for m in list(sys.modules) if m == "torchvision" or m.startswith("torchvision.")]:
            del sys.modules[m]
        tv = types.ModuleType("torchvision"); tv.__path__ = []; tv.__version__ = "0.0-stub"
        tio = types.ModuleType("torchvision.io")
        class ImageReadMode:
            UNCHANGED = 0
        tio.ImageReadMode = ImageReadMode
        tio.decode_image = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
        tv.io = tio
        ttrans = types.ModuleType("torchvision.transforms")
        class InterpolationMode:
            NEAREST = 0; NEAREST_EXACT = 1; BILINEAR = 2; BICUBIC = 3
            BOX = 4; HAMMING = 5; LANCZOS = 6
        ttrans.InterpolationMode = InterpolationMode
        def _tg(n):
            if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
            return type(n, (), {})
        ttrans.__getattr__ = _tg
        tfunc = types.ModuleType("torchvision.transforms.functional")
        def _un(*a, **k): raise RuntimeError("stub")
        def _fg(n):
            if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
            return _un
        tfunc.__getattr__ = _fg
        ttrans.functional = tfunc; tv.transforms = ttrans
        sys.modules["torchvision"] = tv; sys.modules["torchvision.io"] = tio
        sys.modules["torchvision.transforms"] = ttrans
        sys.modules["torchvision.transforms.functional"] = tfunc

def build_sft_ids(tokenizer, prompt, response, max_len=128):
    p = tokenizer.encode(prompt, add_special_tokens=False)
    r = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    if not any(l != -100 for l in labels):
        raise ValueError("no supervised tokens")
    return ids, [1] * len(ids), labels

def pad_batch(tokenizer, rows):
    L = max(len(r[0]) for r in rows)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": [[*a, *[pad] * (L - len(a))] for a, _, _ in rows],
        "attention_mask": [[*m, *[0] * (L - len(m))] for _, m, _ in rows],
        "labels": [[*l, *[-100] * (L - len(l))] for _, _, l in rows],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/simin/local_models/gpt2")
    ap.add_argument("--tiers", default="20,100,2000")
    ap.add_argument("--ranks", default="4,8,16")
    ap.add_argument("--out", default="/tmp/vm60_qlora_train.json")
    args = ap.parse_args()
    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    _stub()
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[load] {args.model} 4bit-nf4 -> {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, quantization_config=qconf, low_cpu_mem_usage=True,
            attn_implementation="eager")
    except Exception as e:
        print(f"[load] local miss, trying online: {str(e)[:200]}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=False,
            quantization_config=qconf, low_cpu_mem_usage=True, attn_implementation="eager")
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    class LoRAWrapper(nn.Module):
        def __init__(self, orig, in_f, out_f, rank, alpha, dropout):
            super().__init__()
            self.orig = orig; self.scaling = alpha / rank
            self.drop = nn.Dropout(dropout)
            self.A = nn.Parameter(torch.empty(rank, in_f))
            self.B = nn.Parameter(torch.zeros(out_f, rank))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            for p in self.orig.parameters(): p.requires_grad = False
        def forward(self, x):
            base = self.orig(x)
            delta = (self.drop(x).to(self.A.dtype) @ self.A.T @ self.B.T) * self.scaling
            return base + delta.to(base.dtype)

    def dims_of(mod):
        if isinstance(mod, nn.Linear): return mod.in_features, mod.out_features
        try: return int(mod.weight.shape[0]), int(mod.weight.shape[1])
        except Exception: return None

    train_rows = [build_sft_ids(tok, d["prompt"], d["response"]) for d in TINY_DATA[:10]]
    val_rows = [build_sft_ids(tok, d["prompt"], d["response"]) for d in TINY_DATA[10:]]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    tiers_out = []
    for steps in tiers:
        rows = []
        for rk in ranks:
            replaced = {}
            for name, mod in list(model.named_modules()):
                if not any(s in name for s in ["c_attn", "c_proj"]): continue
                if isinstance(mod, LoRAWrapper): continue
                d = dims_of(mod)
                if d is None: continue
                in_f, out_f = d
                wrap = LoRAWrapper(mod, in_f, out_f, rk, 2 * rk, 0.05).to(dev, dtype)
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
            for step in range(steps):
                batch = pad_batch(tok, [train_rows[step % len(train_rows)], train_rows[(step + 1) % len(train_rows)]])
                ids = torch.tensor(batch["input_ids"], device=dev)
                msk = torch.tensor(batch["attention_mask"], device=dev)
                lab = torch.tensor(batch["labels"], device=dev)
                out = model(input_ids=ids, attention_mask=msk)
                shift, shift_lab = out.logits[..., :-1, :].float(), lab[..., 1:].contiguous()
                loss = loss_fn(shift.reshape(-1, shift.size(-1)), shift_lab.view(-1))
                loss.backward(); opt.step(); opt.zero_grad()
                if step == 0: first = float(loss.detach())
                last = float(loss.detach())
            wall = (time.perf_counter() - t1) * 1000 / max(1, steps)
            model.eval()
            with torch.no_grad():
                vb = pad_batch(tok, val_rows)
                ids = torch.tensor(vb["input_ids"], device=dev)
                msk = torch.tensor(vb["attention_mask"], device=dev)
                lab = torch.tensor(vb["labels"], device=dev)
                out = model(input_ids=ids, attention_mask=msk)
                shift, shift_lab = out.logits[..., :-1, :].float(), lab[..., 1:].contiguous()
                vloss = float(loss_fn(shift.reshape(-1, shift.size(-1)), shift_lab.view(-1)))
            peak = float(torch.cuda.max_memory_allocated() / 1024 ** 2) if dev == "cuda" else 0.0
            with torch.no_grad():
                q = tok(TINY_DATA[0]["prompt"], return_tensors="pt").to(dev)
                g = model.generate(**q, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
                gen = tok.decode(g[0], skip_special_tokens=True)[:200]
            rows.append({"rank": rk, "alpha": 2 * rk, "dropout": 0.05,
                "targets": ["c_attn", "c_proj"], "trainable_params": int(n_train),
                "train_loss_first": round(first, 4), "train_loss_last": round(last, 4),
                "val_loss": round(vloss, 4), "step_time_ms": round(wall, 2),
                "peak_mem_mb": round(peak, 1), "simulated": False, "sanity_gen": gen})
            print(f"[qlora] steps={steps} rank={rk} train {first:.4f}->{last:.4f} val={vloss:.4f} peak={peak:.0f}MB", flush=True)
            for name, orig in replaced.items():
                parent = model
                parts = name.split(".")
                for a in parts[:-1]: parent = getattr(parent, a)
                setattr(parent, parts[-1], orig)
        rows.sort(key=lambda r: r["val_loss"])
        tiers_out.append({"steps": steps, "best": rows[0]["rank"], "rows": rows})
    rep = {"mode": "qlora_train", "model": args.model, "base": "bnb-nf4-4bit", "device": dev,
        "tiers": tiers_out, "simulated": False,
        "environment": {"host": platform.node(), "python": sys.version.split()[0]}}
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved to {args.out}", flush=True)
    print("PASS qlora_train", flush=True)

if __name__ == "__main__":
    main()
