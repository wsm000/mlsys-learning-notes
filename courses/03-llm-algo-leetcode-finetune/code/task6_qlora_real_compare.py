#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task6 QLoRA真对比(GPT2实权重,vm-60): NF4 vs 均匀INT4量化误差 + bf16/4bit基座显存.
无peft依赖(手写LoRA部分沿用sweep结论);本脚本只做量化侧实测,诚实口径.
用法(vm-60): python3 ~/task6_qlora_real_compare.py --model ~/local_models/gpt2 --out /tmp/vm60_qlora_real_compare.json
"""
import argparse, json, platform, sys, time
from pathlib import Path

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
        def _hub_getattr(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _factory
        stub.__getattr__ = _hub_getattr
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
        def _tv_getattr(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return type(name, (), {})
        ttrans.__getattr__ = _tv_getattr
        tfunc = types.ModuleType("torchvision.transforms.functional")
        def _unavailable(*a, **k):
            raise RuntimeError("torchvision stub: unavailable")
        def _tfunc_getattr(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _unavailable
        tfunc.__getattr__ = _tfunc_getattr
        ttrans.functional = tfunc; tv.transforms = ttrans
        sys.modules["torchvision"] = tv; sys.modules["torchvision.io"] = tio
        sys.modules["torchvision.transforms"] = ttrans
        sys.modules["torchvision.transforms.functional"] = tfunc

def nearest(levels, x):
    best, bd = levels[0], abs(x - levels[0])
    for v in levels[1:]:
        d = abs(x - v)
        if d < bd: best, bd = v, d
    return best

NF4 = [-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,
 -0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,0.16093020141124725,
 0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/simin/local_models/gpt2")
    ap.add_argument("--out", default="/tmp/vm60_qlora_real_compare.json")
    ap.add_argument("--sample", type=int, default=4096)
    args = ap.parse_args()
    _stub()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model} -> {dev} bf16", flush=True)
    tok = AutoModelForCausalLM  # placeholder to keep import order
    from transformers import AutoTokenizer as T
    tokenizer = T.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    def _load_bf16(**kw):
        try:
            return AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager", **kw)
        except Exception as e:
            print(f"[load] local miss, trying online: {e}"[:220], flush=True)
            return AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=False,
                dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager", **kw)
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    m_bf16 = _load_bf16().to(dev).eval()
    mem_bf16 = float(torch.cuda.max_memory_allocated() / 1024 ** 2) if dev == "cuda" else 0.0
    n_params = sum(p.numel() for p in m_bf16.parameters())
    print(f"[bf16] params={n_params} peak={mem_bf16:.1f}MB", flush=True)
    # 取真实c_attn权重做NF4 vs 均匀INT4(采样前N个,保证可复现)
    w = None
    for name, p in m_bf16.named_parameters():
        if "c_attn.weight" in name:
            w = p.detach().float().flatten()[:args.sample]; wname = name; break
    assert w is not None, "c_attn.weight not found"
    vals = w.tolist(); absmax = max(abs(v) for v in vals)
    uni = [-1.0 + i * (2.0 / 15) for i in range(16)]
    se_nf4 = se_uni = 0.0
    for v in vals:
        x = max(-1.0, min(1.0, v / absmax))
        dn = nearest(NF4, x) * absmax; du = nearest(uni, x) * absmax
        se_nf4 += (v - dn) ** 2; se_uni += (v - du) ** 2
    mse_nf4, mse_uni = se_nf4 / len(vals), se_uni / len(vals)
    print(f"[nf4-real] {wname} n={len(vals)} absmax={absmax:.4f} mse_nf4={mse_nf4:.7f} mse_uni={mse_uni:.7f} nf4_win={mse_nf4 < mse_uni}", flush=True)
    # bitsandbytes真实4bit装载显存(GPT2 Conv1D可能不被量化,如实记录)
    mem_4bit, qinfo = None, {}
    try:
        from transformers import BitsAndBytesConfig
        del m_bf16; torch.cuda.empty_cache() if dev == "cuda" else None
        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        try:
            m4 = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                quantization_config=qconf, low_cpu_mem_usage=True, attn_implementation="eager").eval()
        except Exception:
            m4 = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=False,
                quantization_config=qconf, low_cpu_mem_usage=True, attn_implementation="eager").eval()
        mem_4bit = float(torch.cuda.max_memory_allocated() / 1024 ** 2) if dev == "cuda" else 0.0
        from collections import Counter
        c = Counter(str(p.dtype) for p in m4.parameters())
        qinfo = {"dtypes": dict(c)}
        print(f"[4bit] peak={mem_4bit:.1f}MB dtypes={qinfo}", flush=True)
    except Exception as e:
        qinfo = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(f"[4bit] FAIL {qinfo['error']}", flush=True)
    rep = {"mode": "qlora_real_compare", "model": args.model, "device": dev,
        "params": int(n_params), "mem_bf16_mb": round(mem_bf16, 1), "mem_4bit_mb": mem_4bit,
        "nf4_real": {"weight": wname, "n": len(vals), "absmax": round(absmax, 5),
            "mse_nf4": round(mse_nf4, 7), "mse_uniform_int4": round(mse_uni, 7),
            "nf4_win": bool(mse_nf4 < mse_uni)},
        "bnb": qinfo, "simulated": False,
        "environment": {"host": platform.node(), "python": sys.version.split()[0]}}
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:2000], flush=True)
    print(f"saved to {args.out}", flush=True)
    print("PASS qlora_real_compare", flush=True)

if __name__ == "__main__":
    main()
