# -*- coding: utf-8 -*-
"""Task5（量化与显存容量扩展）公共模块。

与 Task4 同一套账本口径；新增：
  - 权重量化工具：per-tensor / per-channel / per-group INT8，INT4 打包（uint8 两值一字节）
  - FP8 e4m3 表示（torch.float8_e4m3fn，真实硬件格式）
  - 教学版 W8A16Linear（与上游 02·25 参考实现一致）
  - AWQ 通道保护 + GPTQ 逐列误差补偿（与上游 02·40 参考实现同构的简化版）
  - KV Cache 量化：per-tensor INT8 / 分组 INT8 / FP8 E4M3
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

QMAX_INT8 = 127


def source_sha256(path):
    """脚本自身 SHA256 前 16 位，写进结果 JSON 保证可追溯到代码版本。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- 账本公式
def ledger_weight_bytes(num_params, bytes_per_param):
    return num_params * bytes_per_param


def per_token_kv_bytes(layers, kv_heads, head_dim, dtype_bytes, batch=1):
    return 2 * layers * kv_heads * head_dim * dtype_bytes * batch


def mib(x):
    return x / 1024 ** 2


def gib(x):
    return x / 1024 ** 3


# ---------------------------------------------------------------- 通用量化
def absmax_scale(x, qmax=QMAX_INT8, eps=1e-8):
    absmax = torch.max(torch.abs(x))
    safe = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    return (qmax / safe).clamp_min(1.0 / eps if eps is None else eps)


def quantize_per_tensor(x, qmax=QMAX_INT8):
    """对称 per-tensor INT8；scale 为标量张量。"""
    scale = absmax_scale(x, qmax)
    q = torch.clamp(torch.round(x * scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_per_tensor(q, scale):
    return q.to(torch.float32) / scale


def quantize_per_channel(x, axis=0, qmax=QMAX_INT8):
    """按 output channel（axis=0）对称 INT8（权重矩阵 [out, in] 的每行一个 scale）。"""
    reduce_dims = tuple(i for i in range(x.ndim) if i != axis)
    absmax = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    scale = qmax / absmax
    q = torch.clamp(torch.round(x * scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_per_channel(q, scale):
    # scale = qmax / absmax（乘子口径），反量化 = q / scale
    return q.to(torch.float32) / scale


def quantize_per_group(x, group_size=128, qmax=15):
    """按行内 group 对称 INT4（qmax=15 -> 4bit 有符号）；返回 int8 容器（教学口径，未打包）。"""
    out, in_f = x.shape
    pad = (group_size - in_f % group_size) % group_size
    xp = F.pad(x, (0, pad)) if pad else x
    grouped = xp.reshape(out, -1, group_size)
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = qmax / absmax
    q = torch.clamp(torch.round(grouped * scale), -qmax, qmax).to(torch.int8)
    return q, scale, pad  # q: [out, n_groups, group_size]


def dequantize_per_group(q, scale, pad, in_f):
    out = q.shape[0]
    # scale = qmax / absmax（乘子口径），反量化 = q / scale
    rec = (q.to(torch.float32) / scale).reshape(out, -1)
    return rec[:, :in_f]


def pack_int4(q):
    """两个 int4（int8 容器）打包进一个 uint8（低 4 位在前）。"""
    flat = q.flatten()
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])
    pairs = flat.reshape(-1, 2)
    low = (pairs[:, 0] & 0x0F).to(torch.uint8)
    high = (pairs[:, 1] & 0x0F).to(torch.uint8)
    return (low | (high << 4))


def int4_packed_bytes(q):
    return pack_int4(q).numel() * 1  # uint8


def quantize_fp8_e4m3(x):
    """真实 FP8 E4M3 表示（硬件格式），往返到 float32。"""
    q = x.to(torch.float32).to(torch.float8_e4m3fn)
    return q.to(torch.float32)


def quantize_fp8_kv(kv, group=None):
    """KV FP8 per-tensor（e4m3）。KV 是 [.., last_dim]，整块一个 scale 的实际由 e4m3 自身指数承担。"""
    return quantize_fp8_e4m3(kv)


# ---------------------------------------------------------------- W8A16
class W8A16Linear(nn.Module):
    """教学版 W8A16：INT8 权重 + scale buffer，前向反量化到激活 dtype 后 F.linear。

    与上游 02·25 参考实现一致：存 INT8、算 FP16/BF16。
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def from_float(self, linear_layer):
        with torch.no_grad():
            q, scale = quantize_per_tensor(linear_layer.weight.detach().float())
            self.weight_int8.copy_(q)
            self.scale.copy_(scale)
            if linear_layer.bias is not None:
                self.bias.copy_(linear_layer.bias.detach().float())

    def forward(self, x):
        w_fp = self.weight_int8.to(dtype=x.dtype)
        w_dequant = w_fp / self.scale.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype)
        return F.linear(x, w_dequant, bias)


def w8a16ify(model):
    """把模型中所有 nn.Linear 原地替换为 W8A16Linear（权重转 INT8+scale，bias 保持浮点）。

    返回被替换的 Linear 数量。Embedding / LayerNorm 等不碰。
    """
    n = 0
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.Linear):
            q = W8A16Linear(mod.in_features, mod.out_features).to(mod.weight.device)
            q.from_float(mod)
            setattr(model, name, q)
            n += 1
        else:
            n += w8a16ify(mod)
    return n


class W8A16LinearPC(nn.Module):
    """per-channel INT8 权重版（每个输出通道一个 scale）：与 per-tensor 版对照粒度收益。"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.register_buffer('weight_int8', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features, 1))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def from_float(self, linear_layer):
        with torch.no_grad():
            w = linear_layer.weight.detach().float()
            q, s = quantize_per_channel(w)  # s: [out,1] 乘子口径
            self.weight_int8.copy_(q)
            self.scale.copy_(s)
            if linear_layer.bias is not None:
                self.bias.copy_(linear_layer.bias.detach().float())

    def forward(self, x):
        w_dequant = (self.weight_int8.to(torch.float32) / self.scale).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_dequant, bias)


def w8a16ify_pc(model):
    n = 0
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.Linear):
            q = W8A16LinearPC(mod.in_features, mod.out_features).to(mod.weight.device)
            q.from_float(mod)
            setattr(model, name, q)
            n += 1
        else:
            n += w8a16ify_pc(mod)
    return n


# ---------------------------------------------------------------- AWQ/GPTQ 教学版
@torch.no_grad()
def awq_protect_quantize(weight, act_rms, group_size=16, protect_ratio=0.15, qmax=QMAX_INT8):
    """AWQ：按激活 RMS 找 top-k 显著通道，原值保护（不量化），其余分组 INT8。

    weight: [out, in]；act_rms: [in]（校准激活的逐通道 RMS）。
    返回 (重建权重, 被保护通道数, scale 张量)。
    """
    out_f, in_f = weight.shape
    k = max(1, int(in_f * protect_ratio))
    topk = torch.topk(act_rms, k).indices
    mask = torch.zeros(in_f, dtype=torch.bool)
    mask[topk] = True
    base = weight.float()
    n_groups = (in_f + group_size - 1) // group_size
    scales = torch.ones(out_f, n_groups)
    recon = base.clone()
    for g in range(n_groups):
        s, e = g * group_size, min((g + 1) * group_size, in_f)
        cols = ~mask[s:e]
        if cols.any():
            blk = base[:, s:e][:, cols]
            scale = (blk.abs().max() / qmax).clamp_min(1e-8)
            q = torch.clamp(torch.round(blk / scale), -qmax, qmax)
            recon[:, s:e][:, cols] = (q * scale).to(recon.dtype)
            scales[:, g] = scale
    # 显著通道保持浮点原值（误差精确为 0）
    recon[:, mask] = base[:, mask]
    return recon, int(mask.sum().item()), scales


@torch.no_grad()
def gptq_column_quantize(weight, hessian_diag=None, group_size=16, qmax=QMAX_INT8):
    """GPTQ 教学版：按列顺序量化，用对角 Hessian 近似把已量化列的残差补偿到未量化列。

    hessian_diag: [in]（由校准激活 X^T X 的对角近似）。None 时退化为顺序分组量化。
    """
    out_f, in_f = weight.shape
    w = weight.float().clone()
    n_groups = (in_f + group_size - 1) // group_size
    scales = torch.ones(out_f, n_groups)
    if hessian_diag is None:
        hessian_diag = torch.ones(in_f)
    h = hessian_diag.clone()
    for j in range(in_f):
        col = w[:, j]
        scale = (col.abs().max() / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round(col / scale), -qmax, qmax)
        err = col - q * scale  # 本列量化残差
        w[:, j] = q * scale
        if j + 1 < in_f:
            # 残差按 Hessian 比例补偿给后续列（对角近似）
            w[:, j + 1:] += err.unsqueeze(1) * (h[j + 1:] / h[j].clamp_min(1e-12)).unsqueeze(0)
    return w, scales


# ---------------------------------------------------------------- KV 量化
def kv_quantize_int8(kv, group=None):
    """KV INT8：group=None 为 per-tensor，否则按最后一维分组。返回 (q, scale, meta)。"""
    if group is None:
        q, scale = quantize_per_tensor(kv)
        return q, scale, {'mode': 'per-tensor', 'group': None}
    last = kv.shape[-1]
    flat = kv.reshape(-1, last)
    q, scale, pad = quantize_per_group(flat, group_size=group)
    n_groups = q.shape[1]
    return q, scale, {'mode': 'per-group', 'group': group, 'pad': pad, 'n_groups': n_groups, 'last': last}


def kv_dequantize_int8(q, scale, meta):
    if meta['mode'] == 'per-tensor':
        return dequantize_per_tensor(q, scale).reshape(meta['shape'])
    rec = dequantize_per_group(q, scale, meta['pad'], meta['last'])
    return rec.reshape(meta['shape'])


# ---------------------------------------------------------------- 环境/输出
def env_info():
    props = torch.cuda.get_device_properties(0)
    return {
        'gpu': torch.cuda.get_device_name(0),
        'total_memory_mib': round(props.total_memory / 1024 ** 2, 1),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'python': platform.python_version(),
        'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[json] {path}')


def sync_time_ms(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f'  [ok] {msg}')


def build_llama(num_kv_heads, layers=12, hidden=1024, heads=8, vocab=32000,
                intermediate=2816, dtype=torch.float16, device='cuda', max_position=8192):
    """与 Task4 同一玩具 LLaMA（随机初始化教学形状，只改 num_key_value_heads）。"""
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=num_kv_heads, max_position_embeddings=max_position,
        torch_dtype=dtype,
    )
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device('cpu'):
            model = LlamaForCausalLM(config)
        model = model.to(device).eval()
    finally:
        torch.set_default_dtype(prev)
    return model, config


def seeded_ids(shape, seed, low=0, high=32000, device='cuda'):
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randint(low, high, shape, generator=g).to(device)
