"""Task1 附加实验：固定 GPU/workload 下，物化 Attention 与 SDPA 的 prefill 规模扫描。

只回答一个问题：序列变长时，中间 score 矩阵带来的延迟与峰值显存如何增长，
以及 SDPA（PyTorch 内部走 flash / memory-efficient backend）把这条曲线压到什么位置。
结果不命名为 FlashAttention-2/3/4 的性能结论，也不代替重复 benchmark。
"""
import json, math, platform, time
from pathlib import Path

import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16
BATCH, HEADS, HEAD_DIM = 1, 8, 64
CAUSAL = True
WARMUP, ITERS = 5, 20
SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]


def naive_attention(q, k, v, causal):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -float('inf'))
    return torch.softmax(scores, dim=-1) @ v


def sdpa_attention(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def measure(fn, q, k, v, causal):
    for _ in range(WARMUP):
        fn(q, k, v, causal)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out = None
    for _ in range(ITERS):
        out = fn(q, k, v, causal)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return out, {
        'latency_ms': round(elapsed * 1000 / ITERS, 3),
        'peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('需要 CUDA 环境')
    torch.manual_seed(42)
    device = torch.device('cuda')
    rows = []
    for seq in SEQ_LENS:
        shape = (BATCH, HEADS, seq, HEAD_DIM)
        try:
            q = torch.randn(shape, device=device, dtype=DTYPE)
            k = torch.randn(shape, device=device, dtype=DTYPE)
            v = torch.randn(shape, device=device, dtype=DTYPE)
        except torch.cuda.OutOfMemoryError:
            rows.append({'seq_len': seq, 'status': 'OOM(inputs)'})
            continue
        row = {
            'seq_len': seq,
            'score_matrix_mb_theory': round(BATCH * HEADS * seq * seq * 2 / 2**20, 2),
        }
        outputs = {}
        for name, fn in (('naive', naive_attention), ('sdpa', sdpa_attention)):
            try:
                out, metrics = measure(fn, q, k, v, CAUSAL)
                outputs[name] = out
                row[name] = {**metrics, 'status': 'ok'}
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row[name] = {'latency_ms': None, 'peak_allocated_mb': None, 'status': 'OOM'}
        if len(outputs) == 2:
            diff = (outputs['naive'].float() - outputs['sdpa'].float()).abs().max().item()
            row['max_abs_error'] = round(diff, 6)
            if row['naive']['latency_ms'] and row['sdpa']['latency_ms']:
                row['speedup'] = round(row['naive']['latency_ms'] / row['sdpa']['latency_ms'], 2)
                row['peak_mem_ratio'] = round(row['naive']['peak_allocated_mb'] / row['sdpa']['peak_allocated_mb'], 2)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        del q, k, v, outputs
        torch.cuda.empty_cache()

    result = {
        'schema_version': 'task1-prefill-scaling/v1',
        'config': {
            'dtype': str(DTYPE), 'batch': BATCH, 'heads': HEADS, 'head_dim': HEAD_DIM,
            'causal': CAUSAL, 'warmup': WARMUP, 'iters': ITERS, 'seq_lens': SEQ_LENS,
        },
        'environment': {
            'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'python': platform.python_version(),
        },
        'rows': rows,
        'note': '单卡固定 workload 的单次测量；naive 与 SDPA 采用同一批输入，误差来自 bf16 归约顺序。',
    }
    out_path = Path('benchmarks/results/task1_prefill_scaling.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved', out_path)


if __name__ == '__main__':
    main()
