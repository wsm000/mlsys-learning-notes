"""Plot task1_prefill_scaling.json as prefill scaling curves (latency + peak memory).

Labels are ASCII-only: the remote host lacks CJK fonts.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = json.loads(Path('benchmarks/results/task1_prefill_scaling.json').read_text(encoding='utf-8'))
rows = [r for r in data['rows'] if 'naive' in r and r['naive']['status'] == 'ok']
seq = [r['seq_len'] for r in rows]
naive_t = [r['naive']['latency_ms'] for r in rows]
sdpa_t = [r['sdpa']['latency_ms'] for r in rows]
naive_m = [r['naive']['peak_allocated_mb'] for r in rows]
sdpa_m = [r['sdpa']['peak_allocated_mb'] for r in rows]
theory = [r['score_matrix_mb_theory'] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(seq, naive_t, 'o-', label='naive (materialized score)', color='#c0392b')
axes[0].plot(seq, sdpa_t, 's-', label='SDPA (flash backend)', color='#2471a3')
axes[0].set_xscale('log', base=2); axes[0].set_yscale('log')
axes[0].set_xlabel('seq_len'); axes[0].set_ylabel('latency (ms, log)')
axes[0].set_title('Single attention call latency'); axes[0].grid(True, which='both', alpha=.3); axes[0].legend()

axes[1].plot(seq, naive_m, 'o-', label='naive peak allocated', color='#c0392b')
axes[1].plot(seq, sdpa_m, 's-', label='SDPA peak allocated', color='#2471a3')
axes[1].plot(seq, theory, '^--', label='score matrix (theory, bf16)', color='#7d3c98', alpha=.7)
axes[1].set_xscale('log', base=2); axes[1].set_yscale('log')
axes[1].set_xlabel('seq_len'); axes[1].set_ylabel('peak memory (MB, log)')
axes[1].set_title('Peak allocated memory'); axes[1].grid(True, which='both', alpha=.3); axes[1].legend()

config = data['config']; env = data['environment']
fig.suptitle(f"{env['device']} | torch {env['torch']} | {config['dtype']} | B={config['batch']} H={config['heads']} D={config['head_dim']} causal={config['causal']}")
fig.tight_layout()
out = Path('task1_prefill_scaling.png')
fig.savefig(out, dpi=140)
print('saved', out)
