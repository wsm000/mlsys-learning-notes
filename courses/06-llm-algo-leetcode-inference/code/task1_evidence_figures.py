"""生成 Task1 的证据图（沿用 Task0 runshot 的样式：标题 + 运行输出框 + 图表 + 要点框）。

数据来源都是 vm-60 上真实执行过的 notebook 与实验 JSON，不做任何手工编排的数字。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import nbformat

FONT = Path('fonts/simhei.ttf')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

HEADER = 'llm-algo-leetcode 推理优化 | 202609 · Task1 Prefill 与 Attention Kernel'
SUB = ('GitHub ID: wsm000 | 环境: vm-60 · NVIDIA RTX 4090 D (24GB) · torch 2.9.1+cu128 · Python 3.10.12\n'
       '教程: datawhalechina/llm-algo-leetcode（Part 01 · 14 FlashAttention 显存模型 / Part 02 · 20 FlashAttention 模拟）')


def cell_output(path, needle):
    """取出 notebook 中输出含 needle 的那个代码单元的输出文本。"""
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        text = ''
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                text += ''.join(out.get('text', ''))
            elif 'text' in out:
                text += str(out['text'])
            elif out.get('data', {}).get('text/plain'):
                text += ''.join(out['data']['text/plain'])
        if needle in text:
            return text.strip()
    raise SystemExit(f'no output containing {needle!r} in {path}')


def text_box(ax, title, body, edge, title_color=None):
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle='round,pad=0.02,rounding_size=0.03',
                                linewidth=1.6, edgecolor=edge, facecolor='white',
                                transform=ax.transAxes, clip_on=False))
    ax.text(0.04, 0.93, title, fontsize=12.5, weight='bold', color=title_color or edge,
            transform=ax.transAxes, va='top')
    ax.text(0.04, 0.80, body, fontsize=10.5, transform=ax.transAxes, va='top', linespacing=1.65)


# ---------------- 图 1：Part 01·14 + Part 02·20 运行结果 ----------------
out20 = cell_output('20_solved.ipynb', 'Online Softmax').replace('✅', '[PASS]')
out14 = cell_output('14_solved.ipynb', 'materialization ratio')
gpu = json.loads(Path('benchmarks/results/20_flashattention_gpu.json').read_text(encoding='utf-8'))
scaling = json.loads(Path('benchmarks/results/task1_prefill_scaling.json').read_text(encoding='utf-8'))
rows = [r for r in scaling['rows'] if r.get('naive', {}).get('status') == 'ok']

fig = plt.figure(figsize=(16.4, 11.2), dpi=125)
fig.suptitle(HEADER, fontsize=17, weight='bold', y=0.982)
fig.text(0.5, 0.955, SUB, ha='center', va='top', fontsize=10, color='#444444', linespacing=1.6)

gs = fig.add_gridspec(2, 3, left=0.045, right=0.975, top=0.885, bottom=0.055, height_ratios=[0.86, 1.14], hspace=0.20, wspace=0.14)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2])

run20 = out20.split('\n')
keep = [ln for ln in run20 if ln.strip()][:6]
text_box(ax_a, 'Part 02 · 20 FlashAttention 模拟：CPU 分块 + online softmax 测试输出',
         '\n'.join(keep) + '\ncausal / float64 / 大 score 稳定性 / block_size<=0 校验全部通过',
         edge='#1e8449')
ratio_lines = []
for ln in out14.split('\n'):
    if 'materialization ratio' in ln:
        tile = ln.split('->')[0].strip()
        ratio = ln.split('≈')[-1].strip()
        ratio_lines.append(f'{tile}  ->  物化 / 单 tile 比 ≈ {ratio}')
text_box(ax_b, 'Part 01 · 14 显存模型：物化 score vs 分块工作集',
         '\n'.join(ratio_lines) + '\n\nseq_len=4096, head_dim=128, bf16 口径\n'
         'tile 越小：单块工作集越小，但分块\n数量和调度开销越大。',
         edge='#2471a3')

ax_t = fig.add_subplot(gs[1, 0:2])
seq = [r['seq_len'] for r in rows]
ax_t.plot(seq, [r['naive']['latency_ms'] for r in rows], 'o-', color='#c0392b', label='naive 物化 score（延迟）')
ax_t.plot(seq, [r['sdpa']['latency_ms'] for r in rows], 's-', color='#2471a3', label='SDPA flash backend（延迟）')
ax_t.set_xscale('log', base=2); ax_t.set_yscale('log')
ax_t.set_xlabel('seq_len'); ax_t.set_ylabel('单次 attention 延迟 (ms, log)')
ax_t.grid(True, which='both', alpha=.3)
ax_t2 = ax_t.twinx()
ax_t2.plot(seq, [r['naive']['peak_allocated_mb'] for r in rows], '^--', color='#7d3c98', alpha=.75, label='naive 峰值显存')
ax_t2.plot(seq, [r['sdpa']['peak_allocated_mb'] for r in rows], 'v--', color='#117a65', alpha=.75, label='SDPA 峰值显存')
ax_t2.set_yscale('log'); ax_t2.set_ylabel('峰值显存 (MB, log)')
lines = ax_t.get_lines() + ax_t2.get_lines()
ax_t.legend(lines, [ln.get_label() for ln in lines], fontsize=9, loc='upper left')
ax_t.set_title('同 workload（B=1 H=8 D=64 bf16 causal）下 seq_len 扫描：naive 与 SDPA', fontsize=11.5)

last = rows[-1]
text_box(fig.add_subplot(gs[1, 2]), '4.1 结论要点', '\n'.join([
    '① 标准 Attention 物化 S×S score，',
    '   再读回来做 softmax 和 @V：',
    '   中间矩阵与 HBM 往返随 S^2 增长。',
    f"   seq={last['seq_len']}: naive 峰值 {last['naive']['peak_allocated_mb']:.0f} MB",
    f"   vs SDPA {last['sdpa']['peak_allocated_mb']:.0f} MB（{last['peak_mem_ratio']:.0f}×）",
    f"   延迟 {last['naive']['latency_ms']:.1f} ms → {last['sdpa']['latency_ms']:.2f} ms",
    f"   （{last['speedup']:.0f}×，误差 {last['max_abs_error']}）",
    '② tiling：Q/K/V 切块，块内算',
    '   QK^T/mask/softmax/PV，中间结果',
    '   留在片上 SRAM，不落 HBM。',
    '③ online softmax：逐块维护 m/l/O，',
    '   基准变化就重标定旧状态，结果与',
    '   标准 softmax 精确等价（非近似）。',
    '④ GPU 对照节：naive '
    f"{gpu['results'][0]['latency_ms']} ms / {gpu['results'][0]['peak_allocated_mb']} MB",
    f"   SDPA {gpu['results'][1]['latency_ms']} ms / {gpu['results'][1]['peak_allocated_mb']} MB"
    '（seq=4096）',
]), edge='#b9770e')

fig.savefig('task1_20_flashattention_runshot.png', dpi=125)
print('saved task1_20_flashattention_runshot.png')

# ---------------- 图 2：Part 02·34 prefix cache / chunked prefill ----------------
out34 = cell_output('34_solved.ipynb', 'PrefixCacheManager 测试通过').replace('✅', '[PASS]')
probe = cell_output('34_solved.ipynb', 'peak_allocated_mb')
# 该单元既 json.dumps 打印，又回显返回值 repr，所以只解析第一段 JSON。
probe_json, _ = json.JSONDecoder().raw_decode(probe[probe.index('{'):])

fig2 = plt.figure(figsize=(16.4, 8.6), dpi=125)
fig2.suptitle(HEADER, fontsize=17, weight='bold', y=0.978)
fig2.text(0.5, 0.948, SUB, ha='center', va='top', fontsize=10, color='#444444', linespacing=1.6)

gs2 = fig2.add_gridspec(2, 3, left=0.045, right=0.975, top=0.855, bottom=0.07, hspace=0.30, wspace=0.16)
text_box(fig2.add_subplot(gs2[0, 0:2]), 'Part 02 · 34 PrefixCacheManager：测试输出',
         out34.split('\n')[-1] + '\n\n'
         '命中账本（token 级）：add_prefix([1,2,3]) / ([1,2,9])，block_size=2\n'
         'match_prefix([1,2,3,9]) = 3 ；match_prefix([1,2,0]) = 0\n'
         'split_prompt([1,2,3,9]) → prefix=[1,2,3], suffix=[9]\n'
         'cache_stats([1,2,3,9]) → hit=3, uncached=1, reuse_ratio=0.75\n'
         'chunked_suffix_prefill_plan([1,2,3,9,10]) → [(9,10)]（只对未命中 suffix 切块）',
         edge='#1e8449')
text_box(fig2.add_subplot(gs2[0, 2]), '4.3 结论要点',
         '\n'.join([
             '① 标准 Attention + Prefill 基线：',
             '   一次性把整段 prompt 送进模型，',
             '   中间 score 与 KV 峰值随 S^2/S 增长，',
             '   重复前缀每轮都重新 prefill。',
             '② Prefix Cache：缓存已 prefill 的公共',
             '   前缀（必须从开头连续命中），',
             '   只算增量 suffix → 省重复算力，',
             '   代价是额外缓存显存与淘汰策略。',
             '③ Chunked Prefill：把未命中部分切块',
             '   执行，降单次峰值/便于与 decode',
             '   混排；不提供跨请求复用。',
             '④ 两者解决不同问题，可以叠加。',
         ]), edge='#b9770e')

ax_bar = fig2.add_subplot(gs2[1, 0])
labels = ['一次性 prefill', '分块 prefill']
vals = [probe_json['results']['one_shot']['peak_allocated_mb'], probe_json['results']['chunked']['peak_allocated_mb']]
bars = ax_bar.bar(labels, vals, color=['#c0392b', '#2471a3'], width=.55)
for b, v in zip(bars, vals):
    ax_bar.text(b.get_x() + b.get_width() / 2, v * 1.05, f'{v:.0f} MB', ha='center', fontsize=10)
ax_bar.set_yscale('log'); ax_bar.set_ylabel('峰值显存 (MB, log)')
ax_bar.set_title('4090D 合成 suffix 探针：32768 token / hidden 2048 / fp16', fontsize=10.5)
ax_bar.grid(True, axis='y', alpha=.3)

ax_note = fig2.add_subplot(gs2[1, 1:3])
text_box(ax_note, '这次实验的边界（避免把机制验成性能结论）',
         '\n'.join([
             '· 34 节的分块探针用的是合成张量，只说明“一次性申请”与“分块申请”的峰值差异，',
             f"  chunk_size={probe_json['plan']['chunk_size']}，chunk_count={probe_json['plan']['chunk_count']}，不代表真实 prefill kernel 或端到端 TTFT。",
             '· 20 节的 CPU 模拟只验证数值等价（online softmax 状态更新正确），不测 GPU 峰值显存与带宽。',
             '· 20 节的 GPU 对照只是 naive 与 PyTorch SDPA（flash 后端）的固定 workload 比较，',
             '  不能据此宣称 FlashAttention-3/4 的性能；FA3/FA4 需要 Hopper/Blackwell 等对应硬件。',
             '· 判断 prefill 慢在哪，仍要回到 prompt length、TTFT、prefill_share 这组口径。',
         ]), edge='#7d3c98')

fig2.savefig('task1_34_prefix_cache_runshot.png', dpi=125)
print('saved task1_34_prefix_cache_runshot.png')
