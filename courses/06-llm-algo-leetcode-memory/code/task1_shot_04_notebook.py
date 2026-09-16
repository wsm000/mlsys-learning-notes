# -*- coding: utf-8 -*-
"""把已执行的 04 节 notebook 渲染成"运行截图"（真实 cell 源码 + 真实输出）。"""
import os, sys, json
import nbformat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task1_common import setup_fonts, page, card, save
import matplotlib.pyplot as plt

print("CJK font:", setup_fonts(), flush=True)
nb = nbformat.read("04_solved_executed.ipynb", as_version=4)

code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
solved_i, solved_src, solved_ec = None, None, None
test_i, test_src, test_out, test_ec = None, None, None, None
# 定位"解答版说明"标记之后紧接着的那个 code cell —— 就是真正被补全并参与测试的 cell
anchor = None
for i, c in enumerate(nb.cells):
    if c.cell_type == "markdown" and "解答版说明" in "".join(c.source):
        anchor = i
        break
solved_target = None
if anchor is not None:
    for j in range(anchor + 1, len(nb.cells)):
        if nb.cells[j].cell_type == "code":
            solved_target = j
            break

for i, c in code_cells:
    s = "".join(c.source)
    if i == solved_target or (solved_target is None and "TODO 1: Reshape 为多头形式" in s):
        solved_i, solved_src, solved_ec = i, s, c.get("execution_count")
    if "test_mha_mqa_gqa()" in s and "def test_mha_mqa_gqa" in s:
        test_i, test_src, test_ec = i, s, c.get("execution_count")
        test_out = "".join(o.get("text", "") for o in c.get("outputs", []) if o.output_type == "stream")

assert solved_src and test_out, "notebook 未包含期望内容"

# 只展示 forward 方法本体（TODO 落点都在这里），并去掉行尾空行
body = solved_src[solved_src.index("    def forward("):].rstrip()
body = "\n".join(ln.rstrip() for ln in body.splitlines() if ln.strip())
solved_head = body[:2400]
# emoji 在 Noto CJK 下缺字形，替换成可读标记（notebook 本体不动）
test_out_show = test_out.replace("✅", "[PASS]").strip()

env = "GitHub ID: wsm000    |    微信昵称: empty    |    vm-60 · NVIDIA RTX 4090D (24GB) · torch 2.9.1+cu128"
link = ("教程: datawhalechina/llm-algo-leetcode · Part02 04 Attention MHA GQA    |    DataWhale 社区    |    "
        "Task1 Issue #149")
fig, gs = page("llm-algo-leetcode 显存优化 | 202609 · Task1 硬件与显存账本 —— Part02 04 节 多头注意力 运行截图",
               env, link, nrows=2, ncols=2, figsize=(13.5, 10.2))

ax = fig.add_subplot(gs[0, :])
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#c3cee6")
ax.set_facecolor("white")
ax.text(0.012, 0.975, "In [%s]:  GroupedQueryAttention.forward —— 四个 TODO 已补全（本题解答版，本 cell 实际执行并被下方测试调用）"
        % solved_ec, transform=ax.transAxes, fontsize=10.2, fontweight="bold", color="#16264a", va="top")
ax.text(0.012, 0.915, solved_head, transform=ax.transAxes, fontsize=8.0, family="monospace",
        color="#1f2b45", va="top", linespacing=1.34)

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xticks([]); ax2.set_yticks([])
for s in ax2.spines.values():
    s.set_color("#c3cee6")
ax2.set_facecolor("#fbfcff")
ax2.text(0.03, 0.95, "In [%s]:  运行本题自带测试" % test_ec, transform=ax2.transAxes,
         fontsize=10.2, fontweight="bold", color="#16264a", va="top")
ax2.text(0.03, 0.80, "test_mha_mqa_gqa()", transform=ax2.transAxes, fontsize=8.4, family="monospace",
         color="#1f2b45", va="top")
ax2.text(0.03, 0.66, test_out_show, transform=ax2.transAxes, fontsize=9.0, family="monospace",
         color="#0b5d1e", va="top", linespacing=1.5)
ax2.text(0.03, 0.16,
         "断言覆盖：\n"
         "  · MHA 前向输出形状 [B,S,H*D]\n"
         "  · GQA 前向输出形状（4 个 Q 头 : 2 个 KV 头）\n"
         "  · KV Cache 自回归更新后 [B, H, S_prefill+1, D/H]",
         transform=ax2.transAxes, fontsize=8.4, family="monospace", color="#22304f", va="top", linespacing=1.45)

lines = ["同一份实现在 GPU 上的额外对照（见 task1_43_runshot.png）:",
         "  · 与 torch.nn.functional.scaled_dot_product_attention",
         "    (is_causal=True) 输出最大误差 0.0e+00 (float64)",
         "  · repeat_kv 延迟扩展：缓存只存 num_kv_heads",
         "  · KV cache 账本: MHA 32KB / GQA 4KB / MLA 1.12KB",
         "    (单 token 单层, fp16, LLaMA-2-70B 尺寸)",
         "",
         "配套证据:",
         "  evidence/task1/task1_43_runshot.png   汇总图",
         "  evidence/task1/task1_43.log           运行日志",
         "  code/04_solved.ipynb                  本题解答版 notebook",
         "  code/task1_shot_43.py                 GPU 侧对照脚本"]
card(gs[1, 1], "本节产物与对照", lines)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_43_notebook_runshot.png")
save(fig, out)
print("NOTEBOOK_RUNSHOT_DONE", flush=True)
