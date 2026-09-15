# llm-algo-leetcode-inference

本目录收纳 `llm-algo-leetcode` 的**推理优化篇**（LLM-algo-code / 推理优化）学习记录。教程地址：<https://github.com/datawhalechina/llm-algo-leetcode>

课程形式：共 6 个 Task，按时间节点完成任务并写学习笔记打卡；坚持完成全部打卡可获得带唯一编号的结营证书，优秀学习者另有专属证书和荣誉。

## Task 列表

> 主题以教程仓库为准，开始每个 Task 后在此登记并链接对应笔记。

| Task | 主题（待学习后回填） | 笔记 |
|---|---|---|
| Task 0（预习） | 请求结构与指标：Attention/MHA/GQA/MLA、Prefill 与 Decode、TTFT 与 TPOT、Roofline 两阶段分析（[Issue #146](https://github.com/datawhalechina/llm-algo-leetcode/issues/146)） | [task00-request-structure-and-metrics.md](notes/task00-request-structure-and-metrics.md) · [代码](code/04_solved.ipynb) · [截图](evidence/task0/task0_04_attention_runshot.png) |
| Task 1 | Prefill 与 Attention Kernel：FlashAttention 的 tiling 与 online softmax、HBM/SRAM 的作用、FA1-4 演进与硬件要求、Chunked Prefill 与 Prefix Cache（[Issue #147](https://github.com/datawhalechina/llm-algo-leetcode/issues/147)） | [task01-prefill-and-attention-kernel.md](notes/task01-prefill-and-attention-kernel.md) · [代码](code/20_solved.ipynb) · [截图](evidence/task1/task1_20_flashattention_runshot.png) |
| Task 2 | 待定 | 待写 |
| Task 3 | 解码算法 | 待写 |
| Task 4 | 待定 | 待写 |
| Task 5 | 量化推理与部署 | 待写 |
| Task 6 | 待定 | 待写 |

## 内容索引

- Task 0 · 请求结构与指标 → [笔记](notes/task00-request-structure-and-metrics.md)
- Task 1 · Prefill 与 Attention Kernel → [笔记](notes/task01-prefill-and-attention-kernel.md)｜补全并跑通 [Part 01 · 14](code/14_solved.ipynb)、[Part 02 · 20](code/20_solved.ipynb)、[Part 02 · 34](code/34_solved.ipynb)

### Task 1 证据清单

| 证据 | 文件 | 说明 |
|---|---|---|
| Part 02 · 20 CPU 模拟运行 | [20_solved.ipynb](code/20_solved.ipynb) · [runshot](evidence/task1/task1_20_flashattention_runshot.png) | 补全 TODO 1–6 + causal 扩展；数值等价、causal、float64、稳定性、非法 block_size 校验全部通过 |
| Part 01 · 14 工作集模型 | [14_solved.ipynb](code/14_solved.ipynb) | 物化 score 与分块工作集的理论数量级（tile 64/128/256） |
| Part 02 · 20 GPU 对照 | [20_flashattention_gpu.json](evidence/task1/20_flashattention_gpu.json) | 4090D、bf16、S=4096、causal：naive 3.153 ms / 556 MB vs SDPA 0.234 ms / 32 MB |
| prefill 规模扫描（自加实验） | [task1_prefill_scaling.py](code/task1_prefill_scaling.py) · [json](evidence/task1/task1_prefill_scaling.json) · [图](evidence/task1/task1_prefill_scaling.png) | S=512→16384：naive 峰值显存 18.9→8544 MB，SDPA 13.2→104.6 MB |
| Part 02 · 34 Prefix Cache / Chunked Prefill | [34_solved.ipynb](code/34_solved.ipynb) · [runshot](evidence/task1/task1_34_prefix_cache_runshot.png) | PrefixCacheManager 测试通过；合成探针一次性 256 MB vs 分块 4 MB |

## 学习方式

- 按时间节点完成任务，写学习笔记打卡。
- 打卡从开课后的周一起陆续开放。
- 性能/推理实验同样遵循仓库惯例：记录输入与硬件环境、计时边界、统计口径、瓶颈证据。
