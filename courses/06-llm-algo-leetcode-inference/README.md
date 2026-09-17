# llm-algo-leetcode-inference

本目录收纳 `llm-algo-leetcode` 的**推理优化篇**（LLM-algo-code / 推理优化）学习记录。教程地址：<https://github.com/datawhalechina/llm-algo-leetcode>

课程形式：共 6 个 Task，按时间节点完成任务并写学习笔记打卡；坚持完成全部打卡可获得带唯一编号的结营证书，优秀学习者另有专属证书和荣誉。

## Task 列表

> 主题以教程仓库为准，开始每个 Task 后在此登记并链接对应笔记。

| Task | 主题（待学习后回填） | 笔记 |
|---|---|---|
| Task 0（预习） | 请求结构与指标：Attention/MHA/GQA/MLA、Prefill 与 Decode、TTFT 与 TPOT、Roofline 两阶段分析（[Issue #146](https://github.com/datawhalechina/llm-algo-leetcode/issues/146)） | [task00-request-structure-and-metrics.md](notes/task00-request-structure-and-metrics.md) · [代码](code/04_solved.ipynb) · [截图](evidence/task0/task0_04_attention_runshot.png) |
| Task 1 | Prefill 与 Attention Kernel：FlashAttention 的 tiling 与 online softmax、HBM/SRAM 的作用、FA1-4 演进与硬件要求、Chunked Prefill 与 Prefix Cache（[Issue #147](https://github.com/datawhalechina/llm-algo-leetcode/issues/147)） | [task01-prefill-and-attention-kernel.md](notes/task01-prefill-and-attention-kernel.md) · [代码](code/20_solved.ipynb) · [截图](evidence/task1/task1_20_flashattention_runshot.png) |
| Task 2 | 单请求 Decode 与生成策略：KV Cache 增长账本、解码策略（Greedy/Temperature/Top-k/Top-p）、投机解码、多 Token 解码、推理性能指标口径（[Issue #165](https://github.com/datawhalechina/llm-algo-leetcode/issues/165)） | [task02-single-request-decode-and-generation-strategies.md](notes/task02-single-request-decode-and-generation-strategies.md) · [代码](code/21_solved.ipynb) · [截图](evidence/task2/task2_21_decoding_runshot.png) |
| Task 3 | 解码算法 | 待写 |
| Task 4 | 待定 | 待写 |
| Task 5 | 量化推理与部署 | 待写 |
| Task 6 | 待定 | 待写 |

## 内容索引

- Task 0 · 请求结构与指标 → [笔记](notes/task00-request-structure-and-metrics.md)
- Task 1 · Prefill 与 Attention Kernel → [笔记](notes/task01-prefill-and-attention-kernel.md)｜补全并跑通 [Part 01 · 14](code/14_solved.ipynb)、[Part 02 · 20](code/20_solved.ipynb)、[Part 02 · 34](code/34_solved.ipynb)
- Task 2 · 单请求 Decode 与生成策略 → [笔记](notes/task02-single-request-decode-and-generation-strategies.md)｜补全并跑通 [Part 01 · 11](code/11_solved.ipynb)、[Part 02 · 21](code/21_solved.ipynb)、[Part 02 · 23](code/23_solved.ipynb)、[Part 02 · 35](code/35_solved.ipynb)、[Part 02 · 66](code/66_solved.ipynb)、[Part 02 · 68](code/68_solved.ipynb)｜自加实验 [KV 账本](code/task2_exp1_kv_ledger.py)、[采样策略](code/task2_exp2_sampling.py)、[指标口径](code/task2_exp3_metrics.py)、[KV 复用](code/task2_exp3b_kv_reuse.py)

### Task 1 证据清单

| 证据 | 文件 | 说明 |
|---|---|---|
| Part 02 · 20 CPU 模拟运行 | [20_solved.ipynb](code/20_solved.ipynb) · [runshot](evidence/task1/task1_20_flashattention_runshot.png) | 补全 TODO 1–6 + causal 扩展；数值等价、causal、float64、稳定性、非法 block_size 校验全部通过 |
| Part 01 · 14 工作集模型 | [14_solved.ipynb](code/14_solved.ipynb) | 物化 score 与分块工作集的理论数量级（tile 64/128/256） |
| Part 02 · 20 GPU 对照 | [20_flashattention_gpu.json](evidence/task1/20_flashattention_gpu.json) | 4090D、bf16、S=4096、causal：naive 3.153 ms / 556 MB vs SDPA 0.234 ms / 32 MB |
| prefill 规模扫描（自加实验） | [task1_prefill_scaling.py](code/task1_prefill_scaling.py) · [json](evidence/task1/task1_prefill_scaling.json) · [图](evidence/task1/task1_prefill_scaling.png) | S=512→16384：naive 峰值显存 18.9→8544 MB，SDPA 13.2→104.6 MB |
| Part 02 · 34 Prefix Cache / Chunked Prefill | [34_solved.ipynb](code/34_solved.ipynb) · [runshot](evidence/task1/task1_34_prefix_cache_runshot.png) | PrefixCacheManager 测试通过；合成探针一次性 256 MB vs 分块 4 MB |

### Task 2 证据清单

| 证据 | 文件 | 说明 |
|---|---|---|
| Part 01 · 11 KV Cache 与显存增长 | [11_solved.ipynb](code/11_solved.ipynb) · [runshot](evidence/task2/task2_11_kv_cache_runshot.png) · [log](evidence/task2/run_11_KV_Cache_and_Memory_Growth.log) | 账本公式与三个小验证（该节无 TODO，直接执行） |
| Part 02 · 21 解码策略 | [21_solved.ipynb](code/21_solved.ipynb) · [runshot](evidence/task2/task2_21_decoding_runshot.png) · [log](evidence/task2/run_21_Decoding_Strategies.log) | 补全 temperature / top-k / top-p；ties、非法参数、batch、随机种子断言全通过 |
| Part 02 · 23 投机解码 | [23_solved.ipynb](code/23_solved.ipynb) · [runshot](evidence/task2/task2_23_speculative_runshot.png) · [log](evidence/task2/run_23_Speculative_Decoding.log) | 补全输入契约 / 接受 / residual 修正 / bonus；q=0 与 residual=0 边界通过 |
| Part 02 · 35 多 Token 解码 | [35_solved.ipynb](code/35_solved.ipynb) · [runshot](evidence/task2/task2_35_multi_token_runshot.png) · [log](evidence/task2/run_35_Multi_Token_Decoding.log) | 补全 propose / _accept_token / verify / decode；回退后缀与推进比例断言通过 |
| Part 02 · 66 推理性能比较 | [66_solved.ipynb](code/66_solved.ipynb) · [runshot](evidence/task2/task2_66_inference_metrics_runshot.png) · [log](evidence/task2/run_66_Inference_Performance_Comparison.log) | 补全 7 个函数（请求模拟 → 指标 → 瓶颈 → 对照 → 选型） |
| Part 02 · 68 投机解码基准 | [68_solved.ipynb](code/68_solved.ipynb) · [runshot](evidence/task2/task2_68_speculative_benchmark_runshot.png) · [log](evidence/task2/run_68_Speculative_Decoding_Benchmark.log) | 补全逐轮验证 / 汇总 / 对照 / 决策；G0–G2 实验计划正常输出 |
| 自加实验：KV head 配置 | [task2_exp1_kv_ledger.py](code/task2_exp1_kv_ledger.py) · [json](evidence/task2/task2_exp1_kv_ledger.json) | 4090D 实测 cache 与账本一致；S=8192/B=16 时 MHA 步时 22.6 ms vs GQA/MQA 11.3 ms |
| 自加实验：采样策略 | [task2_exp2_sampling.py](code/task2_exp2_sampling.py) · [json](evidence/task2/task2_exp2_sampling.json) | 8 策略 × 6 prompt × 3 seed：重复率、distinct-2、自困惑度、与 greedy 一致率 |
| 自加实验：指标口径 | [task2_exp3_metrics.py](code/task2_exp3_metrics.py) · [json](evidence/task2/task2_exp3_metrics.json) | prompt 与 batch 扫描下的 TTFT/TPOT/吞吐/峰值显存/e2e |
| 自加实验：KV 复用收益 | [task2_exp3b_kv_reuse.py](code/task2_exp3b_kv_reuse.py) · [json](evidence/task2/task2_exp3b_kv_reuse.json) | 带缓存 vs 每步重算：1.5× → 6.6× → 34× |
| 打卡文案 | [issue165-checkin.md](evidence/task2/issue165-checkin.md) | 提交到 Issue #165 的正文 |

## 学习方式

- 按时间节点完成任务，写学习笔记打卡。
- 打卡从开课后的周一起陆续开放。
- 性能/推理实验同样遵循仓库惯例：记录输入与硬件环境、计时边界、统计口径、瓶颈证据。
