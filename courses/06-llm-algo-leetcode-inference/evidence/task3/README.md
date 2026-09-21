# Task3 · KV Cache 状态与生命周期 —— 实验证据（vm-60 · RTX 4090 D）

Issue: https://github.com/datawhalechina/llm-algo-leetcode/issues/166
环境：vm-60 · NVIDIA RTX 4090 D（24564 MiB）· torch 2.9.1+cu128 · transformers 5.16.1 · Python 3.10.12

## Notebook 解答版（TODO 独立实现，测试全部通过）

| 文件 | 内容 | 测试结果 |
|---|---|---|
| 22_solved.ipynb | Part 02 · 22 vLLM PagedAttention：TODO 1-7（容量账本/prefill 分配/decode 跨块扩容/占用报告/释放复用/按块表恢复）+ Prefix Cache 引用计数扩展 | 12 项断言全过（原子 OOM、OOM 回滚、逻辑顺序恢复、重复释放拒绝等） |
| 24_solved.ipynb | Part 02 · 24 SGLang RadixAttention：TODO 1-6（LCP/insert 边分裂/match_prefix/split_prompt） | 共享边、最长命中、回退逻辑全过 |
| 34_solved.ipynb | Part 02 · 34 Prefix Caching & Chunked Prefill：TODO 1-8（归一化/分块/登记/最长匹配/拆分/账本/后缀计划） | PrefixCacheManager 测试通过 |
| 66_solved.ipynb | Part 02 · 66 推理性能比较（Task2 解答版，Task3 复跑） | 模板校验通过；CPU-first 模式 |
| 69_solved.ipynb | Part 02 · 69 Prefix Caching Benchmark：TODO 1-4（simulate/summarize/compare/recommend） | CPU 模拟与决策测试通过 |

执行日志：22_vLLM_PagedAttention.log / 24_SGLang_RadixAttention.log / 34_Prefix_Caching_and_Chunked_Prefill.log / 66_Inference_Performance_Comparison.log / 69_Prefix_Caching_Benchmark.log
已执行 notebook（含输出）：*_solved_executed.ipynb

## 自加实验（vm-60 实测）

| 脚本 | 问题 | 关键结果 |
|---|---|---|
| task3_exp1_paged_fragmentation.py | 分页 vs 连续预留：碎片账本 + 真实 GPU 分配 | 分页 2072 MiB vs 连续 16384 MiB（省 87.4%，并发 ×7.9）；block_size 8→128 尾块浪费 32→400 token、块数 514→35 |
| task3_exp2_prefix_reuse.py | Radix 前缀复用 + LRU 容量治理 | 146 请求（共享 256 token prompt）：多轮命中最长 373 token；LRU 容量 128 token 命中 0% → 512 token 99.3%（驱逐 209 次）→ 4096 token 零驱逐 |
| task3_exp3_prefix_cache_gpu.py | 真实模型 Prefix Cache 对照（gpt2 124M fp16） | prefill token -70%；串行 TTFT ×1.05~1.07（固定开销摊薄），批量 8 请求 TTFT ×3.23、峰值 -542 MiB；logit 差异与近 tie 边距同量级（输出分布不变） |

日志：run_task3_exp*.log；数据：task3_exp*.json

## 运行截图（打卡用）

- task3_22_paged_attention_runshot.png —— 22 节测试 + block_size 扫描 + GPU 分配实测
- task3_24_radix_attention_runshot.png —— 24 节测试 + 前缀树结构 + GPU 边界探针
- task3_34_prefix_chunked_prefill_runshot.png —— 34 节测试 + 执行账本 + 分块峰值实测
- task3_66_inference_metrics_runshot.png —— 66 节复跑 + 指标口径 + bottleneck 分类
- task3_69_prefix_cache_benchmark_runshot.png —— 69 节测试 + 决策逻辑 + G0/G1 对照
- task3_exp_prefix_reuse_runshot.png —— 三个自加实验汇总

## 证据边界

- CPU 模拟验证机制逻辑，不推真实 TTFT/吞吐；
- 合成 GPU 探针（[token, head_dim] 张量）不代表真实 vLLM/SGLang kernel；
- vLLM 未安装（66 节预检 vllm_on_current_kernel=False），真实 backend 命中率指标留待 serving 环境；
- 命中率证据必须来自 backend metrics/日志，不能只用 TTFT 反推。
