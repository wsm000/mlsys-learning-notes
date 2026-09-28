# Task4 证据清单 · 推理侧 KV Cache 与显存容量（Issue #177）

环境：vm-60 · NVIDIA RTX 4090 D 24 GB · torch 2.9.1+cu128 · transformers 5.16.1 · Python 3.10.12
上游基准：datawhalechina/llm-algo-leetcode commit e518638（2026-09-27T03:07:23Z）
打卡选择：4.1 + 4.2 + 4.3（全量）。issue 截止时间 09-26 12:00，本帖为截止后补交（issue 仍 open）。

## 实测实验（vm-60 真机）

| 文件 | 内容 | 结果 |
|---|---|---|
| task4_exp1_growth.json / .log | 账本 vs 真实分配、六维线性、MHA/GQA/MQA 实测、MLA latent、FP8/INT8 误差 | ALL_TESTS_PASS |
| task4_exp2_paged.json / .log | 分页机制单测、构造性外部碎片（真机 OOM）、同分布碎片对比、块大小扫描、gather 注意力等价 | ALL_TESTS_PASS |
| task4_exp3_prefix.json / .log | 块哈希链前缀缓存机制、真实模型 prefill 跳过与数值等价、存储共享、连续前缀必要性 | ALL_TESTS_PASS |
| task4_exp4_capacity.json / .log | 容量规划（权重/缓冲实测）、准入压力（MHA/GQA/MQA）、同 workload 缓存策略对照 | ALL_TESTS_PASS |

## 上游 notebook（补全 TODO 后 vm-60 执行）

| 文件 | 对应节 | 说明 |
|---|---|---|
| task4_11_KV_Cache_and_Memory_Growth_solved.ipynb | Part01·11 | 直接执行，断言通过 |
| task4_22_vLLM_PagedAttention_solved.ipynb | Part02·22 | TODO 1-5（容量账本/原子分配/跨块扩容/占用报告/逻辑恢复），机制测试全过 |
| task4_34_Prefix_Cache_Matching_and_Reuse_solved.ipynb | Part02·34 | TODO 1-5（规范化/登记去重/最长命中/拆分/账本） |
| task4_24_SGLang_RadixAttention_solved.ipynb | Part02·24 | TODO 1-6（LCP/边分裂/匹配/拆分） |
| task4_66_Inference_Performance_Comparison_solved.ipynb | Part02·66 | TODO 1-3（波次/瓶颈分类/策略决策） |
| task4_69_Prefix_Caching_Benchmark_solved.ipynb | Part02·69 | TODO 1-3（连续命中/LRU/排序）；另修复上游 5.5 单元格缺失的 `import json` |
| notebook_task4_*.log | 以上 | nbconvert 执行日志（stdout） |

补全方法：各 notebook 自带"参考实现"单元格，用其替换题目单元格后整体 `nbconvert --execute`（kernel python3）。执行产物 notebook 内含全部输出与 ✅ 测试行。

## 证据图（实测数据排版，非终端截图）

- 01_growth_ledger.png：增长账本与每 token 成本
- 02_paged_tradeoff.png：碎片对比 / 块大小权衡 / 构造性 OOM / 注意力等价
- 03_prefix_cache.png：TTFT 收益与规模依赖 / 存储共享 / 连续前缀
- 04_capacity.png：容量规划 / 准入实测 / 策略对照与决策
- task4_runshot.png：四实验 + notebook 总览

## 复现

```bash
# vm-60
cd courses/06-llm-algo-leetcode-memory/code
python3 task4_exp1_growth.py && python3 task4_exp2_paged.py \
  && python3 task4_exp3_prefix.py && python3 task4_exp4_capacity.py
python3 task4_render_evidence.py --evidence ../evidence/task4
# notebook：python3 -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \
#   --ExecutePreprocessor.kernel_name=python3 task4_*_solved.ipynb
```

## 口径声明

- 计时：CUDA 边界（synchronize + perf_counter），warmup 后取样本；独立 stage 不替代端到端墙钟。
- 显存：torch.cuda.max_memory_allocated / memory_reserved；nvidia-smi 口径在文中单独注明。
- 真实模型为随机初始化的教学形状；MLA/FP8/INT8 行按账本推导 + 张量误差实测，kernel 级收益需 backend 复测。
- 同一 workload 对照只改一个变量；改变 workload 的结论（如缩小 batch）单独记录，不混入主表。
