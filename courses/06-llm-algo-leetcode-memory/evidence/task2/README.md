# Task2 证据：单机训练显存策略

环境：vm-60 · NVIDIA RTX 4090 D · torch 2.9.1+cu128 · VRAM 23.53 GiB

## 文件

| 文件 | 用途 |
|---|---|
| `task2_code_runshot.png` | 代码关键片段截图：梯度累积、checkpoint、offload 与指标采集 |
| `task2_terminal_summary.png` | 可读的 vm-60 四方案结果截图，含 `ALL_TESTS_PASS` |
| `task2_memory_strategy_runshot.png` | 峰值显存 / step time 柱状图 |
| `task2_results.json` | 结构化结果 |
| `task2_run_tuned.log` | 完整终端日志 |

## 关键结果

| 方案 | 峰值 allocated | step time | 判定 |
|---|---:|---:|---|
| baseline batch=4 | 1383.34 MiB | 0.2337 s | 对照 |
| micro=1, accum=4 | 475.41 MiB | 0.0324 s | 推荐的第一步 |
| + checkpoint | 343.65 MiB | 0.0449 s | 显存紧张时优先 |
| + activation offload | 343.65 MiB | 0.7035 s | 仅在显存仍不够时 tune |

结论：四种固定 effective batch=4 的实验均通过；loss 保持正常。checkpoint 相对 micro-step 再降约 27.7% 峰值，offload 在本 workload 上未进一步降低 allocated 峰值，却带来明显传输/同步时间代价。

打卡 Issue：[ #168 ](https://github.com/datawhalechina/llm-algo-leetcode/issues/168) · 教程：[llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)
