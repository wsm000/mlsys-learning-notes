# Task5 证据清单 · 量化与显存容量扩展（Issue #178）

环境：vm-60 · NVIDIA RTX 4090 D 24 GB（24564 MiB）· torch 2.9.1+cu128 · transformers 5.16.1 · torchao 0.15.0 · Python 3.10.12
上游基准：datawhalechina/llm-algo-leetcode commit e518638（与 Task4 相同）
打卡选择：4.1 + 4.2 + 4.3（全量）。issue 截止时间 09-28 12:00，本篇为截止后补交（issue 已被官方标记 completed）。

## 实测实验（vm-60 真机）

| 文件 | 内容 | 结果 |
|---|---|---|
| task5_exp1_object_timing.json / .log | 真实权重三档（169 矩阵 494M 参数：FP16/INT8-pt/INT8-pc/INT4-g128 字节+MSE）、容量账本表、同 7B 两种 workload 账本构成切换 | ALL_TESTS_PASS |
| task5_exp2_w8a16_vs_kv.json / .log | 真实 q_proj W8A16 对照（latency/peak/字节/MSE）、全模型 per-tensor/per-channel W8A16（存储收益、峰值反升、24 层 hidden 误差复合、logits MSE）、固定节省 vs KV 线性增长换算 | ALL_TESTS_PASS |
| task5_exp3_kv_quant_capacity.json / .log | 真实 KV 五档量化误差（int8-pt/int8-row/int4-row/fp8-e4m3）、容量账本（KV 池 20.44 GiB）、构造性准入至 OOM（123→246 请求 ×2.00）、运行时预留项 | ALL_TESTS_PASS |
| task5_exp4_candidates.json / .log | 同 workload 四候选（FP16/W8A16/INT4/W8A16+KV-INT8）：TTFT/TPOT/吞吐/peak/KV/logits 余弦 + accept-tune-reject 决策 | ALL_TESTS_PASS |

## 上游 notebook（补全 TODO 后 vm-60 执行）

| 文件 | 对应节 | 说明 |
|---|---|---|
| task5_21_Quantization_Theory_and_INT4_INT8_solved_executed.ipynb | Part01·21 | 直接执行，3 组测试全过 |
| task5_25_Quantization_W8A16_solved_executed.ipynb | Part02·25 | TODO 1-2；real_gpu + torchao Int8WeightOnlyConfig 探针 |
| task5_40_GPTQ_and_AWQ_Weight_Quantization_solved_executed.ipynb | Part02·40 | TODO 1-5；real_gpu 校准（32 prompt/256 激活），GPTQ vs AWQ |
| task5_41_FP8_and_KV_Cache_Quantization_solved_executed.ipynb | Part02·41 | TODO 1-3；real_gpu 真实 KV + torchao Float8WeightOnlyConfig 探针 |
| task5_66_Inference_Performance_Comparison_solved_executed.ipynb | Part02·66 | TODO 1-3（vLLM backend 未安装，G0/G1 未跑，如实标注） |
| task5_67_Quantized_Inference_and_Deployment_solved_executed.ipynb | Part02·67 | TODO 1-3；artifact 契约四态测试；`decision=pending`（无 vLLM） |
| task5_*_solved.ipynb（未执行版） | 同上 | 题目单元格已换上游参考实现、施加 5 处执行路径修复 |
| notebook_results/*.json | 25/40/41 的 GPU 步骤产物 | real_gpu 测量与 torchao 探针 JSON |
| notebook_summary.json | 六份执行汇总 | errors=0、各含 ✅ 测试行 |

补全方法：各 notebook 自带"参考实现"单元格，用其替换题目单元格后整体 `nbconvert --execute`。67 另需上游 `tools/` 包（已随附于 vm-60 执行目录）。

## 证据图（实测数据排版，非终端截图）

- 01_object_timing.png：权重三档与 scale 成本 / 重建 MSE / 同模型两种 workload 账本 / KV 每 token 成本
- 02_w8a16_vs_kv.png：真实 q_proj 对照 / 全模型存储 vs 峰值 / 固定节省 vs 线性增长 / 逐层误差复合
- 03_kv_quant_capacity.png：真实 KV 字节-误差 / 容量账本并发 / 准入实测 / 运行时预留
- 04_candidates.png：TTFT-TPOT / 吞吐 / 峰值与 KV / 决策门
- task5_runshot.png：四实验 + notebook 总览

## 复现

```bash
# vm-60
cd courses/06-llm-algo-leetcode-memory/code   # 工作目录 ~/projects/memopt-task5/code
python3 task5_exp1_object_timing.py && python3 task5_exp2_w8a16_vs_kv.py \
  && python3 task5_exp3_kv_quant_capacity.py && python3 task5_exp4_candidates.py
python3 task5_render_evidence.py --evidence ../evidence/task5
```

## 口径声明

- 计时：CUDA 边界（synchronize + perf_counter），warmup 后取中位数；单层 0.07 ms 量级受 launch 开销影响，latency 比率不稳定（报告区间），峰值显存稳定复现。
- 显存：`torch.cuda.max_memory_allocated` 为主；exp3D 另测 `memory_reserved` 与分配器 gap。
- 候选对照用随机初始化 toy-12L（同 Task2/4 形状）保证"同权重、单变量"；其权重量化误差是下界，质量侧结论以 exp2B 真实 Qwen 24 层复合为准。
- KV-INT8 为"存储 int8、读取反量化"往返路径，非真实 kernel；vLLM/SGLang 未安装，serving 级数字不报告。
