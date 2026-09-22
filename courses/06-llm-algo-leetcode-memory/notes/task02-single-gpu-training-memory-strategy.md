# 202609 Task2：单机训练显存策略（vm-60 实验笔记）

> 对应 DataWhale issue：[#168](https://github.com/datawhalechina/llm-algo-leetcode/issues/168)；教程仓库：[datawhalechina/llm-algo-leetcode](https://github.com/datawhalechina/llm-algo-leetcode)。

## 1. 先回答问题

显存紧张时，按代价从轻到重排查：micro-step → checkpoint → CPU-GPU activation offload。先缩小单次前向的激活峰值；若时间预算允许，再用重算换保存；最后才把 activation 搬到 CPU，因为它会引入 PCIe 传输、同步和 CPU 内存占用。

## 2. 最小打卡的关键理解

### 梯度累积

有效 batch = micro_batch × accumulation_steps。每个 micro-batch 都执行 forward/backward，但只在累积完成后 optimizer.step()；梯度在一轮开始前 zero_grad(set_to_none=True)，不能在中间清掉。loss 要除以 accumulation_steps，否则同一个有效 batch 的梯度会被放大 N 倍，等价于把学习率放大 N 倍。

它省的是激活峰值，不是参数/梯度/优化器常驻状态；吞吐也可能下降，因为小矩阵更难打满 GPU、kernel/同步次数更多。

### Activation checkpointing

普通前向把反向所需的中间 activation 留在 GPU；checkpoint 只保留边界输入，backward 时重新执行该段 forward。结果是激活显存下降，代价是额外 FLOPs 和 step time，且随机算子要注意 RNG 一致性。

### Activation offload

offload 保存的仍是反向要用的 activation，但把它从 GPU 搬到 CPU，反向前再搬回 GPU。checkpoint 不保存中间值、反向重算；offload 保存中间值、只是换存储位置。因此 offload 通常更省 GPU 峰值，但多了 D2H/H2D 传输、同步、CPU 内存和 PCIe 带宽压力。

## 3. 统一对照实验

代码固定模型、序列长度、effective batch、优化器和随机种子，只改变策略；指标同时记录 torch.cuda.max_memory_allocated()、reserved memory、step time 和 loss。不能只看峰值下降：还要检查 loss 是否正常、吞吐/时间代价、CPU 内存与传输是否成为新瓶颈。

运行：

    python3 code/task2_memory_strategy.py --out-dir evidence/task2

产物：evidence/task2/task2_results.json、evidence/task2/task2_run.log、evidence/task2/task2_memory_strategy_runshot.png。

## 4. 启发式复盘

- 如果 baseline 单个 micro-batch 就 OOM：先减 micro-batch；accumulation 维持 effective batch。
- 如果峰值主要来自长序列 activation：优先 checkpoint，观察时间是否在预算内。
- 如果仍差一点且 CPU/PCIe 有余量：尝试 offload，并核对 D2H/H2D 是否真正发生。
- 如果模型状态本身就放不下：这三招不够，需要 dtype/量化、optimizer 8-bit、ZeRO/FSDP 或模型并行；不要把 activation 技巧误当成模型状态分片。

## 5. 温故知新

Task1 的账本告诉我：参数、梯度、optimizer state 是常驻项，activation 是随 workload 变化的峰值项；Task2 则把“峰值从哪里来”推进到“用什么代价换空间”。因此每次优化都要写清楚：省掉了哪一类对象，代价转移到哪里，是否值得。
