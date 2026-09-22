# Task3：训练侧测量与预算决策

学习社区：[DataWhale](https://www.datawhale.cn/) · [教程仓库](https://github.com/datawhalechina/llm-algo-leetcode) · [打卡 Issue #169](https://github.com/datawhalechina/llm-algo-leetcode/issues/169)。选择 **4.1 + 4.2**。Issue 标题为 Task3，正文开头仍写 Task2；这里按第 3、4 节训练测量要求完成。

## 4.1 最小打卡

### 如何区分计算、访存和等待

我先判断 GPU 时间花在哪个阶段，再判断该阶段为什么慢。矩阵乘 kernel 占比高只能说明计算算子重要，不能单凭算子名字断言 compute-bound；还需结合 FLOPs、实际带宽、算术强度与硬件上限。逐元素算子、拷贝或优化器占比高则提示访存压力，但证明带宽饱和需要硬件计数器。

| 证据 | 可支持的判断 | 下一步验证 |
|---|---|---|
| 长 GEMM kernel、较高运算量 | 计算是重点调查对象 | 检查 Tensor Core 使用、算术强度与吞吐 |
| 大量逐元素 kernel、读写大而运算少 | 可能受显存带宽限制 | 检查 DRAM 吞吐与融合机会 |
| H2D/D2H memcpy 随 offload 增多 | 代价转移到 PCIe 搬运 | 看搬运耗时与计算是否重叠 |
| CUDA kernel 之间空白、CPU launch 或 synchronize 很长 | 调度/同步可能令设备等待 | 按 CPU/GPU 时间线关联调用和依赖 |

CPU 的 synchronize 时间可能包含等待之前已提交的 GPU 工作，不能直接加到 GPU kernel 时间上当作额外代价。多个 stream 的事件也可能重叠，事件耗时之和不等于墙钟时间。本次单卡无 NCCL 通信，communication 标为不适用，不能虚构多卡通信测量。

### 如何建立可复查 baseline

固定模型结构、初始权重、合成输入与目标、种子、effective batch、sequence length、dtype、AdamW 超参数、warmup 和重复次数；每种候选从同样初始状态开始。先 warmup 触发内核初始化和 Adam 状态分配，再 reset peak stats，测量多次完整训练更新，并保存每次耗时而非只报一次。CUDA 是异步的：完整 step 墙钟计时需要在边界 synchronize；profiler 另跑，避免采样开销污染 benchmark。

梯度累积只改变 micro-batch，同时维持 micro_batch × accumulation_steps 的有效 batch。每个 micro loss 按样本比例缩放，一次有效 batch 只做一次 optimizer.step。不同策略还需通过相同初始状态下的 loss 和参数更新检查。数值接近仅证明短程一致性，不能替代真实数据上的长期收敛与验证集质量。

### 训练阶段如何拆分

- data loading：这里使用预先生成的合成 CPU 张量，记录取 batch 的开销，**不代表磁盘读取、文本解码或真实 DataLoader 性能**。
- transfer：CPU 到 GPU 输入拷贝。
- forward：模型前向及 loss。
- backward：反向传播；checkpoint 重算与 saved-tensor offload 回传会出现在这里。
- optimizer：AdamW 更新，状态是持久显存的一部分。
- communication：单 GPU 不涉及分布式通信。

阶段测量用于解释，完整 step 的独立墙钟基准用于做性能决策。若阶段边界加同步，会改变原本重叠，因此不能用阶段相加取代生产吞吐。

## 4.2 策略与预算

### 显存账本

allocated 是活跃的 PyTorch 张量分配；reserved 是 caching allocator 向设备保留的池，包含可复用空间。reserved - allocated 不能直接叫泄漏，也不能全部归因于碎片。nvidia-smi 还包含 CUDA context 等框架外开销，三者不能混用。

静态账本按真实 tensor 的 numel × element_size 统计参数、梯度、优化器状态；激活和临时缓冲随执行变化。峰值减去常驻状态所得残差还可能包含输入、输出、loss 中间量、算子 workspace，**不能直接称精确激活大小**。不同时间点的峰值不能相加，forward/backward/optimizer 的最大值可能出现在不同阶段。

### 为什么显存下降不一定值得采用

工程目标是在显存上限、吞吐下限和质量门槛同时成立时选方案。如果 baseline 已经放得下且最快，再省显存可能没有当前收益；若 baseline 超预算，则可以接受适度重算或搬运来换可运行性。预算必须留空间给运行时缓存和波动，allocated 合格不等于设备物理容量一定足够。

| 策略 | 主要收益 | 性能或质量代价 | 对照边界 |
|---|---|---|---|
| 梯度累积 | 降低单次激活峰值 | 更多小 kernel、较差硬件利用率 | 固定有效 batch，正确归一化 loss |
| Checkpoint | 减少保存的中间张量 | backward 重算 | 固定模型、输入和有效 batch |
| Saved-tensor offload | 将反向所需保存张量搬到 CPU | H2D/D2H、同步、CPU 内存 | PyTorch save_on_cpu 也可能搬保存的权重，非严格仅激活 |
| 量化 | 压缩权重/部分状态 | 反量化、kernel 支持、数值与收敛风险 | 本次不实测；不能把推理量化效果当训练结论 |
| 直接减小有效 batch | 激活减少 | 改变优化轨迹与每步样本数 | 不与固定 workload 的主表混比 |
| 缩短 sequence | 激活减少，attention 项可能显著下降 | 丢失上下文，任务改变 | 需独立质量评估，本次不实测 |

预算决策先过滤质量不合格，再过滤显存/吞吐不合格，然后在可行候选中选吞吐最高者。改变预算后重新筛选：若赢家随小幅预算调整改变，应报告预算敏感性；没有候选就如实写不可行。这里的预算实验使用人为设置的上限，要求 PyTorch reserved 峰值加 64 MiB 余量不超预算，吞吐至少达到 baseline 的 25%；不宣称真实发生 OOM，也不把 64 MiB 当成所有真实任务都足够的 CUDA context 余量。

## 实测结果与证据

2026-09-20 在 vm-60 RTX 4090 D（24 GB）、PyTorch 2.9.1+cu128 / CUDA 12.8 上执行。模型为 6 层残差 MLP，width=256、hidden=1024；固定 batch=4、seq=512、FP32、seed=73、AdamW lr=0.001。3 次 warmup 后测 9 次更新，另做 3 次阶段诊断与两份独立 profiler trace。全部数值检查通过，日志末尾为 `ALL_TESTS_PASS`。

| 策略 | allocated 峰值 MiB | reserved 峰值 MiB | step 中位数 ms | tokens/s |
|---|---:|---:|---:|---:|
| baseline | 230.09 | 258 | 7.838 | 261,295 |
| 梯度累积 1×4 | 146.62 | 164 | 23.431 | 87,407 |
| checkpoint | 149.12 | 178 | 13.442 | 152,358 |
| saved-tensor offload | 147.12 | 174 | 24.670 | 83,015 |

checkpoint 比 baseline 省约 35.2% allocated，但 step 约慢 71.5%；offload 的峰值与 checkpoint 接近，却明显更慢。预算为 256 MiB 时，reserved + 64 MiB 余量使 baseline 不可行，三种优化候选满足最低 65,324 tokens/s 的门槛，选最快的 checkpoint；512 MiB 时选 baseline；64/128/192 MiB 均无可行候选。结论是预算相关的，而非 checkpoint 永远更优。

baseline 独立阶段诊断：取数据 0.019 ms、H2D 0.223 ms、forward 2.183 ms、backward 3.100 ms、optimizer 2.393 ms。反向是最大阶段，优化器也值得关注；仅凭这些阶段时间不能宣称硬件 compute-bound。checkpoint backward 增至 6.525 ms，与重算机制相符。offload forward 达 17.104 ms；独立 trace 中 memcpy 从 baseline 的 4 次、累计 0.173 ms 增到 66 次、累计 10.784 ms，支持“空间收益换来搬运开销”的判断。trace 统计并非未采样时的端到端耗时，不能直接与基准相加。

静态账本：参数约 12.280 MiB、梯度 12.280 MiB、Adam CUDA 状态 24.561 MiB，合计 49.121 MiB；Adam CPU step 标量共 104 bytes。模型只有约 322 万参数，不能将其性能比例直接外推到完整 LLM。质量检查的 loss 相对误差最大约 6.29e-8，参数更新相对 L2 差最大约 1.74e-5，全部满足阈值；未测长期收敛。

证据目录：`../evidence/task3/`，包含三张 PNG、结果 JSON、原始日志、两份真实 trace 和复现命令。PNG 为实测数据排版图，不冒充原屏截图。

复跑：`CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 task3_training_budget.py --out-dir ./evidence`；出图：`python3 task3_render_evidence.py --evidence ./evidence`。源码位于 `../code/`。

## 复现与资料

- [Part01 13：Profiling](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)
- [Part02 73：训练性能分析](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
- [06：基准测试与权衡决策](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/topic_discussion/memory_performance_tuning/06_benchmark_and_tradeoff_decision.md)
- [76：Checkpoint 与 Offload](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) · [75：预算压缩](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) · [74：Profiling 驱动优化](https://github.com/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)
