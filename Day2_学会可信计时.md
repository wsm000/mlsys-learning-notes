# Day 2：学会可信计时

> Day 2 的必做核心对应教程第 5 章。第 6 章的 rocprofv3 内容在文末作为预习保留：它能交叉验证 Event 结果，但不能替代本日的完整计时协议。

## Gate 0：打卡清单

- [ ] 已固定同一 Vector Add baseline 的输入规模、dtype、输入值和输出缓冲区。
- [ ] 已记录无 benchmark warmup 的单次同步墙钟结果。
- [ ] 已记录一个明确标为错误的“只测 CPU enqueue”结果。
- [ ] 已执行 warmup 后的多次 GPU Event 测量，并保留 mean、median、min、max、p95 和标准差。
- [ ] 已在结果旁写明计时边界、同步位置和统计方式。
- [ ] 已完成每日一问，并保存两项打卡截图。

## 本日要回答的问题

**怎样证明测到的是 Kernel 时间，而不是 CPU 提交或首次运行开销？**

不能只看一个很小的墙钟数字。先固定输入和测量对象，执行 warmup 以减少首次调用和短期状态转换带来的影响；随后在同一 GPU stream 中用 GPU Event 包住目标操作，并在读取 Event 时间前同步。warmup 可能使缓存变热，但不能保证时钟频率、缓存或页表状态完全一致，因此这些仍应作为受控但可能波动的实验条件记录。对多次独立样本报告 mean、median、min 等统计量，并把 Event 结果与第 6 章 profiler 的 Kernel trace 交叉检查。只用 CPU 墙钟包住异步 launch、却不等待 GPU 完成时，测到的主要是 CPU enqueue 时间，不能叫 Kernel 时间。

## 先明确“测什么”

本练习只测已分配张量上的 <code>torch.add(a, b, out=c)</code>。下列工作不属于计时边界：张量分配、随机/固定输入构造、正确性检查、打印、文件写入和统计计算。

| 项目 | 固定口径 |
| --- | --- |
| 操作 | float32 Vector Add，<code>c = a + b</code> |
| 输入 | a 的全部元素为 1，b 的全部元素为 2，预先分配 c |
| 默认规模 | N=16,777,216；每个数组约 64 MiB |
| warmup | 20 次，不纳入统计；之后同步 |
| repeat | 100 个独立样本 |
| inner | 每个样本连续执行 20 次并取每次平均，减少 Event 边界在极短操作中的相对影响；比较实验时必须保持不变 |
| 主计时方法 | GPU Event 位于同一默认 stream，Event 后同步再读取 elapsed time |
| 主统计 | count、mean、median、min、max、p95、标准差 |

Vector Add 每个元素约读 8 B、写 4 B、做 1 次浮点加法，算术强度约为 0.083 FLOP/B；多数情况下受显存带宽限制。性能数字必须附带实际 GPU、ROCm/PyTorch 版本、N 和完整协议，不能横向照搬教程样机结果。

## 三种结果分别说明什么

| 输出标签 | 计时边界 | 能说明什么 | 能否当作 Kernel 时间 |
| --- | --- | --- | --- |
| <code>single_once_no_benchmark_warmup_cpu_wall_ms</code> | CPU <code>perf_counter</code> 包住一次 add 和结束同步 | 一次 <code>torch.add</code> 加同步的墙钟区间；不含此前的张量分配和输入构造，但可能混入首次被测 add 的状态影响与 CPU 提交 | 不可以；仅作对比。 |
| <code>incorrect_cpu_enqueue_only_ms</code> | CPU <code>perf_counter</code> 只包住一次 add，之后才同步清队列 | 异步提交开销 | **不可以**；脚本故意把它标为 INVALID。 |
| <code>warmup_gpu_event_per_torch_add_ms</code> | GPU Event 包住 warmup 后的 inner 次 add 批次，完成后同步，再除以 inner | 固定条件下每次 <code>torch.add</code> 的 GPU Event 平均时间 | 可以作为本练习的主结果。 |

Event 结果描述的是 Event 边界内实际提交的 GPU 工作。每个样本是一批 inner 次 <code>torch.add</code>，脚本将该批时间除以 inner；若框架或编译器把一个源操作实现为多个 dispatch，结果测到的是这段 GPU 工作的总时间，而不是自动等同于“一个 Kernel”。第 6 章的 profiler 可以用来确认实际 Kernel trace，不能仅凭源码名称臆断。p95 使用 nearest-rank 定义：排序后取第 <code>ceil(0.95 * repeat)</code> 个样本。

## 运行计时脚本

ROCm PyTorch 仍通过 <code>torch.cuda</code> 使用设备，这是正常接口命名。脚本会拒绝没有 ROCm/HIP 的 PyTorch，避免把 CPU 或非目标环境结果误当作 ROCm 结果。

~~~bash
python3 hello-gpu/day2_timing/benchmark_vector_add.py \
  --n 16777216 \
  --warmup 20 \
  --repeat 100 \
  --inner 20
~~~

运行前可先确认版本和 GPU 可见性：

~~~bash
python3 -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
~~~

完整终端输出会显示实际 device、PyTorch、HIP、输入、计时边界、同步规则、三个结果组和统计值。保存完整文本，不能只截取 min 这一项。

## 常见错误计时方法

~~~python
start = time.perf_counter()
torch.add(a, b, out=c)
end = time.perf_counter()
~~~

这是错误示例：GPU launch 默认异步，<code>end - start</code> 常常只反映 CPU 把工作放入队列的耗时。另一个常见错误是把 <code>torch.randn</code>、张量分配或 <code>torch.allclose</code> 放进循环；这改变了被测对象。第三个错误是把第一次调用当作稳态性能，它可能包含 JIT、模块加载、冷缓存和时钟状态变化。

## 第 6 章预习：用 trace 交叉验证

学习指南将第 6 章放在 Day 3 的性能瓶颈阶段。完成本日 Event 计时后，可在教程源仓库的 <code>code/part1-profiling/chapter6</code> 中运行其 Vector Add 和 <code>rocprofv3 --kernel-trace</code>，检查 Kernel 名称、Grid/线程划分与 Kernel 时间。trace 的作用是让“Event 边界内发生了什么”可见；它不应取代 Day 2 要求的固定输入、warmup、repeat 和统计报告。

连续访问与跨步访问的对照实验也要谨慎解释：若一次改动同时改变地址布局、每线程工作量或 Grid Size，就不能把所有差异都归因于“合并访存”。应控制单一变量并记录协议。

## 打卡材料

1. 同一 Vector Add baseline 的单次结果和 warmup + repeat 结果对比截图。截图中要能看出两者使用的计时方法不同。
2. 完整协议截图：至少包含 N、dtype、固定输入、warmup、repeat、inner、同步/GPU Event、mean、median、min 等统计结果。
3. 结果旁的一句话：**warmup 后以 GPU Event 对所选 repeat 个独立样本报告 mean、median、min 的结果更可信，因为它减少了首次调用影响，并由 GPU 时间戳界定了完成的 Vector Add 工作区间；只测 CPU enqueue 不代表 Kernel 时间。**

将截图和未裁剪的原始文本输出放入 [打卡材料](打卡材料/README.md) 指定的位置。
