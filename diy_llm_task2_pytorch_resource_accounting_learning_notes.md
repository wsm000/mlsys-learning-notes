# DIY-LLM Task 2：PyTorch 与资源核算学习笔记

## 1. 学习主线

本笔记对应 DIY-LLM 第 3 章《PyTorch 与资源核算》。这一章不是单纯记忆 PyTorch API，而是建立一条统一的分析链：

    张量形状
      -> 前向计算
      -> 自动求导
      -> 参数更新
      -> 显存、FLOPs 与运行时间估算

面对任意一段训练代码，都应能回答：张量形状是什么？dtype 和 device 是什么？占多少内存？执行多少浮点运算？瓶颈更接近计算还是带宽？

课程链接：https://datawhalechina.github.io/diy-llm/chapter3/chapter3_pytorch%E4%B8%8E%E8%B5%84%E6%BA%90%E6%A0%B8%E7%AE%97.html

## 2. 张量：形状、dtype 与存储

张量是数据、参数、梯度、优化器状态和激活值的统一表示。

线性层的基本形状为：

    X: (B, D)
    W: (D, K)
    Y = X @ W: (B, K)

其中 B 是 batch 中的样本或 token 数，D 是输入维度，K 是输出维度。忽略 bias 时，该层参数量为：

    P = D x K

张量本体的字节数为：

    Memory = numel x bytes_per_element

常用数值类型：

| dtype | 每元素字节数 |
|---|---:|
| FP32 | 4 |
| FP16 | 2 |
| BF16 | 2 |
| INT8 | 1 |

注意 MB 与 MiB 的口径不同：1 MB 等于 10^6 bytes，1 MiB 等于 2^20 bytes。核算时应明确单位。

## 3. 视图、连续性与额外副本

切片、转置和部分 reshape 操作通常只创建视图，复用原始底层存储；它们不必然复制数据。转置后的张量常是非连续的：

    x = torch.randn(2, 3)
    y = x.T
    z = y.view(6)       # 通常报错

原因是 view 只能在底层存储和 stride 能够无复制重解释形状时成立。修复方式是：

    z = y.contiguous().view(6)

contiguous 会复制数据，因而产生新的存储空间。模型调优时，隐式或显式的连续化副本可能成为额外显存峰值来源。

## 4. 矩阵乘法与 FLOPs

对 X(B, D) @ W(D, K)，输出的每个元素需要约 D 次乘法和 D 次加法。因此：

    Forward FLOPs approximately equals 2 x B x D x K

同一层的参数量是 D x K，所以也可写作：

    Forward FLOPs approximately equals 2 x B x P

这解释了为何 Transformer 的主要计算量通常来自矩阵乘法：线性层、QKV 投影和 FFN 都是大规模 GEMM。逐元素激活、残差加法、LayerNorm 和 Softmax 也有成本，但在粗略训练预算中常不是主项。

单位要严格区分：FLOPs 是总运算次数；FLOP/s 才是每秒吞吐率。若题目把一个 fused multiply-add 记作 1 次操作，数值会与“乘、加各记 1 次”的口径相差约 2 倍，必须先声明计数规则。

## 5. Autograd 与反向传播

设一层线性变换为：

    H = XW

若上游梯度为 dL/dH，则：

    dL/dW = X^T @ (dL/dH)
    dL/dX = (dL/dH) @ W^T

这两个梯度各自都是与前向规模相近的矩阵乘法。因此，当输入也需要梯度时，一层完整反向传播约是前向的两倍，总训练计算约为前向的三倍。

PyTorch 中：

1. requires_grad=True 让张量参与计算图和梯度跟踪；
2. loss.backward() 从损失沿计算图应用链式法则；
3. 叶子参数的梯度累积在 parameter.grad；
4. 中间张量默认不保存 .grad，除非调用 retain_grad()；
5. optimizer.step() 读取梯度并更新参数；
6. 下一步 backward 前必须清除旧梯度。

标准单步训练顺序：

    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()

zero_grad 的关键原因是梯度默认累加。它必须位于本轮 backward 和 step 之间之外；通常放在循环开头最不容易出错。

首层输入若不需要梯度，PyTorch 可以跳过 dL/dX 的计算。因此某个具体两层例子中的反向 FLOPs 不一定严格等于前向的两倍；训练量级估算中的 6 倍规则仍是对大量层的有效主项近似。

## 6. 训练计算量、MFU 与时间估算

对稠密语言模型，处理总计 T 个 token、参数量为 P 时：

    Total training FLOPs approximately equals 6 x P x T

直觉是：前向约为 2PT，反向约为 4PT。该近似主要覆盖大矩阵乘法，忽略注意力、归一化、embedding、优化器更新、通信和系统开销。长上下文时，注意力相关开销会更重要。

模型 FLOPs 利用率定义为：

    MFU = measured FLOP/s / hardware peak FLOP/s

估计训练时间时，应用有效吞吐而不是宣传峰值：

    Time approximately equals Total FLOPs /
        (number of GPUs x per-GPU peak FLOP/s x MFU)

GPU 运算默认是异步的。测量 CUDA kernel 时间时需要在计时前后调用 torch.cuda.synchronize()，并进行预热与多次测量；否则测到的可能只是 CPU 发射异步任务的时间。

## 7. 训练显存：不要只计算参数

训练显存由多个部分组成：

    total memory
      = model states
      + activations
      + temporary buffers and workspaces
      + allocator fragmentation
      + communication buffers when distributed

在朴素 FP32 AdamW 中，每个参数通常需要四份 FP32 状态：

| 状态 | bytes/parameter |
|---|---:|
| 参数 | 4 |
| 梯度 | 4 |
| Adam 一阶矩 m | 4 |
| Adam 二阶矩 v | 4 |
| 合计 | 16 |

因此模型状态的下界是：

    model state memory approximately equals 16 x P bytes

这仍未包括激活值。激活值通常随 batch size、序列长度、层数和隐藏维度增长，是实际 OOM 的常见主因。

多卡核算必须说明并行策略：

- 普通数据并行或 DDP：每张卡都保存完整的参数、梯度和优化器状态；多卡主要增加吞吐，不会自动让单卡模型状态变成原来的 1/N。
- FSDP、ZeRO 或模型并行：部分模型状态或模型层被分片，才能利用总显存容纳更大模型。

因此“8 张 80 GB GPU 有 640 GB 显存”不能直接推出普通 DDP 能训练 40B 参数模型；这个总量级只有在正确的状态分片/模型并行方案和激活预算下才可能接近。

混合精度也不能简单等同于“显存减半”。不同实现可能保留 FP32 主权重、FP32 Adam 状态或额外的缩放状态，必须逐项按实际 dtype 计算。

## 8. 算术强度与 Roofline

算术强度定义为：

    Arithmetic Intensity = FLOPs / bytes moved

Roofline 上限为：

    Attainable performance
      = min(peak FLOP/s, memory bandwidth x arithmetic intensity)

硬件 ridge point 为：

    Ridge point = peak FLOP/s / memory bandwidth

判断规则：

    AI < ridge point   -> memory-bound
    AI >= ridge point  -> compute-bound

ReLU、逐元素加法等操作的 AI 通常较低，性能更容易受显存带宽限制。大矩阵乘法能复用输入与权重，AI 通常较高，更可能接近计算限制。

“FLOPs 更少”不保证更快：一个算子可能少算了，却需要搬运更多数据。对 memory-bound 工作负载，减少读写、提升缓存复用和算子融合往往比提高峰值算力更有效。

## 9. 初始化、数据与训练工程

初始化的目的不是制造随机数，而是让前向激活与反向梯度在层间保持合理尺度。若随机权重尺度不随 fan-in/fan-out 调整，信号会在深层网络中爆炸或消失。

Xavier 初始化的核心直觉是同时考虑输入、输出连接数；常见方差尺度约为：

    variance approximately equals 2 / (fan_in + fan_out)

数据加载、随机种子、checkpoint 与混合精度也属于训练系统的一部分：

- 数据加载不足会导致 GPU 空转；
- 固定随机性有助于复现实验和定位回归；
- checkpoint 应保存模型、优化器、训练步数及必要随机状态；
- mixed precision 可减少部分数据移动、提升 Tensor Core 吞吐，但需检查数值稳定性。

## 10. 自测记录

完成了以下八类问题的书面推导与判断：

1. 张量形状、参数量和 BF16 存储；
2. 线性层前向与完整反向 FLOPs；
3. requires_grad、叶子张量与梯度累积；
4. 使用 6PT、GPU 数和 MFU 的训练时间估算；
5. FP32 AdamW 状态、DDP 复制与 OOM 原因；
6. 转置视图、非连续内存与 contiguous 副本；
7. Roofline 带宽屋顶和计算屋顶判断；
8. 标准训练循环的正确顺序。

自测中的关键校正：

- 134.22 MFLOPs 表示总计算量，不应写作 MFLOP/s；
- 2,097,152 bytes 是 2 MiB，按十进制是 2.097152 MB；
- 梯度累积通常不会复制每个 micro-batch 的梯度，真正容易放大峰值显存的是保留计算图或不必要缓存中间张量；
- zero_grad 可放在上一次 step 之后或下一次 backward 前，但绝不能放在当前 backward 与 step 之间。

## 11. 一页公式表

    Tensor bytes                 = numel x bytes per element
    Linear parameters            = D x K
    X(B,D) @ W(D,K) output       = (B,K)
    Linear forward FLOPs         approximately 2BDK
    Full linear training FLOPs   approximately 6BDK
    LLM training FLOPs           approximately 6PT
    FP32 AdamW model states      approximately 16P bytes
    MFU                          = measured FLOP/s / peak FLOP/s
    Training time                approximately total FLOPs / effective throughput
    Arithmetic intensity         = FLOPs / bytes moved
    Roofline                     = min(peak FLOP/s, bandwidth x AI)
    Ridge point                  = peak FLOP/s / bandwidth

## 12. 一句话总结

PyTorch 训练的每一步都能从线性层 XW 出发理解：形状决定参数量，矩阵乘法决定主导 FLOPs，链式法则带来反向计算，参数/梯度/优化器/激活共同决定显存，而算术强度决定应优先优化计算还是数据搬运。
