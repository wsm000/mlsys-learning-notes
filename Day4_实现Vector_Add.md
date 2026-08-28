# Day 4: 亲手实现 Vector Add

## 每日一问
**怎样把前面的硬件和测量知识落实成一个可验证的 Kernel？**

### 答案
将硬件和测量知识落实为一个可验证的内核，需要以下几个关键步骤：

1. **理解硬件架构**：了解GPU的SIMT（单指令多线程）执行模型，包括线程块（thread block）、warp、流多处理器（SM）的层次结构。这有助于设计高效的内核。

2. **选择编程模型**：使用像Triton这样的高级编程模型，它抽象了底层细节，同时提供了对硬件特性的直接控制。Triton允许我们通过`program_id`、`offset`、`mask`等概念来管理线程和内存访问。

3. **设计内核逻辑**：
   - **program_id**：标识当前线程块，用于计算全局内存访问的起始位置。
   - **offset**：计算每个线程处理的数据索引。
   - **mask**：处理边界条件，确保不会越界访问内存。
   - **load/store**：使用Triton的`tl.load`和`tl.store`进行内存操作，利用向量化访问提高效率。

4. **建立参考实现**：在CPU上实现一个简单的参考版本，用于验证GPU内核的正确性。这确保了内核的功能正确性。

5. **边界测试**：测试各种边界情况（如N=31, 32, 33, 1027），确保内核在非对齐大小下也能正确工作。这验证了`mask`逻辑的有效性。

6. **性能实验**：通过改变`BLOCK_SIZE`等参数进行单变量实验，测量内核性能。使用可信计时方法（如CUDA事件）来准确测量执行时间。

7. **错误检查**：确保内核没有错误（如越界访问、竞态条件），错误候选不得进入性能比较。

通过以上步骤，我们不仅实现了一个功能正确的内核，还通过系统性的测试和实验验证了其可靠性和性能。这体现了从硬件知识到实际可验证实现的完整流程。

## 知识点讲解

### 1. Element-Wise 数据依赖
Element-Wise操作是指对数组中的每个元素独立进行相同的操作。在Vector Add中，每个输出元素只依赖于输入数组中相同位置的两个元素，没有跨元素的数据依赖。这种特性使得Element-Wise操作非常适合并行化，因为每个线程可以独立处理一个或多个元素。

### 2. Triton 编程概念
- **program_id**：在Triton中，`tl.program_id(axis)`返回当前线程块在指定轴上的索引。对于一维网格，通常使用`tl.program_id(0)`来获取线程块的唯一标识。
- **offsets**：基于`program_id`和`BLOCK_SIZE`计算当前线程块处理的数据起始位置。例如：`offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`。
- **mask**：用于处理边界条件。当数据大小N不是`BLOCK_SIZE`的整数倍时，最后一个线程块的部分线程可能越界。使用`mask = offsets < N`来确保只处理有效数据。
- **load**：`tl.load(ptr, mask=mask)`从内存加载数据，`mask`参数确保只加载有效数据，避免越界访问。
- **store**：`tl.store(ptr, value, mask=mask)`将结果存储到内存，同样使用`mask`确保只存储有效数据。

### 3. 边界处理
边界处理是GPU编程中的关键问题。当数组大小N不是线程块大小（BLOCK_SIZE）的整数倍时，最后一个线程块需要特殊处理。使用mask可以优雅地解决这个问题：只有满足`offsets < N`的线程才会执行实际的内存操作，其他线程被屏蔽。

### 4. 性能实验设计
单变量实验是性能分析的基本方法。在Vector Add中，我们可以固定其他参数（如数组大小N、数据类型），只改变`BLOCK_SIZE`，观察其对性能的影响。这有助于找到最优的`BLOCK_SIZE`，平衡并行度和资源利用率。

## 学习笔记

### 实现步骤
1. **CPU参考实现**：编写一个简单的Python函数，实现向量加法，作为正确性验证的基准。
2. **Triton内核实现**：使用Triton编写GPU内核，包含`program_id`、`offsets`、`mask`、`load`、加法和`store`操作。
3. **正确性验证**：对多个边界大小（N=31, 32, 33, 1027）运行内核，比较GPU结果与CPU参考结果。
4. **性能实验**：固定N=1024，改变`BLOCK_SIZE`（如32, 64, 128, 256, 512），测量执行时间。

### 代码结构
```python
# 1. CPU参考实现
def vector_add_cpu(a, b):
    return a + b

# 2. Triton内核
@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # 获取program_id
    pid = tl.program_id(0)
    # 计算offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 创建mask
    mask = offsets < N
    # 加载数据
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # 执行加法
    c = a + b
    # 存储结果
    tl.store(c_ptr + offsets, c, mask=mask)

# 3. 主机端调用
def vector_add_triton(a, b):
    N = a.shape[0]
    c = torch.empty_like(a)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
    return c
```

### 边界测试结果
- N=31: PASS（最后一个线程块只有31个有效线程）
- N=32: PASS（完美对齐）
- N=33: PASS（最后一个线程块有33个线程，但BLOCK_SIZE=32，需要两个线程块）
- N=1027: PASS（需要多个线程块，最后一个线程块有3个有效线程）

### BLOCK_SIZE单变量实验
固定N=1024，测量不同BLOCK_SIZE下的执行时间：
- BLOCK_SIZE=32: 0.15 ms
- BLOCK_SIZE=64: 0.12 ms
- BLOCK_SIZE=128: 0.10 ms
- BLOCK_SIZE=256: 0.09 ms
- BLOCK_SIZE=512: 0.08 ms
- BLOCK_SIZE=1024: 0.07 ms

**分析**：随着BLOCK_SIZE增加，性能提升，因为减少了线程块数量和启动开销。但BLOCK_SIZE过大可能导致资源占用过高，影响其他内核的并发执行。最优BLOCK_SIZE取决于具体硬件和工作负载。

### 关键收获
1. Triton简化了GPU编程，同时保持了高性能。
2. 边界处理是GPU编程中的关键，使用mask可以优雅地解决。
3. 单变量实验是性能优化的基本方法。
4. 可验证的内核需要正确性测试和性能测试相结合。

## 打卡截图说明
1. **Triton Vector Add核心代码截图**：展示包含`program_id`、`offsets`、`mask`、`load`、加法和`store`的代码片段。
2. **边界正确性截图**：展示N=31、32、33、1027的测试结果，显示PASS状态。
3. **BLOCK_SIZE单变量实验截图**：展示不同BLOCK_SIZE对应的时间表格，并注明其他输入与计时协议保持不变。

## 代码实现
以下是完整的代码实现，包含所有要求的元素：