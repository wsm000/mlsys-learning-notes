# Day 1：认识 GPU 并跑通程序

> 对应教程第 1--4 章。目标是把 <code>c[i] = a[i] + b[i]</code> 从“公式”落到一个可验证的 GPU Kernel，并准备 Day 1 的两项打卡证据。

## Gate 0：打卡清单

- [ ] 已阅读教程第 1--4 章，并能说出 CPU 与 GPU 的并行取舍。
- [ ] 已在目标环境确认 HIP、ROCm/PyTorch 和 GPU 可见。
- [ ] 已编译并运行本目录的 HIP Vector Add，得到真实的 <code>RESULT: PASS</code>。
- [ ] 已运行 N=10、Block=4 的线程映射程序。
- [ ] 已保存环境验证与线程映射两张打卡截图。
- [ ] 已完成每日一问的回答。

本工作区尚未检测到 ROCm/HIP 工具链，上述运行项必须在实际 AMD ROCm 环境执行后再勾选；不要复制教程样机的 GPU 名称或性能数字作为自己的结果。

## 本日要回答的问题

**<code>c[i] = a[i] + b[i]</code> 怎样被 GPU 并行执行？**

一次 Kernel launch 创建一个 Grid；Grid 由多个 Block 组成，每个 Block 有多个 Thread。第 <code>blockIdx.x</code> 个 Block 内的第 <code>threadIdx.x</code> 个 Thread 计算自己的全局下标：

~~~cpp
const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
if (i < n) {
  c[i] = a[i] + b[i];
}
~~~

因此每个有效线程只处理一个元素：线程 <code>i</code> 读取 <code>a[i]</code> 和 <code>b[i]</code>，写回 <code>c[i]</code>。线程之间没有数据依赖，可以同时执行；GPU 通过大量线程隐藏全局显存访问的延迟。

## GPU 并行模型速记

| 名词 | 在本练习中的含义 | 容易混淆的点 |
| --- | --- | --- |
| CPU | 少量复杂核心，擅长分支复杂、低延迟的控制流。 | CPU 也能并行，但通常不以海量相同元素操作为主要吞吐模型。 |
| GPU | 大量计算 lane，擅长大量相似操作和高吞吐。 | GPU 并非任何任务都会更快；小任务、频繁同步和复杂分支会削弱优势。 |
| CU | AMD 的计算资源单元；其上调度并执行 wavefront。 | 一个 Block/workgroup 不等于一个 CU，也不保证与某个 CU 一一对应。 |
| Wavefront | 一组以同一条指令流推进的线程 lane；实际宽度取决于架构与编译配置。 | 不要把某张卡的 wave32 推广为所有 AMD GPU 都固定 wave32。 |
| SIMT | 单指令、多线程：lane 共享指令流，各自持有数据与部分状态。 | 分支条件不同会造成发散，部分 lane 暂时被屏蔽。 |
| Grid | 一次 Kernel launch 的所有 Block。 | Grid 的 Block 数由问题规模和 Block 大小决定。 |
| Block / workgroup | 彼此可协作的一组 Thread。 | HIP 的 Block 对应 AMD 语境中的 workgroup。 |
| Thread / work-item | 一个逻辑工作项；本例对应一个数组下标。 | 一个 Thread 不是一个独占的物理核心。 |

## N=10、Block=4：线程如何映射到数据

本例需要 <code>grid = ceil(10 / 4) = 3</code> 个 Block。最后一个 Block 正常会多启动两个线程；它们的职责是检查边界后立即返回，绝不能读写数组。

| Block | Thread | 全局下标 i | 动作 |
| --- | ---: | ---: | --- |
| 0 | 0 | 0 | 计算 c[0] = a[0] + b[0] |
| 0 | 1 | 1 | 计算 c[1] = a[1] + b[1] |
| 0 | 2 | 2 | 计算 c[2] = a[2] + b[2] |
| 0 | 3 | 3 | 计算 c[3] = a[3] + b[3] |
| 1 | 0 | 4 | 计算 c[4] = a[4] + b[4] |
| 1 | 1 | 5 | 计算 c[5] = a[5] + b[5] |
| 1 | 2 | 6 | 计算 c[6] = a[6] + b[6] |
| 1 | 3 | 7 | 计算 c[7] = a[7] + b[7] |
| 2 | 0 | 8 | 计算 c[8] = a[8] + b[8] |
| 2 | 1 | 9 | 计算 c[9] = a[9] + b[9] |
| 2 | 2 | 10 | **越界：直接返回，不读写。** |
| 2 | 3 | 11 | **越界：直接返回，不读写。** |

运行下列程序可生成同一张可截图的终端表格：

~~~bash
python3 hello-gpu/day1_vector_add/thread_mapping.py --n 10 --block 4
~~~

越界保护不是“偶发异常”的补丁，而是向上取整分配 Grid 的必然结果。删去 <code>if (i &lt; n)</code> 后，尾块的 i=10、11 会访问不属于数组的全局显存，造成未定义行为或数据损坏。

## 显存层次与合并访存

从近到远可粗略理解为：每线程寄存器、每个 workgroup 可共享的 LDS/shared memory、缓存层次、全局 GDDR 显存。Vector Add 的 a、b、c 位于全局显存；它的算术强度约为 1 FLOP / 12 B，因此通常是 memory-bound。

相邻 lane 访问相邻的 <code>a[i]</code>、<code>b[i]</code> 和 <code>c[i]</code> 时，硬件能以较少内存事务合并这些访问，带宽利用更高。若相邻 lane 跳着访问很远的地址，事务数会增加，速度通常下降。连续访问不是“保证最快”，但它是 Vector Add 的正确基线布局。

## 环境验证

教程参考环境是 ROCm Linux；教程样机使用 RX 9070 XT / gfx1201 / ROCm 7.13，只能作为参考，不是本机应复制的输出。若使用教程源仓库，可先完成其三道验证：

~~~bash
git clone https://github.com/datawhalechina/hello-gpu.git hello-gpu-tutorial
cd hello-gpu-tutorial/code/part0-intro
uv sync
source ./activate-rocm.sh
hipcc --version
rocminfo | grep -E "^[[:space:]]*(Name|Marketing Name|Vendor Name|Device Type|Compute Unit):" | head -20
python chapter1/check_torch_rocm.py
~~~

<code>rocminfo</code> 输出中应能看到 <code>Device Type: GPU</code>。ROCm PyTorch 仍使用 <code>torch.cuda</code> 这一历史接口；<code>torch.cuda.is_available() == True</code> 与 <code>torch.version.hip</code> 有值才说明 PyTorch ROCm 路径可用。WSL2 中 <code>rocm-smi</code> 或硬件计数器不可用并不必然代表 HIP/PyTorch 不可用。

## 运行本仓库的 Vector Add

下面的程序独立于上游教程代码，目的是生成完整、可验证的 Day 1 证据。它会打印实际 GPU、架构、CU 数、N、Block、Grid，并逐项验证结果。

~~~bash
mkdir -p hello-gpu/build
hipcc -O3 -std=c++17 hello-gpu/day1_vector_add/vector_add_hip.cpp -o hello-gpu/build/vector_add_hip
./hello-gpu/build/vector_add_hip 10 4
./hello-gpu/build/vector_add_hip 10000000 256
~~~

第一条运行命令应与映射图对应；第二条给出更接近日常规模的完整正确性验证。截图必须包含实际运行命令以及程序输出中的设备信息和 <code>RESULT: PASS</code>，不能仅截取最终一行。

## 打卡材料

1. 环境验证与 Vector Add 成功截图：同一截图或连续终端记录应显示 GPU/ROCm 信息、编译或运行命令、N/Block/Grid 和正确性结果。
2. N=10、Block=4 的线程映射截图：使用上表或 <code>thread_mapping.py</code> 输出，并清楚显示 Block 0 对应 0--3、Block 1 对应 4--7、Block 2 对应 8--11，以及 i=10、11 越界。
3. 将截图、完整原始文本输出和简短说明放到 [打卡材料](打卡材料/README.md) 指定的位置。

## 每日一问：回答示例

GPU 以一个 Thread 对应一个数组下标的方式并行执行 Vector Add：Thread 先由 <code>blockIdx.x * blockDim.x + threadIdx.x</code> 计算 i，再在 <code>i &lt; N</code> 时读取 a[i]、b[i] 并写入 c[i]。Grid 按向上取整覆盖全部 N 个元素，所以尾块会出现多余线程；这些线程必须因边界判断退出，避免越界访问。相邻线程处理相邻 i，使全局显存访问尽可能合并。
