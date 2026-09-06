# Datawhale 高性能计算训练营 Day 2：OpenMP 并行编程学习笔记

> 本笔记承接 [Day 1：认识 GPU 并跑通程序](Day1_认识GPU并跑通程序.md) 和 [可信计时](Day2_学会可信计时.md)。Day 1 关注 GPU 的线程映射、边界保护和内存访问；本日转向 CPU 共享内存多核并行，使用 OpenMP 完成循环级并行化，并在 vm-60 上实测加速效果。

## 1. 今日目标

- 理解 `parallel`、`for`、`parallel for` 的含义；
- 判断循环是否适合 OpenMP 加速；
- 区分共享变量和私有变量；
- 理解竞态条件以及 `reduction` 的作用；
- 完成 π 计算和像素风格处理的串行/并行版本；
- 用相同输入、相同输出语义和不同线程数比较加速比。

## 2. OpenMP 心智模型

可以把 OpenMP 想成“叫来一组帮厨并分配任务”：

- `#pragma omp parallel`：启动一组线程，线程共同执行一个代码块；
- `#pragma omp for`：把循环迭代分给线程；
- `#pragma omp parallel for`：启动线程并直接分配循环迭代。

典型向量加法：

```cpp
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
}
```

该循环适合并行的原因是：第 `i` 次迭代只读取 `a[i]`、`b[i]`，只写入 `c[i]`，不同迭代之间没有数据依赖，也没有写入同一位置。

编译 GCC/MinGW 程序时必须启用 OpenMP：

```bash
g++ -O3 -fopenmp program.cpp -o program
```

Linux/macOS：

```bash
OMP_NUM_THREADS=8 ./program
```

PowerShell：

```powershell
$env:OMP_NUM_THREADS=8
.\program.exe
```

## 3. 共享变量、私有变量与竞态条件

变量属性的快速判断：

| 变量 | 通常属性 |
|---|---|
| `for` 循环控制变量 `i` | OpenMP 自动私有 |
| 循环体内声明的临时变量 | 每个线程私有 |
| 并行区域外声明的普通变量 | 通常共享 |
| `reduction` 变量 | 线程私有副本，结束时合并 |

危险示例：

```cpp
double sum = 0.0;

#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    sum += value(i);  // 错误：多个线程竞争地读写 sum
}
```

`sum += value(i)` 不是一个不可分割的动作，而是“读取、相加、写回”。如果两个线程同时读取旧值，后写回的线程可能覆盖前一个线程的贡献，这就是竞态条件。结果通常不稳定，不能简单认为一定只会偏小。

正确写法：

```cpp
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; ++i) {
    sum += value(i);
}
```

`reduction(+:sum)` 的语义是：每个线程先累加自己的私有 `sum`，循环结束后再把各线程的部分和相加。

更严格的教学写法是：

```cpp
#pragma omp parallel for default(none) \\
    shared(n, step) reduction(+:sum)
```

`default(none)` 强制显式写出变量属性，可以减少不小心共享变量的错误。

## 4. 练习 1：两个循环的判断

### 4.1 π 计算循环

```cpp
for (long long i = 0; i < n; ++i) {
    double x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
}
```

判断：**适合并行，但必须使用归约。**

- `n = 500000000`，计算量很大；
- 每个迭代都可以独立计算 `x` 和函数值；
- 唯一的共享累加点是 `sum`；
- 因此使用 `reduction(+:sum)` 可以安全并行化。

并行核心代码：

```cpp
#pragma omp parallel for default(none) shared(n, step) reduction(+:sum)
for (long long i = 0; i < n; ++i) {
    const double x = (i + 0.5) * step;
    sum += 4.0 / (1.0 + x * x);
}
```

浮点加法不严格满足结合律，所以串行和并行结果最后几位可能不同；应比较数值误差，而不是要求文本完全一致。

### 4.2 只有 100 次的循环

```cpp
for (int i = 0; i < 100; ++i) {
    sum += i * 0.5;
}
```

判断：**理论上可并行，实际上通常不值得。**

它同样需要 `reduction` 才能保证正确，但线程启动、任务分配、同步和归约的成本，可能高于 100 次简单加法。

因此要区分：

> “能不能并行”与“值不值得并行”是两个不同问题。

## 5. 练习 2：像素风格算法

本次实现使用简单的 PPM `P6` 图片格式，避免引入额外图像库。算法步骤：

1. 将图片划分成 `block_size × block_size` 小块；
2. 对每个小块计算平均 RGB 颜色；
3. 用平均颜色填满该小块；
4. 输出像素风格图片。

每个图像块都是一个独立任务。把块编号成一维循环：

```cpp
int blocks_x = (width + block_size - 1) / block_size;
int blocks_y = (height + block_size - 1) / block_size;
int total_blocks = blocks_x * blocks_y;

#pragma omp parallel for schedule(static)
for (int k = 0; k < total_blocks; ++k) {
    int block_x = k % blocks_x;
    int block_y = k / blocks_x;

    // 计算当前块的边界
    // 计算当前块的平均 RGB
    // 只写当前块覆盖的像素
}
```

安全性的关键：

- 多个线程都可以读取输入图像；
- 每个线程的 `r`、`g`、`b`、`count` 都在循环体内声明，因此是私有的；
- 每个块只写自己的输出矩形；
- 块之间不重叠，所以不会有两个线程写同一个像素；
- 该算法不需要 `reduction`，因为没有跨块共享的累加器。

完整实现位于 [`day2_openmp/pixel_openmp.cpp`](day2_openmp/pixel_openmp.cpp)。

## 6. vm-60 实验结果

### 6.1 π 计算

实验参数：`n = 500000000`，串行与并行版本使用同一台 vm-60、同一输入规模。

| 线程数 | 串行时间 | 并行时间 | 加速比 |
|---:|---:|---:|---:|
| 1 | 0.8495 s | 0.8530 s | 0.996× |
| 2 | 0.8494 s | 0.4265 s | 1.991× |
| 4 | 0.8427 s | 0.2135 s | 3.948× |
| 8 | 0.8531 s | 0.1073 s | 7.947× |

8 线程接近线性加速，说明该任务计算量大、迭代独立，OpenMP 并行化效果很好。

串行和并行结果都接近 π：

```text
serial pi   = 3.141592653589814
parallel pi = 3.1415926535899676
```

并行结果误差约为 `1.75e-13`，属于浮点累加顺序变化导致的正常差异。

### 6.2 像素风格

实验参数：输入图片 `4096×4096`，块大小 `8×8`。

| 线程数 | 串行时间 | 并行时间 | 加速比 | 输出一致性 |
|---:|---:|---:|---:|---|
| 1 | 0.07748 s | 0.07839 s | 0.988× | YES |
| 2 | 0.07715 s | 0.05419 s | 1.424× | YES |
| 4 | 0.07739 s | 0.04219 s | 1.834× | YES |
| 8 | 0.07713 s | 0.03611 s | 2.136× | YES |

像素程序的串行与并行输出完全一致，说明并行划分没有引入数据竞争。

8 线程只获得约 2.14 倍加速，明显低于 π 计算，原因包括：

- 图像处理有大量内存读写；
- 内存带宽成为瓶颈；
- 单个 `8×8` 块的计算量较小；
- 任务分配和线程同步开销占比更高。

这说明线程数增加并不保证等比例加速，必须先判断程序的实际瓶颈。

## 7. 计时方法

使用：

```cpp
double start = omp_get_wtime();
// 只执行被测计算
 double end = omp_get_wtime();
```

注意：

- 输入构造、内存分配、文件读写不应混入核心计算计时；
- 串行和并行版本必须做相同的工作；
- 应先验证输出正确，再讨论速度；
- 至少测试 1、2、4、8 个线程；
- 最好重复多次，报告均值、中位数或最小值；
- 线程数为 1 的并行版本可能比纯串行版本慢，这是并行管理开销造成的正常现象。

这与此前可信计时笔记中的原则一致：明确计时边界、固定输入条件、区分计算时间和调度/提交时间。

## 8. 与 Day 1 GPU 学习的联系

Day 1 的 GPU Vector Add 和本日 OpenMP Vector/Pixel 处理有共同原则：

> 将一个大问题拆分为许多互不依赖的小任务，并让每个执行单元负责自己的数据区域。

区别在于：

| 对比项 | GPU/HIP | CPU/OpenMP |
|---|---|---|
| 并行执行单元 | Thread / Block / Wavefront | CPU 线程 |
| 任务规模 | 通常需要大量线程 | 通常是少量核心/线程 |
| 典型优势 | 高吞吐、隐藏显存延迟 | 改造成本低、适合共享内存循环 |
| 边界保护 | `if (i < n)` 防止尾块越界 | 循环边界和块边界检查 |
| 典型风险 | 越界、访存不合并、分支发散 | 竞态、错误共享、并行开销 |

两日共同的性能思维是：先分析数据依赖和访存，再选择并行模型，最后用可信实验验证，而不是看到“多核”或“GPU”就盲目加速。

## 9. 本日结论

1. `parallel for` 适合迭代之间互不依赖的循环；
2. 共享累加变量需要 `reduction`，否则会产生竞态条件；
3. 变量放在循环体内通常会自然成为线程私有变量；
4. 循环足够大才值得并行，小循环可能越并行越慢；
5. 图像块之间不重叠时可以安全并行；
6. 正确性检查必须先于性能比较；
7. 加速比受到计算密度、内存带宽、任务粒度、同步和线程数的共同影响。

## 10. 文件与复现实验命令

本次代码：

- [`day2_openmp/pi_openmp.cpp`](day2_openmp/pi_openmp.cpp)
- [`day2_openmp/pixel_openmp.cpp`](day2_openmp/pixel_openmp.cpp)
- [`day2_openmp/make_test_ppm.py`](day2_openmp/make_test_ppm.py)

编译 π 程序：

```bash
g++ -O3 -fopenmp pi_openmp.cpp -o pi_openmp
OMP_NUM_THREADS=8 ./pi_openmp 500000000
```

编译像素程序：

```bash
g++ -O3 -fopenmp pixel_openmp.cpp -o pixel_openmp
python3 make_test_ppm.py 4096 4096 input.ppm
OMP_NUM_THREADS=8 ./pixel_openmp input.ppm output.ppm 8
```

输出文件使用 PPM `P6` 格式，可用 ImageMagick、GIMP 或其他支持 PPM 的工具查看。
