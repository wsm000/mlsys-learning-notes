# Day 3：解释性能瓶颈

> 对应教程第 6 章（用 rocprof 找到慢在哪里）与第 7 章（读懂 Roofline 图）。注意：**Kernel Trace 截图出自 chapter6.ipynb，不是你链接的 chapter7.ipynb**；chapter7 负责 Roofline 工作点和性能记录。

## Gate 0：打卡清单

- [ ] 已按顺序运行 chapter6.ipynb 的环境、编译、3 个 benchmark 单元格。
- [ ] 已运行 rocprofv3 检查与 kernel trace 采集单元格，得到本机 trace 输出（或已排查 rocprofv3 不可用的原因）。
- [ ] 已保存截图 1：Kernel 名称 + Grid/线程划分 + Kernel 时间。
- [ ] 已保存截图 2：同一协议下连续访存 vs 跨步访存的时间与有效带宽（可加 Roofline 工作点）。
- [ ] 已写出一条"证据—当前判断—下一步单变量实验"，不是一句"需要继续优化"。
- [ ] 已完成每日一问的回答。

## 本日要回答的问题

**Vector Add 慢在哪里，下一步应该改什么？（见文末参考回答）**

## 知识点 1：rocprof Kernel Trace 在证据链里的位置

Day 2 的 GPU Event 回答的是"这段边界内总共花了多久"；`rocprofv3 --kernel-trace` 回答的是**"这些时间里具体跑了哪几次 dispatch、每次多久、用什么配置启动的"**。两者互相印证而不是互相替代：

| 工具 | 回答的问题 | 本练习中的读法 |
| --- | --- | --- |
| GPU Event（Day 2） | 这段 GPU 工作的总时长 | warmup 后 repeat 取 mean/median/min |
| `rocprofv3 --kernel-trace` | 每个 kernel dispatch 各自的名字、起止时间戳、Grid 配置、寄存器分配 | 程序共启动 15 次 dispatch：前 5 行是 warmup，统计时跳过，只看后 10 行 |

trace CSV 里先找这几列（列名可能略有变体，notebook 的解析函数会自动适配）：

| 列 | 用它回答什么 |
| --- | --- |
| `Kernel_Name` | 到底运行了哪个 kernel |
| `Start_Timestamp` / `End_Timestamp` | 单次 dispatch 耗时，单位纳秒：(End − Start) / 1000 = 微秒 |
| `Grid_Size` | 一共启动了多少个 work-item（线程划分的直接证据） |
| `VGPR_Count` / `SGPR_Count` | 静态寄存器分配；本实验里两种配置相同，说明资源不是变量 |

交叉验证的意义：trace 统计出的单次最短时间（如 coalesced 约 329 μs）应与 benchmark 的 min（约 0.334 ms）基本一致。对上了，才说明两种测量测的是同一段工作。

## 知识点 2：Grid Size 与线程划分

coalesced 版每个线程算 1 个元素，`i = blockIdx.x * blockDim.x + threadIdx.x`；linecross 版每个 lane 处理 `stride` 个元素。以 N=16,777,216、block=256 为例：

| 配置 | 每线程输出数 | 总线程（Grid_Size） | 相对变化 |
| --- | --- | --- | --- |
| coalesced | 1 | 16,777,216 | 基准 |
| linecross stride=1 | 1 | 16,777,216 | 与基准相同 → 时间应接近 |
| linecross stride=32 | 32 | 524,288（1/32） | 循环次数 ×32，wavefront 数 ÷32 |

trace 中 Grid_Size 这一列就是把"源码层面的线程划分"变成"实测证据"的关键：它证明 stride=32 时工作划分确实变了，而不只是地址变了。

## 知识点 3：连续访存 vs 跨步访存

一个 wavefront 里的多个 lane 同时发起访问。若相邻 lane 的地址相邻（lane0→a[0]，lane1→a[1]…），GPU 能把多次请求**合并成较少的显存事务**（Memory Coalescing）。linecross 让相邻 lane 的起点隔开 `stride` 个 float：stride=32 即隔 128 Byte，一次事务搬回的 cache line 里大部分字节用不上，等效带宽随之下降。

但本章刻意保留了一个坑：**改 stride 这个命令行参数同时改变了三件事**——地址排布、每线程循环次数、Grid Size。所以 6.7 倍的差距是组合效果，不能全部归因于合并访存。公平对照应当固定线程数与每线程循环次数，只改循环内的索引公式：

~~~text
连续版：i = tile_base + j * 32 + lane   （同一轮 j，各 lane 地址相邻）
分散版：i = tile_base + lane * 32 + j   （同一轮 j，各 lane 地址隔 32 个 float）
~~~

## 知识点 4：有效带宽与算法口径

benchmark 按"算法有效字节"折算带宽：每个元素读 a(4B)+读 b(4B)+写 c(4B)=12 B，

~~~text
有效带宽 = 12 × N / kernel 时间
~~~

它是**为了方便比较而换算的数值，不等于硬件实际发出的 DRAM 事务量**。所以会出现 coalesced 有效带宽 603 GB/s 高于参考显存稳态带宽 510 GB/s 的现象——工作集可能部分命中 cache，且口径不同。这不矛盾，也不该被写成"突破了硬件上限"。

## 知识点 5：Roofline 三步读图与 memory-bound 判断

1. **横轴**：算术强度 AI = 计算量 / 数据量。vector add 每 12 B 做 1 次加法，AI = 1/12 ≈ **0.083 FLOP/Byte**。
2. **拐点**：ridge = compute ceiling / memory ceiling = 10.6 TFLOPS ÷ 510 GB/s ≈ **20.78 FLOP/Byte**（9070XT 参考值）。AI 远小于拐点 → 工作点在斜线一侧，理论上是 **memory-bound**。
3. **纵轴与距离**：实际性能 P = AI × 有效带宽。coalesced 工作点 P ≈ 0.050 TFLOPS，利用率约 118%（相对 510 GB/s 参考线，超出源于口径与 cache）；linecross stride=32 的 P ≈ 0.0075 TFLOPS，利用率仅约 17.6%，离线很远。

排查方向由位置决定：左侧远点先查访存（地址是否连续、有无多余读写）；右侧远点查计算路径；离两条线都远则查 launch、同步和输入规模。Roofline 只选方向，不定位到代码行。

## 证据分级：实测 / 推测 / 未验证

| 级别 | 本日的例子 |
| --- | --- |
| 实测证据 | benchmark 时间（0.334 ms vs 2.25 ms）；trace 的单次耗时、Grid_Size 16M→524K、VGPR/SGPR 不变；stride 扫描的单调恶化趋势 |
| 工程推测（有依据但未隔离） | "差距主要来自合并访存被破坏"——合理，因为 stride 实验同时改了三个变量 |
| 尚未验证的判断 | "固定线程数后仅改索引公式就能收回大部分差距"；任何把 6.7 倍全部归给访存的说法 |

## 完整重跑代码：用本机实测值替代参考数据

我另外提供了可直接复制到虚拟机仓库中的脚本：`day3_rerun/rerun_performance_record.py`。它会重新编译 `vector_add.hip`，新建本次运行目录，重新执行 coalesced、linecross stride=1、linecross stride=32 三个 benchmark，读取新 JSON，计算 AI/有效带宽/TFLOPS，并在 `rocprofv3` 可用时自动采集 Kernel Trace。

在虚拟机中把脚本放在 `hello-gpu/day3_rerun/`，从仓库根目录运行：

~~~bash
cd /path/to/hello-gpu
python3 day3_rerun/rerun_performance_record.py
~~~

它会在 `code/part1-profiling/chapter6/logs/day3_rerun_<时间>/` 下生成：

- `performance_record.md`：可直接检查和截图的本机性能记录；
- `chapter7_results.json`：本机 coalesced 与 linecross stride=32 结果，格式与 chapter7 第 13 格的 `chapter6_results` 相同，可直接 `json.load` 后使用；linecross stride=1 仍保留在性能记录中；
- `coalesced.json`、`linecross_stride1.json`、`linecross_stride32.json`；
- `rocprof/`：若 profiler 成功，里面有本次新生成的 Kernel Trace CSV。

如果只想先重跑 benchmark、不采集 trace：

~~~bash
python3 day3_rerun/rerun_performance_record.py --no-profile
~~~

如果自动识别架构失败，可以显式指定编译 target，例如：

~~~bash
python3 day3_rerun/rerun_performance_record.py --arch gfx1201
~~~

`--arch` 只控制编译 target，不是硬件型号的独立证明；优先使用 notebook 的 `rocminfo` 检测结果。只有已经在本机测得 memory/compute ceiling 时，才额外传 `--memory-ceiling` 和 `--compute-ceiling`，不要把其他 GPU 的 510 GB/s、10.6 TFLOPS 直接当成本机上限。

脚本也可以从 Notebook 中用 `%run` 调用，但路径应按你的 Notebook 当前工作目录调整。运行完成后，可在 chapter7 第 13 格执行 `chapter6_results = json.loads(Path("/实际路径/chapter7_results.json").read_text())`，再运行第 23 格使用本机工作点。注意：仓库的第 17 格绘图脚本本身使用固定的 RX 9070 XT 参考线和参考点，不会自动读取这个 JSON；要画本机 Roofline，需要把本机数据接入绘图代码，或直接使用 benchmark 的时间/有效带宽表作为截图证据。

## 在虚拟机上要跑的 ipynb 单元格

按"从上往下数第 N 个单元格（含 Markdown）"定位；若你的仓库版本不同，以单元格首行文字为准。

### chapter6.ipynb（notebooks/part1-profiling/chapter6.ipynb）

| 顺序 | 单元格 | 首行标识 | 作用 |
| --- | --- | --- | --- |
| 1 | 第 8 格（code） | `import csv ...` | 定位仓库根目录 |
| 2 | 第 10 格（code） | `detect_gpu_arch` | 检测 gfx 架构 |
| 3 | 第 12 格（code） | 编译命令 | hipcc 编译 vector_add_bench |
| 4 | 第 14 格（code） | `run_benchmark(...)` | coalesced 与 linecross stride=1 各 100 次 → **截图 2 上半部分** |
| 5 | 第 16 格（code） | `linecross32_ok = run_benchmark` | linecross stride=32 → **截图 2 下半部分** |
| 6 | 第 20 格（code） | `rocprofv3 --version` | 确认 profiler 可用 |
| 7 | **第 22 格（code）** | `profile_results = {}` | 采集并解析 kernel trace → **截图 1 来源**：输出里有 Kernel_Name、dispatches、min/median（μs）、Grid、VGPR、SGPR |
| 8 | 第 30 格（code，可选） | `stride_scan_status` | stride∈{1..256} 扫描趋势，佐证假设 |
| 9 | 第 32 格（code） | `CH6_PROFILE_COMPLETE` | 自检是否满足 Pass Criteria |

补充：想展示更原始的证据，可打开 `code/part1-profiling/chapter6/logs/rocprof/<gfx>_<token>/<label>/` 下的 `*_kernel_trace.csv`，直接截 Kernel_Name / Grid_Size / Start_End_Timestamp 几列。若第 20 格显示 rocprofv3 不可用：先在终端跑 `which rocprofv3`，缺包就按发行版安装 rocprofiler3 再重跑；trace 失败时 notebook 只会给内置参考数据，**那不能满足本次打卡要求**。

### chapter7.ipynb（你链接的这个）

| 顺序 | 单元格 | 首行标识 | 作用 |
| --- | --- | --- | --- |
| 1 | 第 9 格（code） | `_find_repo_root` | 定位仓库 |
| 2 | 第 11 格（code） | `bytes_per_elem = 4+4+4` | 算 AI=0.083 |
| 3 | 第 13 格（code） | `chapter6_results = {` | 载入工作点；**建议把 time_ms/effective_bw 改成你虚拟机第 14/16 格实测值**，并在 print 里注明"本机实测" |
| 4 | 第 15 格（code） | `hardware_ceilings = {` | 标注两条参考线来源（9070XT 参考） |
| 5 | 第 17 格（code） | `plot_script = WORK_DIR / ...` | 运行绘图脚本生成 Roofline 图；失败则第 19 格画备用图 → **截图 2 的 Roofline 部分** |
| 6 | 第 23 格（code） | `BW_GDDR6 = 510` | 输出拐点 20.78、左右侧判断、利用率 118%/17.6% → **memory-bound 判断的直接输出** |
| 7 | **第 26 格（code）** | `performance_record = f"""` | 生成一页性能记录 → **截图 3 底稿**：把"当前判断/下一步"改成你自己的一条假设 |

## 每日一问参考回答

**Vector Add 慢在哪里，下一步应该改什么？**

慢分两层说。第一层是本质：vector add 的 AI 只有 0.083 FLOP/Byte，远在 Roofline 拐点（约 20.8）左侧，天生 memory-bound；即使写得最好的 coalesced 版，有效带宽也已贴近参考显存上限，再快只能靠少搬数据（融合进上下游算子）。第二层是对照实验里的慢点：kernel trace 实测 `kernel_linecross`(stride=32) 单次约 2202–2304 μs，是 `kernel_coalesced`（约 330 μs）的近 7 倍，其 Grid_Size 只有 1/32（524,288 vs 16,777,216）。

下一步不是泛泛"继续优化"，而是一个单变量实验：**固定线程数与每线程循环次数（每 lane 都处理 32 个元素），只改索引公式——连续版 `i = tile_base + j*32 + lane`，分散版 `i = tile_base + lane*32 + j`——重新计时**。若两者差距消失，说明此前的差距主要不是访存方式造成；若仍差数倍，才能把主因归给访存合并。预期观察量：两版的 min/median 时间与有效带宽。

## 打卡材料

1. `day3_rocprof_kernel_trace.png`：chapter6 第 22 格输出（或 trace CSV），须可见 Kernel_Name、Grid_Size、Kernel 时间。
2. `day3_coalesced_vs_strided.png`：chapter6 第 14/16 格同协议输出（N=16777216、block=256、warmup=20、repeat=100 的 min/median + 有效带宽），可拼 chapter7 第 17/23 格的 Roofline 图与利用率输出。
3. `day3_hypothesis.txt/png`：chapter7 第 26 格性能记录改写版，含一行"证据—当前判断—下一步单变量实验"。

将未裁剪的原始文本一并存入 [打卡材料](打卡材料/README.md)。所有数字必须是虚拟机上的真实运行结果。
