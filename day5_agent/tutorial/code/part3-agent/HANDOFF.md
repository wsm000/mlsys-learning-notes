# 算子优化思路 · 交接文档

> 给接手「agent 部分」的同学。这份文档讲清楚我们做算子优化的**思路（方法论）**，
> 以及这套思路如何落成一个 agentic 的教学工件。读完你应该能接着把 agent 做下去。
> 背景项目：hello-gpu 第三篇（part3-agent）——用一个 agent 把「算子优化闭环」讲透并跑通。

---

## 0. 一句话定位

> **不是"让大模型随便生成一段更快的代码"，而是把"人类专家做算子优化的完整流程"自动化，
> 并且每一步的性能结论都可信、可追溯。**

这是教学工件（hello-gpu 教程第三篇），所以还要**简单、可读、能讲清楚**。

---

## 1. 优化闭环（核心思路，必须讲透的"一个闭环"）

```
理解任务 → 测硬件峰值 → 跑 baseline → profiling 定位置 → 定优化方向
   → 生成候选 → 校验正确性 → 可信计时 → 配对裁决 → 反思迭代 → 报告
```

逐步说清楚每一步**为什么存在、缺了会怎样**：

1. **理解任务**：算子语义、目标 shape、dtype、入口签名。
   → 缺了它，后面全是在优化一个没定义清楚的东西。

2. **测硬件峰值**：实测本机带宽（GB/s）和算力（fp32/fp16 TFLOPS）。
   → 这是 Roofline 的"天花板"。没有它，你不知道优化到多少算到头。
   → **必须实测，不能写死某个型号的常数**（不同 GPU 差很多）。

3. **跑 baseline**：先有一个正确的、可信计时的基线。

4. **profiling 定位置**：把 baseline 放到 Roofline 上——算算术强度（FLOP/Byte），
   判断它是 **memory-bound** 还是 **compute-bound**，算利用率、距峰值的空间。
   → 这是"先看数据再改代码"（Profiling-Driven），对症下药而不是瞎改。

5. **定优化方向**：按 bound 类型选策略（见 §3 套路库）。
   → memory-bound 就该减搬运，compute-bound 就该减运算；方向错了白干。

6. **生成候选**：LLM 提出一个优化版本的 kernel。

7. **校验正确性**（正确性先于性能）：候选先和 PyTorch reference（oracle）对答案，
   **错了就直接判负、拿不到性能数字**。
   → 防止"快但是错的"kernel 被当成优化成果。

8. **可信计时**：warmup + GPU event + median/MAD（不用最快值），
   候选与 incumbent **配对、交错**计时消除时序漂移。
   → benchmark 不可信，一切结论作废。这是整篇最硬核的部分。

9. **配对裁决**：改进幅度要**超过由实测噪声校准出的阈值**才算真改进。
   → 阈值不能拍脑袋，要从 baseline 自配对噪声实测出来（如噪声 0.39% → 阈值取 ~1%）。

10. **反思迭代**：被拒的候选读报错/瓶颈再改；连续失败就换机制。**失败也记录**。

11. **报告**：baseline→best 的加速比、用了哪些优化、轨迹、Roofline 位置。
    → **搜索与结论分离**：搜索期的单个快样本不是结论，最终 best 要独立复测。

---

## 2. 五条方法论原则（思路的灵魂）

1. **Profiling-Driven / Roofline 驱动**：先看数据（在 Roofline 上的位置），再改代码。
   判断 memory/compute-bound，对症下药。这是全书第一主线。

2. **正确性先于性能**：错误候选拿不到性能数字。oracle（PyTorch reference）是裁判。

3. **可信计时与统计**：warmup + GPU event + median/MAD + 配对比较 + 校准阈值。
   "benchmark 不可信、profiling 没保存、硬件上下文没写清楚——结论再漂亮也不能用。"

4. **失败留痕**：失败的尝试老老实实记下来——失败的记录往往比成功那一次更值钱。

5. **搜索与结论分离**：搜索期单快样本 ≠ 结论；最终 best 用新进程独立复测。

---

## 3. 优化套路库（按 bound 类型，喂给 agent 的"招式"）

**memory-bound**（APU/共享内存上多数算子都是）——减少数据搬运：
- 合并访存（coalescing）：相邻 lane 访问相邻地址。
- 向量化加载/写回（一次搬更宽，如 float4）。
- 减少中间结果写回（能在寄存器完成就别落内存）。
- kernel 融合（多步合一，省中间张量读写，如 softmax/LayerNorm 融合）。
- 避免读放大（同一数据被重复读 → 共享内存缓存复用）。

**compute-bound**——减少运算量 / 提高吞吐：
- 消除冗余计算（循环不变量外提、合并重复表达式）。
- 更高吞吐的数据类型/指令（fp16/bf16、dot 指令）。
- 指令级并行（展开独立计算喂满 SIMD）。

**典型谱系**：
- reduction：atomicAdd → LDS 树形归约 → wavefront shuffle → unroll+向量化 → two-pass。
- matmul：naive → LDS tiling → register blocking（对比 rocBLAS，库是尺子不是 KPI）。
- softmax：naive 三趟（max/sum/normalize）→ online/fused 单趟。

**踩坑提醒**：
- occupancy：寄存器/LDS 用太多会降低活跃 wave，参数不是堆越多越好。
- N 非 2 的幂时注意 LDS bank conflict（可能需 padding）。
- cache 带宽 ≠ 主存带宽：小数据反复命中 cache 会高估带宽。

---

## 4. Agent 化的设计原则（做 agent 的人重点看）

把上面的闭环交给 agent 时，守住这几条：

1. **★ 工具结果就是事实，LLM 不自报数字。**
   性能/正确性只认权威工具（compile_kernel / bench_kernel / profile_kernel /
   accept_candidate / measure_peak）的返回。**LLM 绝不能自己声称"快了 X%"**。
   （血的教训：曾出现 agent 用 run_code 自己 benchmark 出 1.42× 写进报告，
   但权威裁决其实拒绝了——报告撒谎。必须杜绝。）

2. **确定性代码掌权，LLM 受控提议。**
   LLM 负责：理解任务、选优化策略、写/改 kernel、多轮对话、读 profiling 判断方向。
   确定性代码（工具）负责：校验正确性、计时、裁决、测峰值——这些是 ground truth。
   一句话："LLM 负责理解，代码负责相信什么。"

3. **agentic 但克制。**
   流程让 LLM 多轮自主驱动（不要写死成固定 pipeline）；但"真假"由工具锁死。
   优化闭环本质是 **Reflection 范式**：生成 → 评测 → 反思 → 改进，收敛即停。

4. **少而硬的工具，每个对应闭环一步**（str→str，字典注册）：
   `ask_user`（多轮问）/ `get_environment`（探测设备）/ `measure_peak`（测峰值）/
   `compile_kernel`（编译+正确性）/ `bench_kernel`（计时）/ `profile_kernel`（定方向）/
   `accept_candidate`（配对裁决+写轨迹）/ `convert_kernel`（转语言）/ `setup_task`（建任务）/
   `read_reference`（读优化套路）。

5. **优化闭环可封装成一个 Skill**（hello-agents 范式）：
   `SKILL.md`（触发条件 + 优化 SOP + 套路模板）+ `scripts/`（确定性脚本锁死脆弱操作）
   + `references/`（硬件规格、优化套路库）。渐进式披露，按需加载。

6. **语言/硬件无关**：evaluator 执行 **Triton**；非 Triton（CUDA/HIP C++）先转成 Triton
   再优化（CUDA 和 HIP 语法接近，但 evaluator 都跑不了，统一转 Triton）。
   硬件峰值实测（不写死型号）。

7. **多轮对话，别卡死**：缺参数就问用户（ask_user），语言不符就征询转换，
   不要瞎猜或直接退出。

---

## 5. 当前实现状态（截至交接）

**已搭好**（`code/part3-agent/kernel_optimize/`）：
- `agent.py`：Reflection/ReAct 主循环 + 完成护栏（防止 agent 用纯文本"假结束"）。
- `tools.py`：上述全部工具（裹住 chapter14 evaluator）。
- `intake.py`：`setup_task`——LLM 从原始 kernel 生成 reference + task 合同 + costModel，evaluator 自检。
- `measure.py`：硬件峰值测量（通用、按设备缓存）+ Roofline 方向判定。
- `prompts.py`：优化 SOP + 入门引导（语言中立）。
- `skills/rocm-kernel-optimize/`：SKILL.md + scripts + references。
- 入口：`uv run python -m kernel_agent`（零参数、纯对话、多行直接粘贴）。

**已跑通 / 已验证**（真机 9070 XT，gfx1201 / ROCm 7.13）：
- 完整闭环：CUDA vector_add / Triton softmax 从理解→转换→建任务→测峰值→优化→报告。
- `measure_peak` 实测（538 GB/s、fp32 8 TFLOPS 等）。
- `setup_task` 首轮建对任务（修好了 dtype / reference 要 `import torch` / 输出形状等坑）。
- 裁决改成"中位改进 ≥ 阈值 + 多数配对为正"，真实加速不再被噪声误拒。

**关键修复记录**（避免接手人重复踩）：
- 多行输入：用 prompt_toolkit + bracketed paste，多行直接粘贴、回车提交（不要 /paste、/end）。
- reference.py 允许且仅允许一个 `import torch`；输出形状是 (rows, cols) 不要 flatten。
- 标量 dtype 只能 int32/int64（不能 float16）。
- 裁决用中位数而非最差配对（快 kernel 噪声大，最差配对会误拒真实加速）。

---

## 6. 待改进（交接给做 agent 的人，按优先级）

1. **★ run_code 的权威性**：agent 能用 run_code 自己 benchmark 并把数字当结论，
   绕过权威工具。约束方向：SOP 明确"性能数字只认 accept_candidate/bench_kernel，
   run_code 自测不算数"；或限制 run_code 用途；或让报告只引用权威工具数据。

2. **★ 报告 grounding**：最终报告是 LLM 自由生成，可能与 best.py / trajectory.jsonl
   不符甚至编造。方向：加 `get_run_summary` 工具读 trajectory.jsonl 让报告基于真实记录；
   或在 `__main__` 末尾打印一份**确定性汇总**（baseline/best 延迟、加速比、接受轮次），
   无论 LLM 报告说什么都给用户看真账。

3. **闭环效率**：agent 容易在 run_code 上空转十几步。方向：提示词引导聚焦
   accept_candidate、给步数预算、减少 run_code 诱惑。

4. **profiling 做实**：当前 profile_kernel 优先用 Roofline 模型（costModel），rocprof 实测
   计数器（FETCH_SIZE 等）路径已留但未打磨。真机上把 rocprof 路径跑通更扎实。

5. **文档同步**：part3 正文（`docs/part3-agent/chapter13–16`）已按线上第14–17章结构重写，并嵌入 `task-20260806-144647` 真实结果；工具叙事对齐 `compile_kernel` / `bench_kernel` / `profile_kernel` / `accept_candidate`。

6. **优化实效**：让 agent 真能跑出漂亮加速（转换质量 + 按 bound 类型用对套路），
   合龙教程承诺的"优化 3–5 倍"。

---

## 7. 环境与运行

- **机器分工**：本地 Mac 做 git；实验机只跑实验、绝不 git；文件用 scp/rsync 流转。
- **实验机**：9070 XT（gfx1201 / ROCm 7.13）；主线目标 AI MAX 395（gfx1151 / RDNA3.5 /
  共享 LPDDR5X，带宽远低于独显，多数算子深度 memory-bound）。
- **运行**：`uv run python -m kernel_agent`（零参数对话式）。
- **LLM 配置**：`~/.config/hello-gpu/kernel-agent.env`（`KERNEL_AGENT_MODEL=provider/model`
  + `KERNEL_AGENT_API_KEY`）。
- **evaluator**：`chapter14/`（V0→V4：能运行→对答案→污染防护→可信计时→配对裁决），
  执行 Triton，有 AST 白名单（candidate 与 reference 各有约束）。

---

## 8. 参考（思路来源）

- **GenericAgent**（`reference_repositories/GenericAgent`）：不预装技能、用工具进化；
  反思双重编码（提示词升级阶梯 + 引擎护栏）；No Execution No Memory。
- **hello-agents**（`reference_repositories/hello-agents`）：教学范式（裸 loop→模块化→框架）；
  Reflection 范式；工具 = str→str + 字典注册；Skill 封装（自由度光谱：策略给 LLM，脆弱操作脚本锁死）。
- **hello-gpu-youyoulyz**（part3-hip-kernels）：手工优化闭环样板（阶梯版本 + 五步管线 +
  双层校验 + 字节级 profiling 实证 + 诚实记录失败）。
- **part3 文档**（`docs/part3-agent/chapter13–16`）：教学叙事（ch14 的 V0→V4 evaluator 是不可砍的核心）。
