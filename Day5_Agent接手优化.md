# Day 5：让 Agent 接手优化

> 对应教程第 14–17 章（part3-agent 全篇）：14 Agent 入门 → 15 工具封装 → 16 算子优化 Agent 设计 → 17 多轮优化实战。
> 本章节要求先在 AMD 开发者云（Radeon Cloud）注册并获取免费 DeepSeek-V4-Flash-0731 模型算力，再按 17 章 notebook 从步骤 1 跑到步骤 8。
> 教程原文已镜像到 `day5_agent/tutorial/`（docs/、code/、notebooks/ 与 `_cell_map.txt`），虚拟机里可直接对照运行。

## Gate 0：打卡清单

- [ ] 已注册 AMD 开发者云（developer.amd.com.cn），完成模型接口配置（DeepSeek-V4-Flash-0731 的 API Key 已填入 notebook 步骤 2/4 单元格）。
- [ ] 已按顺序运行 chapter17.ipynb 的环境、模型、GPU、baseline 单元格（步骤 1–4），得到本机 `baseline-benchmark.json`。
- [ ] 已运行步骤 5（多轮优化），得到非空 `trajectory.jsonl` 与 `agent-report.md`；即使被限流/未加速也保留了失败轨迹与原因。
- [ ] 已保存截图 1：Agent 运行轨迹——候选生成、`compile_kernel` / `bench_kernel` / `profile_kernel`（或 `accept_candidate`）工具调用及其返回。
- [ ] 已保存截图 2：候选源码与裁决——正确性门禁结果（PASS/FAIL，`wrong_answer` / `compile_error` 等状态）与接受/拒绝理由。
- [ ] 已保存截图 3：同一输入与计时协议下 baseline、Day 4 人工版本、Agent 版本的 median/min 对比与加速比；未加速时保留失败结果并写明原因。
- [ ] 已把 `best.py`（最终候选源码）与 `trajectory.jsonl` 留档在笔记/仓库里。
- [ ] 已完成每日一问的回答（见文末）。

## 本日要回答的问题

**Agent 怎样在不篡改裁判规则的情况下提出、验证和迭代候选？（答案见文末「每日一问」）**

## 第 14–17 章的内容地图

| 章 | 主题 | 核心概念 | 本日要吸收的东西 |
|---|---|---|---|
| 14 | Agent 入门 | Agent = LLM + 工具 + 循环；ReAct + Reflection | 工具返回值才是事实；权威工具 vs 自由工具 |
| 15 | 工具封装 | `compile_kernel` / `bench_kernel` / `profile_kernel` + `accept_candidate` | 结构化输入输出、错误分层、唯一晋升入口 |
| 16 | Agent 设计 | 读题 → 生成 → compile → bench → profile → accept → 反思 | TaskSpec（task.json 契约）、配对裁决、终止条件 |
| 17 | 多轮优化实战 | vector_add 完整闭环 + 留痕 + 失败回退 | 轨迹账本、对比报告、与人工优化对比 |

教程 0.705→0.322 ms（≈2.19×）是 **Radeon 8060S（gfx1151）云端的参考跑次**；你的结论必须以自己工作区的 `baseline-benchmark.json`、`best-benchmark.json`、`trajectory.jsonl` 为准，**不反推、不抄写**历史数字。

## 知识点 1：TaskSpec——任务契约（第 16 章 + task_spec.py）

`task.json` 是把「任务、输入、评测规则」钉死的唯一来源，Agent、工具、人共用同一份。以 `fixtures/vector_add/task.json` 为例（schemaVersion=2，`task_spec.py` 里用 `TaskSpecV2` 严格校验）：

| 字段 | 本练习值 | 作用 |
|---|---|---|
| `shape` | `4096×2048` | 固定输入形状 |
| `tensors` | x / y 输入、output 输出，`float16`，`n_elements=8388608` | 数据类型与 ABI 签名 |
| `reference` | `reference.py` 的 `reference()` | **Oracle**：期望输出的产生者 |
| `correctness` | `seeds:[17,29]`、`atol/rtol:0.001` | 正确性门禁的判定参数 |
| `benchmark` | `warmup:10, samples:25, innerRepeats:5, timeoutSeconds:180` | 计时协议（沿用 Day 2 口径） |
| `optimization` | `minImprovementFraction:0.01`（1%）、`maxRounds/patience` | 晋升阈值与终止条件 |
| `costModel` | `flops=8388608, bytes=50331648` | Roofline 用的算法口径（AI≈0.167） |

要点：**输入分布、形状、容差、warmup/repeat、阈值全部由契约固定**，候选无权改动；`task_spec.py` 会在加载时拒绝任何缺键、越界或多余字段（如 `missing=/extra=` 报错），从源头防止「改题」。

## 知识点 2：baseline 与 Oracle（第 15–17 章）

- **baseline.py**：故意朴素的 Triton 实现——`block_size=256`、不设 `num_warps`。留出「加大 block / 提高 warps」的可搜索头寸。它就是对比的 0 号选手。
- **reference.py**：`return (x.float() + y.float()).to(x.dtype)`。用 float32 中间精度算 fp16 加法，充当**正确性裁判的标准答案**（Oracle）。评测器对每个 seed 生成输入，逐元素与 reference 输出比较，超过 `atol/rtol=1e-3` 即判 `wrong_answer`。
- 本机第一次跑步骤 4 时，`bench_kernel` 会把 baseline 测成你机器上的真实 `median_ms`——**这一步之前所有历史数字都与你无关**。

## 知识点 3：正确性门禁与评测阶梯（第 15 章 + evaluate.py / worker.py）

评测器不是「跑一下看对不对」，而是分层检查，每层独立返回 JSON 状态：

| 层级 | 状态值 | 含义 |
|---|---|---|
| V0 能运行 | `compile_error` / `runtime_error` | 语法/JIT 失败、运行崩溃 |
| V1 对答案 | `wrong_answer` | 与 reference 不一致（附带 `maxAbsError/maxRelError`） |
| V2 污染防护 | （前置拒绝） | AST 白名单（`source_policy.py`）、子进程隔离、输入张量只读校验、输出别名检查 |
| V3 可信计时 | `ok` + `mean_ms/median_ms/p95_ms/std_ms` | warmup + GPU event + 统计 |
| V4 配对裁决 | `accepted: true/false` + `reason` | 与 incumbent **交错配对**计时，中位改进 ≥1% 且多数配对为正才接受 |

关键规则（`accept_candidate`）：**接受才覆盖 `best.py`；拒绝不覆盖（=失败回退）；无论成败都追加一行 `trajectory.jsonl`**。所以「回退」不是 undo 工具，而是评测器契约。

## 知识点 4：LLM 与确定性工具的权限边界（第 14–15 章）

| 能力 | 谁拥有 |
|---|---|
| 提议：候选源码、改动说明（`change`）、下一步策略 | **LLM**（受控提议） |
| 编译、对答案、计时、瓶颈分析、晋升裁决 | **确定性工具**（`compile_kernel`/`bench_kernel`/`profile_kernel`/`accept_candidate`/`measure_peak`），子进程 + 环境白名单执行 |
| 写 `best.py`、写 `trajectory.jsonl` | 只有 `accept_candidate` |
| 性能数字的「真假」 | 只认工具返回 JSON；**禁止** `run_code` 自测加速比写进报告 |

纪律写死在系统提示（SYSTEM_SOP）与代码里：工具失败不算任务失败（结构化错误 → 反思 → 重试）；没进过闭环就输出最终文本会被 nudge 拉回。

## 学习笔记：运行顺序（在 AMD 开发者云工作区或 ROCm 虚拟机）

### 0. 注册开发者云与模型算力（本章节前置）

1. 打开 <https://developer.amd.com.cn/login?source=91kadjjnI>（AMD 开发者云·中国站），微信 / 手机号 / 魔搭账号登录并完善资料（领积分；活动期间注册送 100 小时免费算力）。
2. 进入 Radeon Cloud 创建一个预置 ROCm + PyTorch 模板的工作区（或继续用你的 ROCm 虚拟机）。
3. 按第 16 章notebook 的模型接口说明，在平台的 **Public Free Model APIs** 中选择 `DeepSeek-V4-Flash-0731`，生成 API Key（Token Factory 步骤，以平台实际界面为准）。
4. 在 chapter16 或 chapter17 的「步骤 2：准备依赖、中文字体与模型接口」单元格里粘贴 Key——默认提供商为 Radeon Cloud（`API_BASE=https://developer.amd.com.cn/radeon/api/v1`，`MODEL=openai/DeepSeek-V4-Flash`），也可切换为 DeepSeek 官方（`https://api.deepseek.com` 的 `deepseek/deepseek-v4-flash`）。Key 通过 `KERNEL_AGENT_API_KEY` / `OPENAI_API_KEY` 等环境变量注入，**不要提交进 git**。

### 1. 第 14 章（chapter14.ipynb，只读理解）

步骤 2 装依赖；步骤 3 `inspect` 定位 `kernel_optimize/agent.py` 的 ReAct 主循环；步骤 4 查看工具 schema。这一章不产生打卡证据，但要把「工具 schema 长什么样、主循环怎么回灌 observation」看明白。

### 2. 第 15 章（chapter15.ipynb，认识裁判）

步骤 4 建工作区（fixture 复制进来）；步骤 5 准备 baseline 与候选；步骤 6 `call_tool` 依次调 `compile_kernel` / `bench_kernel` / `accept_candidate`；步骤 7 读 `trajectory.jsonl`。这章跑通一次**人工模拟的 Agent 步骤**，是第 17 章之前唯一一次「手动调工具」。

### 3. 第 17 章（chapter17.ipynb，打卡主战场）

| 步骤 | 单元格（markdown 标题） | 干什么 |
|---|---|---|
| 1 | 「步骤 1：定位仓库根目录」 | 设 `REPO_ROOT` |
| 2 | 「步骤 2：准备依赖、中文字体与模型接口」 | 装包、下载中文字体、填 API Key（四选一提供商） |
| 3 | 「步骤 3：检查模型与 GPU 环境」 | 20 RPM 节流 + 429 退避；`llm.chat("请只回复 READY")` 探活；打印 GPU/ROCm/Triton |
| 4 | 「步骤 4：建立本次实战工作区并测 baseline」 | 从 fixture seed 工作区；`bench_kernel(baseline)` → `baseline-benchmark.json`；**baseline 不过评测就停** |
| 5 | 「步骤 5：运行完整多轮优化」 | `run_agent(max_steps=25)`：LLM 提候选 → 工具裁决 → 循环；`agent-report.md` + `agent-status.json` |
| 6 | 「步骤 6：读取轨迹并复测最终 best」 | `print_trajectory_table` 输出每轮账本；复测 `best.py` → `best-benchmark.json` / `best-profile.json` |
| 7 | 「步骤 7：生成当前运行的可视化」 | 三张图：`rounds_overview.png` / `status_breakdown.png` / `process_timeline.png` |
| 8 | 「步骤 8：生成对比报告」 | `comparison-report.md`：轮次、接受/拒绝数、baseline vs best、失败与回退、证据路径 |

产物都在 `runs/part3-agent/chapter17/<run_id>/`：`trajectory.jsonl`（每轮 `change/status/accepted/latencyMs/improvementFraction`）、`best.py`、`baseline-benchmark.json`、`best-benchmark.json`、`best-profile.json`、`viz/`、`comparison-report.md`。

```bash
cd code/part3-agent
uv sync && source ./activate-rocm.sh   # 或在云工作区直接运行 notebook
# 需要 ~/.config/hello-gpu/kernel-agent.env（API Key，勿提交 git）
bash chapter16/run_and_visualize.sh --skip-pytest   # 非交互复现入口
```

### 4. 第 16 章（chapter16.ipynb，可选回看）

步骤 4 配置模型接口、步骤 5 节流、步骤 8 跑一个 mini Agent（`max_steps` 较小）、步骤 9 看轨迹。与第 17 章逻辑相同，只在入口与步数上有差别。

## 每日一问：Agent 怎样在不篡改裁判规则的情况下提出、验证和迭代候选？

核心答案一句话：**Agent 只拥有「提案权」，裁判权全部落在不可由候选触及的确定性工具与契约文件里；验证与迭代的每一步都强制经过裁判，裁判的输出（而不是模型的叙述）成为下一轮的唯一事实来源。**

具体拆成五个机制（都能在代码里找到落点）：

1. **角色与文件系统分离（想改也改不到）**
   候选目录（工作区，Agent 可写 `best.py/候选项`）与裁判文件（`task.json`、`reference.py`、评测器代码）物理分离；任务契约加载时就做全量校验（`task_spec.py`：缺键/多余键/越界值直接 `ValueError`）。v2 契约还有 `preflightEvidence`：`baselineSha256 / referenceSha256 / contractSha256` 三个哈希——**裁判文件被改动会与哈希不符而被拒绝**，这就是「不篡改规则」的审计锚点。

2. **工具白名单 + 隔离执行（不允许绕路）**
   Agent 只能通过注册过的工具行动：权威五件套 `compile_kernel / bench_kernel / profile_kernel / accept_candidate / measure_peak` + 自由工具（`ask_user / run_code / read_reference ...`）。工具在**子进程 + 环境白名单**里执行（`source_policy.py` 做 AST 白名单、输入张量只读、输出别名检查），候选源码没有机会接触评测逻辑或改计时参数。

3. **验证 = 多层门禁，状态穷举（每个候选都被同一套规矩审）**
   每个候选固定走 编译(compile) → 对答案(正确性门禁, 与 Oracle 逐元素比对) → 计时(bench, warmup+GPU event+median) → 晋升(accept, 与 incumbent 配对裁决)。任何一层失败都以结构化 JSON 返回（`compile_error / wrong_answer / ok / below_threshold`），**慢但错的候选在第二层就被拦下，不会进入性能比较**。

4. **迭代只吃裁判喂回来的观察（自证无效）**
   ReAct 循环里，LLM 每一轮能看到的是上一次工具的返回（observation 回灌 history）；`accept_candidate` 的 `reason`（改进比例、配对方向）成为下一轮改写的依据。模型没有「我觉得更快了」的通道——**加速比只可能来自 bench/accept 的 JSON**，prompt 里明令禁止用 `run_code` 自测加速比写报告。

5. **留痕与回退由契约保证（失败也是证据，且不影响结果账本）**
   每轮无论成败都追加 `trajectory.jsonl`；只有 `accept_candidate` 返回 `accepted=true` 才覆盖 `best.py`，否则 incumbent 保持上一版（失败回退）。于是 Agent 可以放心大胆地试错：坏候选污染不了 best，但它的失败记录会被保留下来，最终报告引用的是轨迹账本而不是模型口头总结。

一句话总结：**Agent 负责「提」，工具负责「判」，契约负责「钉死」——题目、输入、评测规则三者由同一个不可变契约约束，Agent 的一切动作都必须在裁判可见的通道里发生，所以它既能自由迭代，又无法（也不需要）篡改裁判。**

## 打卡截图说明（第 17 章 notebook 单元格映射）

三个截图都来自 **chapter17.ipynb**，按步骤标题定位单元格（markdown 标题唯一，Jupyter 里按顺序执行即可；下面同时给出 0 基下标便于对照 `_cell_map.txt`）：

1. **Agent 运行轨迹截图**（候选生成、工具调用、compile/bench/profile 返回）
   → 单元格「**步骤 5：运行完整多轮优化**」（0 基下标 18）：窗口输出 `[step N] compile_kernel / bench_kernel / profile_kernel / accept_candidate` 的工具调用与每个观察的首行摘录，末尾打印 ═══ Agent Report ═══（含候选生成思路）。
   → 建议同屏补「**步骤 4：建立本次实战工作区并测 baseline**」（下标 16）的 baseline bench JSON（`ok/median_ms/...`），证明「题目与输入没变」。
   → 兜底证据：`runs/.../<run_id>/trajectory.jsonl` 与 `agent-report.md` 的截图/文本。

2. **候选源码与裁决截图**（正确性门禁 + 接受/拒绝理由）
   → 单元格「**步骤 6：读取轨迹并复测最终 best**」（下标 20）：`print_trajectory_table` 打印每轮账本（status / accepted / improvementFraction / 阈值线），以及复测 `best.py` 的 benchmark/profile JSON（门禁通过 = `ok:true`）。
   → 在终端 `cat runs/.../<run_id>/best.py`（或 notebook 里打印 best 源码）截进同一张图，标明这是被接受的最终候选。
   → 拒绝理由示例（可引用轨迹行）：`below_threshold`（如 `num_stages` 改进 <1%）、明显变慢（如 `block_size=8192` 的 -17%）。**被拒绝的候选不覆盖 best.py**，这条要在笔记里写明。

3. **baseline、Day 4 人工版本与 Agent 版本对比截图**
   → 数据来源：baseline 用「步骤 4」输出的 `median_ms`；Agent 用「步骤 6」输出的 `median_ms`；加速比 = baseline ÷ best（<1 表示未加速）。
   → 单元格「**步骤 8：生成对比报告**」（下标 24）打印的 `comparison-report.md` 把两者放进了同一份报告（含失败与回退清单），适合作为对比截图的骨架。
   → **Day 4 人工版本**：把你在 Day 4 手工调出的 Triton 版本（如某组 block_size/num_warps 配置）放进**同一 task 契约**下用同一 `bench_kernel` 口径补测一行（可在笔记里自制三行对比表：实现 / median_ms / 相对 baseline 加速比）。注意 Day 4 若用的是 float32、N=2^24 的口径，与本章 fp16 4096×2048 不是同一输入——**打卡要求「相同输入和计时协议」，所以人工版必须在本章契约下重新计时**，或在表中明确标注口径差异。

   三行表模板：

   | 版本 | 说明 | median_ms | 相对 baseline |
   |---|---|---|---|
   | baseline.py | 朴素 block_size=256 | （步骤 4 实测） | 1.00× |
   | Day 4 人工版 | Block 调优/向量化（本章契约下重测） | （补测） | ? |
   | Agent best.py | 轨迹最终接受版 | （步骤 6 实测） | ? |

4. **Agent 未加速/失败时**：这属于有效结果。保留 `trajectory.jsonl`、`agent-report.md`、`agent-status.json`（若限流未完成会标 `complete_with_warning`），截图照常提交，并在结论里解释原因（常见：本机 baseline 已接近带宽墙、模型步数耗尽、平台 429 限流、`block_size` 已最优等）。**不要为了好看的加速比改动输入或计时协议。**

## 参考资料（已镜像到本地）

- 教程原文：`day5_agent/tutorial/docs/part3-agent/chapter14~17/index.md`、`chapter17/optimization-report.md`
- 代码：`day5_agent/tutorial/code/part3-agent/`（`kernel_optimize/` 主循环与工具、`chapter14/` 评测器、`chapter15/fixtures/vector_add/`）
- Notebook 与单元格地图：`day5_agent/tutorial/notebooks/chapter1[4-7].ipynb`、`_cell_map.txt`
- 云端注册：`day5_agent/tutorial/docs/cloud/amd-radeon-cloud-index.md`