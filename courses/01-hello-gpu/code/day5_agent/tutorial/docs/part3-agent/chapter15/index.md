---
title: "第15章 工具封装"
description: "Hello GPU 第15章 · compile/bench/profile 三件套 + accept_candidate"
---

# 第15章 工具封装

## 本章目标、前置知识与产物

本章把 Part 1 建立的计时习惯、正确性检查和 profiling 直觉，收成 **Agent 可调用、返回可解析** 的标准化工具。这是让 Agent「能动手」的前提——没有工具的 Agent 只会空谈。

学完本章，你应该能够：

- 解释为什么 Agent 优化 kernel 必须依赖结构化工具接口；
- 说清三件套各自返回哪些 JSON 字段；
- 理解「工具失败不算任务失败」：结构化错误 → 反思 → 重试；
- 知道只有 `accept_candidate` 能更新 `best.py` 并写轨迹。

对应代码：

```text
code/part3-agent/chapter14/          # 评测器后端
code/part3-agent/kernel_optimize/
├── tools.py                         # Agent 可见工具
└── measure.py                       # profile_kernel / measure_peak
```

## 15.1 为什么要封装工具

Agent 不能可靠地拼 shell 字符串。封装后的工具满足：

1. **结构化输入**：kernel 源码、task.json；
2. **结构化输出**：JSON（线上契约字段）或明确错误列表；
3. **隔离执行**：子进程 + 环境白名单，避免候选污染宿主机；
4. **失败可回流**：编译错误、对答案失败、超时都以明确字段返回。

工具封装第一原则：**反馈必须是结构化的、可解析的、分层的。** 编译成功/失败、正确性通过/失败、性能数字，这三层信息必须互相独立——Agent 才能知道当前卡在哪一层。

第二原则：工具有明确所有权——谁负责计时、谁负责正确性、谁负责 profiling，都在工具内部完成；Agent 只负责调用和解读。

第三原则：可复现——同一工具、同一输入，应得到同一口径的输出（可信计时的延伸）。

## 15.2 封装 benchmark 工具：`bench_kernel`

把 Part 1 的计时方法包成工具：

```text
工具名：bench_kernel
输入：
  - source：kernel 源码
  - repeats：可选（默认用 task.json 合同）
输出（JSON，成功时）：
  {
    "ok": true,
    "mean_ms": 0.325,
    "median_ms": 0.322,
    "p95_ms": 0.340,
    "std_ms": 0.008,
    "bandwidth_gbps": 156.2,
    "tflops": 0.026
  }
```

计时约定（与 Part 1 一致，写进任务合同）：

- warmup 若干次，甩掉冷启动；
- 用 **GPU event** 量 device 时间；
- 报告 mean / median / p95 / std；
- 用 `costModel` 推算 `bandwidth_gbps` / `tflops`。

**真实样例**（Radeon 8060S / `gfx1151`，`vector_add` 优化后 incumbent，工作区 `task-20260806-144647`）：中位延迟约 **0.322 ms**。注意：`bench_kernel` 只报告「这份源码多快」，**不负责晋升**——和 incumbent 的配对比较在 `accept_candidate`。

## 15.3 封装 profiling 工具：`profile_kernel`

benchmark 告诉你「慢了」，profiling 告诉你「慢在哪」。Agent 需要的是瓶颈信号，不是几千行原始 trace。

```text
工具名：profile_kernel
输入：source
输出（JSON，Roofline 路径示例）：
  {
    "ok": true,
    "bottleneck": "DRAM bandwidth",
    "bound": "memory-bound",
    "ai": 0.167,
    "bandwidth_utilization_pct": null,
    "sm_saturation_pct": null,
    "suggestion": "优先减少启动开销 / 提高访存并行度（更大 block_size、num_warps）"
  }
```

**真实样例**（同上机、同上任务）：

| 字段 | 值 | 含义 |
|---|---|---|
| `measure_peak` 带宽 | **210.5 GB/s** | Roofline 横轴天花板 |
| fp16 峰值 | **27.8 TFLOPS** | 对本算子几乎用不上 |
| AI | **0.167** FLOP/Byte | `flops/bytes ≈ 8.4M / 50.3MB` |
| `bound` | **memory-bound** | 应优先减启动 / 提高访存效率 |

`measure_peak` 实测本机 copy 带宽与 matmul 推得的 TFLOPS，缓存到 `~/.cache/hello-gpu`，**禁止写死某型号常数**。rocprof 计数器路径可增强利用率字段；教学闭环以 Roofline JSON 即可起步。

## 15.4 封装编译工具：`compile_kernel`

编译是 Agent 高频动作里最容易失败的一环。输出必须让 Agent 精确知道失败原因：

```json
{
  "ok": false,
  "stage": "compile",
  "errors": [{"line": 42, "msg": "expected a type"}],
  "warnings": []
}
```

| `stage` | 含义 |
|---|---|
| `compile` | 语法 / JIT 失败 |
| `run` | 能跑但对答案失败（`wrong_answer`） |
| （`ok=true`） | 编译与正确性通过 |

Triton 候选的「编译」发生在 worker 首次加载 JIT 时。错误按行列出，而不是整段日志——Agent 是文本模型，噪音日志会淹没可行动信息。

## 15.5 工具的输入输出 schema 与晋升入口

每个工具的接口用 JSON schema 注册（见 `kernel_optimize/tools.py`）。Agent 的 prompt 里只放 schema，不放实现细节。

三件套之外，本地增加 **唯一晋升入口**：

```text
工具名：accept_candidate
输入：source + change（一句话说明本轮改动）
输出：
  {
    "accepted": true,
    "reason": "improved（中位改进 0.0150，100% 配对为正，…）",
    "latencyMs": 0.322,
    "improvementFraction": 0.0150
  }
```

晋升规则：与 incumbent **交错配对计时**，中位改进 ≥ `optimization.minImprovementFraction`（教程默认 **1%**），且多数配对为正，才接受并覆盖 `best.py`；无论成败都追加 `trajectory.jsonl`。

任务开始前还有一份 **task contract**（`task.json`）：shape、dtype、tensors、正确性容差、benchmark 参数、改进阈值。同一份合同被 Agent / 工具 / 人共享。

评测器能力阶梯（概念上 V0→V4，实现已合并进 `evaluate()`）：

| 阶梯 | 能力 | 缺了会怎样 |
|---|---|---|
| V0 能运行 | 候选能在子进程里 launch | 连「能不能跑」都没有 |
| V1 对答案 | 与 `reference.py` 比对 | 「快但错」会被当成优化 |
| V2 污染防护 | AST 白名单、隔离 env | 候选偷改计时骗过评测 |
| V3 可信计时 | warmup + GPU event + median/MAD | 结论不可复现 |
| V4 配对裁决 | 与 incumbent 交错 + 阈值 | 被噪声当成加速 |

## 15.6 错误处理与重试

| 错误类型 | 例子 | 回传给 Agent | Agent 下一步 |
|---|---|---|---|
| 环境错误 | GPU 忙、依赖缺失 | 明确类型 + 建议 | 等待或换方案 |
| 编译错误 | 语法 / 类型 | `errors[{line,msg}]` | 定位到行修改 |
| 验证错误 | 正确性失败、未过阈值 | `stage` / `reason` / 改进比例 | 改算法或换机制 |

关键原则：**工具失败不算任务失败。** 结构化错误回到对话历史，触发反思；`accept_candidate` 拒绝时 **不写 `best.py`**（回退 = 保持 incumbent），但失败轮次照样留痕。

本地复现评测器：

```bash
cd code/part3-agent
uv sync && source ./activate-rocm.sh
bash chapter14/run_cases.sh
```

## 本章小结

- Agent 优化 kernel 的前提是结构化工具：编译、benchmark、profiling 各有明确输入输出。
- 对外三件套：`compile_kernel` / `bench_kernel` / `profile_kernel`；晋升靠 `accept_candidate`。
- 错误处理原则：结构化错误 → 反思 → 重试；失败留痕，best 不被坏候选覆盖。

## 延伸阅读

- `code/part3-agent/chapter14/EXPERIMENT.md`
- `code/part3-agent/kernel_optimize/tools.py`
- Part 1 计时与 Roofline 章节

