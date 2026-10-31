# Task7 项目验收与修复 · 打卡报告（hello-gpu Day5 kernel_optimize 项目）

> 运行环境：vm-60（simin@10.5.100.60）· Ubuntu 22.04 · RTX 4090 D 24GB · Python 3.10.12 · torch 2.9.1+cu128 · triton 3.5.1
> 运行时间：2026-10-30 01:15–01:17 UTC ｜ 验收对象：`day5_agent/tutorial/code/part3-agent`（chapter14 权威裁判 + chapter15 vector_add fixtures + kernel_optimize agent）
> 验收 harness：`courses/09-deepagents-in-action/task7/`（套件）＋ `courses/09-deepagents-in-action/evidence/task7/`（证据），裁判代码零改动

---

## 一、验证什么

先回答开头的问题：**Agent 跑通、有输出，不等于任务完成。** 完成的标准是「固定输入 + 明确预期 + 证据」三者齐全，且证据能复现。

本阶段把这三件套用到我自己的项目——Day5 的 kernel_optimize Agent 项目上。这个项目的核心资产不是某个候选 kernel，而是**权威裁判**（`chapter14/evaluate.py` + `worker.py`）：Agent 的 compile/bench/accept 全部工具都建立在它之上，它说"对" Agent 才敢晋升。**裁判本身是否可信，是整个项目的地基**——如果裁判会接受算错的 kernel，Agent 一切"加速"都是空中楼阁。

所以本次验收验证的是：**在固定输入下，项目裁判的三种行为都符合预期**：

| Case | 固定输入（同一契约） | 明确预期（跑之前写死在 cases/*.json） |
|---|---|---|
| A 接受正确 | fixture task.json + baseline.py | status=ok，correctness.passed=true |
| B 接受优化 | 同上 + 单变量候选 vs incumbent=baseline | status=ok，correctness.passed=true，有配对比较 |
| C 拒绝错误 | 同上 + 算错的候选（x*y 冒充 x+y） | status=wrong_answer，correctness.passed=false |

外加 Advanced 单变量实验：对"baseline 慢"这一失败事实只改一个变量（block_size 256→512→1024→2048→4096），同一案例重跑，记录实测耗时。

固定输入身份用 sha256 冻结（`frozen_inputs.json`），每次运行前重新核对，输入被改动则拒绝运行。

## 二、改了什么（含问题修复记录）

**没有改动被测项目的任何一行代码**——验收的前提是不改被测对象。改的只是运行环境。过程中真实遇到并修复了 2 个问题：

### 问题 1：裁判子进程找不到 torch（ModuleNotFoundError）

- **现象**：首次在 vm-60 上跑裁判，worker 子进程报 `No module named 'torch'`，status=compile_error。
- **归因**：vm-60 的 torch 2.9.1 装在 **user site-packages**（`~/.local/lib/python3.10/site-packages`），而裁判为隔离候选代码，用 `python -I`（isolated，隐含 -s 禁 user site）+ `PYTHONNOUSERSITE=1` + 全新 HOME 起子进程。三者叠加，user site 对子进程完全不可见。三种环境模式（隔离/半隔离/完整环境）复现结果一致，排除环境变量干扰。
- **修复**：建 `--without-pip --system-site-packages` venv（机器无 ensurepip、无 sudo），把 user site 的包软链进 venv site-packages。`-I` 模式下 venv 自身 site-packages 仍然可见，torch/triton 恢复导入。
- **教训**：项目声明的环境是 uv + ROCm + Py3.12（见 pyproject.toml），实际共享机是 CUDA + Py3.10 + user-site 安装。**"依赖版本"不只是版本号，还包括安装位置与解释器隔离策略**。

### 问题 2：triton 后端注册表为空（0 active drivers）

- **现象**：修完问题 1 后，torch 能导入了，但 triton 报 `RuntimeError: 0 active drivers ([]). There should only be one.`
- **归因**：triton 3.5.1 通过 `importlib.metadata` 的 `triton.backends` entry points 发现后端（`triton/backends/__init__.py` 的 `_discover_backends`）。软链只链了包目录，没链 `*.dist-info` 元数据，entry points 查不到 → 注册表为空 → 0 个活跃驱动。
- **修复**：把 user site 全部 dist-info/egg-info（467 个）软链进 venv。修复后注册表正确显示 nvidia CudaDriver 活跃、amd HIPDriver 不活跃——驱动选择逻辑本身工作正常。
- **教训**：现代 Python 包的"可发现性"依赖打包元数据而不只是文件；软链/复制包目录时必须连元数据一起。

## 三、实际结果（全部来自 evidence/ 原始 JSON，未挑选）

### 3.1 三个验收 Case（裁判原始输出）

| Case | status | correctness | 实测 | 墙钟 | 预期对账 |
|---|---|---|---|---|---|
| A baseline | **ok** | passed（maxAbsError=0.0，与参考逐位一致） | median 0.0944 ms，MAD 0.0007 ms，25 样本 | 3.591 s | ✅ 符合 |
| B block=1024 vs baseline | **ok** | passed（maxAbsError=0.0） | incumbent 0.09530 ms → candidate 0.09468 ms，improvementFraction **+0.81%**（配对 13 组中位数） | 3.596 s | ✅ 符合 |
| C x*y 冒充 x+y | **wrong_answer** | **failed**（maxAbsError=20.61，seed 17 即被抓出） | 无 benchmark（正确性不过不测性能） | 3.240 s | ✅ 符合 |

### 3.2 Advanced 单变量扫描（block_size，其余不动，同一 incumbent=baseline）

| block_size | median (ms) | vs baseline |
|---|---|---|
| 256（基线） | 0.09531 | — |
| 512 | 0.09479 | -0.51% |
| 1024 | 0.09494 | -0.50% |
| 2048 | 0.09592 | -0.03% |
| 4096 | 0.09537 | -0.08% |

### 3.3 Rubric 门（最小验收门，失败即关闭）

6 条全过：G1 固定输入身份 ✅ · G2 caseA ✅ · G3 caseB ✅ · G4 caseC ✅ · G5 扫描 ✅ · G6 证据完整性 ✅ → verdict: ACCEPTED。
负向自测：把 caseA 证据篡改成失败后，门 **passed=False**（G6 防篡改对账抓出）→ 证明"失败就关闭"不是空话。

## 四、证据与解释

**证据清单**（`evidence/task7/`，每个 case 含裁判原始 JSON + 未裁剪终端输出 + 墙钟）：

- `caseA/B/C_*.result.json` + `*.terminal.txt`：裁判原始输出与完整命令行
- `manifest.json`：固定输入 sha256 冻结与核对、环境（torch/triton/GPU）、每 case 预期对账明细
- `sweep_block_size.json` + `sweep_candidates/`：单变量扫描全部数据点与生成的候选源码
- `rubric_gate.json`：门的逐条判定与最终裁决
- `run_full.log`：整套验收的未裁剪运行日志

**解释三个关键数字**：

1. **caseC 的 maxAbsError=20.61 说明门禁抓错是定量的**：N(0,1)×N(0,1) 与 N(0,1)+N(0,1) 的差在 8.4M 样本上必然大幅偏离，且 poison-reset（输出先填 NaN）保证了"没写满"也会被发现——这个门禁不是摆设。
2. **扫描全部 ≈0.095ms 不是实验失败，而是有信息量的结果**：该 kernel 搬运 50.33 MB（3×N×2B）用 0.0944 ms，有效带宽 ≈ 533 GB/s，已达 4090D 峰值（~1008 GB/s）的约 53%；256 个线程/块对 114 个 SM 已有足够并行度，block_size 不改变访存模式，因此**该规模下它不是瓶颈**。各点差异（±0.8%）在配对测量的噪声量级内。
3. **caseB 的 +0.81% 不能当成加速汇报**：同配置（block=1024）在扫描里是 -0.50%，在 caseB 配对里是 +0.81%——两次实测符号相反，差异全部来自噪声。正确表述是"单变量修改未产生可测量收益"，这与 Day5 报告里 KV buffer 案例的方法论一致：**先测量再归因，不把噪声当结论**。

## 五、下一步修正

1. **把验收门挂进 Day5 主流程**：rubric_gate.py 目前是独立脚本，下一步作为 Agent 每轮晋升的前置条件（accept_candidate 前先过门），失败即拒绝晋升并记入 trajectory。
2. **补齐 negative-control 的 policy 维度**：当前 caseC 测的是正确性门禁；还应加一个"能过正确性但违反源码政策"的候选（如 host 端藏计算），验证 invalid_candidate 路径。
3. **换规模重扫单变量**：当前 N=8.4M 下 block_size 无差异；按 Day5 的结论随规模翻转的经验，应在 N=64M+ 或 batch 化场景重扫，检验"结论随规模翻转"是否复现。
4. **token/耗时维度的验收**：本次记录的是 GPU 侧耗时（中位数/MAD/配对改进）与墙钟；Agent 轮次的 token 消耗需在配置 LLM Key 后补测（本机无 DeepSeek Key，未虚构数据）。

---

## 附：模板提交、依赖版本与启动步骤

- **模板提交**：遵循本仓库 `courses/01-hello-gpu/evidence/README.md` 的打卡材料规范（真实运行证据、未裁剪终端输出、同契约同计时协议对比）；本报告 + evidence/task7/ 即提交物。
- **依赖版本**：torch 2.9.1+cu128、triton 3.5.1、Python 3.10.12、Linux 6.8、RTX 4090D（裁判侧声明版本为 torch 2.10.0+rocm7.12.0 / triton 3.6.0+rocm7.12.0 / Py≥3.12，本次为 CUDA 降级环境，差异已在问题 1/2 的记录中如实说明）。
- **启动步骤**：见 `task7/README.md`。
