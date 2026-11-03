# Quanty · Task8 可复现 Demo：固定输入 + 明确预期

> **一句话**：Quanty 是一个把"算数"关在确定性工具里、把"记忆 / 审批 / 权限"交给 Deep Agents 机制栈的量价研究助理；本目录用**一条离线命令、22 项事先声明的检查**，让任何人在 3 分钟内验收它的 Agent 机制链——不需要 API Key、不依赖网络、与 temperature/seed 无关。

教程地址：<https://datawhalechina.github.io/deepagents-in-action/> ｜ 本仓库课程目录：`courses/09-deepagents-in-action/`

---

## 1. 3 分钟跑通

**环境要求**：Python 3.12+（开发与留证环境为 3.12.14）；无需 GPU、无需 API Key、无需网络（安装依赖除外）。

Windows（PowerShell）：

```powershell
cd courses\09-deepagents-in-action
py -3.12 -m venv .venv                                    # 1) 建虚拟环境
.venv\Scripts\pip install -r requirements.lock.txt        # 2) 装锁定依赖（deepagents 0.7.24）
.venv\Scripts\python task8\run_demo.py                    # 3) 跑 Demo（本机实测约 6–8 秒）
```

Linux / macOS：

```bash
cd courses/09-deepagents-in-action
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python task8/run_demo.py
```

**明确预期**：终端打印 22 行 `[PASS]`，最后三行是——

```
总结：22/22 项检查通过
结论：全部通过 —— 机制链按固定输入与明确预期工作
证据：<你选择的输出目录>/summary.json
```

进程**退出码为 0**；任何一项检查不符即打印 `[FAIL]` 并以**退出码 1** 结束（可直接用于 CI）。
默认输出写到 `./demo-output/`； `--out DIR --root DIR` 可自选位置。

---

## 2. 固定输入（全部写死，指纹可核对）

Demo 启动时先打印"固定输入"清单，包括四个关键文件的 sha256——任何人改动 `run_demo.py` / `tools.py` / `data/demo.csv` / `task6/lab_common.py` 都会在指纹上立刻现形：

| 类别 | 固定值 |
|---|---|
| 用户话语 | "分析 DEMO 最近5个交易日上涨是否伴随成交量放大，列出依据和局限。" 等 3 句量价提问 + 4 句记忆/审批提问 |
| 模型驱动 | `ScriptedModel` 确定性脚本（步骤可依赖"实际收到的 system prompt"），**零真实模型调用** |
| 数据 | `data/demo.csv`（10 条合成收盘价/成交量，DEMO 标的） |
| 记忆 | 偏好"用户偏好：我喜欢中文回答，回答要简洁。" → `/memory/AGENTS.md` |
| 审批 | 目标 `/reports/report.md`；提议内容 / 修改后内容写死 |
| 风险规则 | 读全放行；`/public/**` 写放行；`/drafts/**` 写问人；`/secrets/**` 写拒绝（声明序，首个匹配生效） |

## 3. 明确预期（22 项，运行前声明）

每一项检查的预期值都写在 `run_demo.py` 的 `EXPECTED` 表里（不是跑完再补），例如：

| 检查 | 预期 |
|---|---|
| `A1_tool_values_match_hand_computed` | window=5：收益 6.0%、基准日均量 1000.0、近期日均量 1600.0、量比 1.6 |
| `A1c_window3_values_match_hand_computed` | window=3：收益 2.912621%、日均量 1300.0/1700.0、量比 1.3076923076923077 |
| `A2a_tool_boundary_raises_with_exact_message` | window=10/0/-1/"5"/5.0 全部 `ValueError: window must be an integer between 1 and 5` |
| `A2b_agent_propagates_boundary_error` | Agent 真去调 window=10 时，同一 ValueError 冲出 `agent.invoke()` |
| `B2_fresh_thread_system_prompt_has_memory` | 新线程 system prompt 含记忆文件内容 |
| `B6_state_fresh_thread_blind` | StateBackend 下新线程看不到同线程写过的文件 |
| `C2_reject_does_not_execute` | reject 后磁盘无文件 |
| `C6_secrets_write_denied_without_asking` | secrets 写入被拒且不触发人工中断 |
| `C9_direct_backend_call_not_gated` | 直接 `backend.write` 绕过工具级权限（边界，非漏洞宣传） |

完整 22 项见 `run_demo.py` 的 `EXPECTED` 表；每次运行的逐项"预期 vs 实际"与完整证据在 `summary.json`。

## 4. 三段演示 = 四项课程能力

| 段落 | 课程能力 | 检查数 |
|---|---|---|
| A 工具调用与确定性边界 | Tools + 拒绝替代/拒绝编造（Task1 主线） | 7 |
| B 长期记忆 | ch08 文件记忆跨线程/跨进程 + StateBackend 对照（Task6 E1） | 6 |
| C 人工审批与风险分层 | ch09 interrupt_on 三种决定 + L0–L3 权限分层与两条边界（Task6 E2–E6） | 9 |

（另两项课程能力——Task5 Skills 渐进披露、Task6 风险分层**审批方案** L0–L4——分别见 `courses/09-deepagents-in-action/task5/` 与 `task6/risk-tier-approval.md`；Skills 触发需要真实模型，不在本离线 Demo 内。）

## 5. 它证明什么 / 它不证明什么

**证明**：上述机制链在脚本化驱动下按固定输入产生声明过的行为；工具计算与手工计算逐位一致；边界以精确异常拒绝；审批中断发生在执行前；权限分层与声明序生效。全部可重复：同机重复运行结果一致（见 `evidence/task8/` 两次留证）。

**不证明**：真实模型会这样行为——`ScriptedModel` 的回答文本是脚本，不是语义证据；本 Demo 零真实模型调用。真实模型行为另见 Task6 留证 `evidence/task6/live-ds-01/`（每例 n=1）。同理，22/22 通过不构成"安全性证明"：`C9` 恰恰证明权限是**工具级**闸门，绕过工具就绕过它。

## 6. 目录结构

```
task8/
├── run_demo.py      # Demo 本体：固定输入 + EXPECTED 表 + 三段运行 + 逐项比对
├── README.md        # 本文件（3 分钟跑通）
└── reproduce.md     # 同伴复现协议与互评清单
evidence/task8/      # 实际运行留证（stdout + summary.json + 注册文件）
```

## 7. 常见问题

- **没有 API Key 能跑吗？** 能。Demo 全程零真实模型调用；`agent.py`（真实模型入口）才需要 `QUANTY_API_KEY` 或 `AMD_API_KEY`。
- **为什么会写 `./demo-output/`？** Demo 需要落证据（`summary.json`）和文件系统后端沙箱；`--out`/`--root` 可改。
- **升级依赖后检查失败了？** 预期值基于 `tools.py` 计算与 `data/demo.csv` 指纹；SDK 大版本可能改变机制行为（中断语义、权限模型）。本仓库锁定 `deepagents==0.7.24`，见 `requirements.lock.txt`。
- **`A2b` 为什么"预期失败"？** 不是失败：它预期并证实"工具异常会冲出 `invoke()`"——这是本机 langgraph 1.2.15 的真实机制（`_default_handle_tool_errors` 直接 `raise`），调用方必须自己接住。这是能力边界，不是 bug 宣传。
- **想跑真实模型？** `python agent.py "你的问题"`（只支持合成 DEMO 数据，不可冒充真实行情）。

## 8. 复现与互评

同伴独立复现步骤与互评清单：[reproduce.md](reproduce.md)。作者本轮两次留证（含一次"陌生人视角"新进程默认参数运行）：[../evidence/task8/](../evidence/task8/)。
