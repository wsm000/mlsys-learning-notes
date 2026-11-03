# Task8｜可复现 Demo 与结营交流（第 10 天–11 天）

course_id: deepagents-in-action
task_id: Task8
source_id: courses/09-deepagents-in-action/evidence/task8/registration.json; courses/09-deepagents-in-action/task8/run_demo.py
execution_scope: 2026-11-02 本轮；本地开发环境（Python 3.12.14 / deepagents 0.7.24 锁定依赖）；非主网关部署；未改动 Task1–Task6 任何冻结文件（task6/lab_common.py 以冻结字节被复用，哈希见注册文件）

> **状态：三档目标全部有留证。** Lite：代码＋README＋两次演示记录齐备。Basic：同一个改造后的应用（Quanty），Demo 综合 4 项课程能力，相对原模板 5 项实质改造，同伴复现协议与互评清单已备（真实同伴互评待进行，未冒充）。Advanced：陌生人视角独立跑通已自证（新进程＋默认参数＋陌生 cwd，22/22，退出码 0）；改造、架构选择、失败处理、能力边界在本稿回答。投稿案例库/提 PR 未做（需用户确认）；表单未提交。

## 开篇问题的答案（先用一句话，再用实验证明）

**"陌生人 3 分钟能跑通你的项目吗？"——能，前提是把"跑通"从感觉变成判定：一条命令、零依赖密钥、22 项事先声明的检查、明确的退出码。** 本机实测：一个新进程、仓库之外的陌生目录、不带任何参数，`python task8/run_demo.py` 用 6.1 秒跑完 22/22 项检查并以退出码 0 结束（[demo-run-02-stranger](<courses/09-deepagents-in-action/evidence/task8/demo-run-02-stranger/demo-stdout.txt>)）。

这个答案的方法论正是本课程 Task5/Task6 一脉相承的东西：**固定输入 + 明确预期**。Demo 启动先打印固定输入（含 4 个关键文件的 sha256），每一项检查的预期值写在 `run_demo.py` 的 `EXPECTED` 表里（运行前声明，不是跑完再补），实际值逐项比对，任何一项不符即以退出码 1 失败——可以直接挂进 CI。课程仓库 README 用一句话讲清一门课；本 Demo 的 README 也用一句话讲清这个项目，然后只给 3 条命令。

## 验证什么（对照三档目标）

- **Lite 打卡级**：代码（[task8/run_demo.py](<courses/09-deepagents-in-action/task8/run_demo.py>)）、README（[3 分钟跑通](<courses/09-deepagents-in-action/task8/README.md>)）、演示记录（[evidence/task8/](<courses/09-deepagents-in-action/evidence/task8/>) 两次运行的 stdout＋summary.json＋注册文件）三样交齐，本稿讲清做了什么。
- **Basic 基础级**：用同一个改造后的应用做展示（Quanty：`agent.py`＋`tools.py`＋Task6 机制栈，未为 Demo 另做玩具应用）；综合 **4 项**课程能力（≥3）；列出相对原模板的 **5 项**实质改造（≥3，见下）；完成同伴复现或互评——协议与清单已备（[reproduce.md](<courses/09-deepagents-in-action/task8/reproduce.md>)），本轮由作者以"陌生人视角"自复现代替，**不等同于同伴互评**，反馈表留空待真同伴填写。
- **Advanced 挑战级**：陌生人照 README 独立跑通的路径已用陌生人视角自证；改造、架构选择、失败处理、能力边界见本稿对应小节；投稿案例库/提 PR 未做。

## 改了什么（相对原模板的实质改造）

原模板指课程默认 deep agent 模板与本仓库 `main-app-snapshot/`（Quanty 主应用快照）。5 项改造，每项都有证据指向：

1. **工具层：确定性计算＋输入边界＋拒绝编造。** `tools.py` 的量价/RSI 计算不依赖模型；window 超界以精确 `ValueError` 拒绝（A2a：5 种坏输入同一消息）；数据不足返回 `insufficient_data` 且 `rsi=null`，不编造数值（A3）。
2. **提示词层：反替代、反完成宣言。** `agent.py` 的 system prompt 明确"window 超出 1–5 时不得调用较小 window 代替、不得宣称完成、先问人"，并标注合成数据不可冒充行情（A2c 机械验证拒绝路径无工具调用、无数值编造）。
3. **验收层：固定输入＋明确预期的离线骨架。** `task6/lab_common.py` 的 `ScriptedModel`＋`Recorder`（冻结复用）与 `task8/run_demo.py` 的 `EXPECTED` 表，把"看起来能跑"变成 22 项可判定检查，零真实模型调用、与 temperature/seed 无关。
4. **安全层：按风险分层的审批方案（Task6 学余力项）。** L0–L4 五层＋两条 fail-closed 默认＋编译进 SDK 三层执行＋三类事件审计（26 项检查 × 2 环境通过，设计见 [task6/risk-tier-approval.md](<courses/09-deepagents-in-action/task6/risk-tier-approval.md>)）。本 Demo 的 C 段是其文件系统子集的浓缩（L1/L2/L3＋两条边界）。
5. **方法层：把真实失败教训固化为 Skill。** evidence-checkin Skill v2 与 provenance 校验器（Task5）：定位符校验、错路径拒绝、执行状态分离记录——本轮 A1c 的手算修正正是这套纪律的实例（见"失败与保留"）。

## 实际结果

### 1. 三段演示 = 四项课程能力，22/22 通过

| 段落 | 课程能力 | 检查 | 结果 |
|---|---|---:|---|
| A 工具调用与确定性边界 | Tools＋拒绝替代/拒绝编造（Task1 主线） | 7 | 7/7 |
| B 长期记忆 | ch08 文件记忆跨线程/跨进程＋StateBackend 对照（Task6 E1 浓缩） | 6 | 6/6 |
| C 人工审批与风险分层 | ch09 interrupt_on 三决定＋L0–L3 权限分层＋两条边界（Task6 E2–E6 浓缩） | 9 | 9/9 |
| **合计** | **4 项能力** | **22** | **22/22** |

完整逐项"预期 vs 实际"见 [demo-run-01/summary.json](<courses/09-deepagents-in-action/evidence/task8/demo-run-01/summary.json>) 与 [stdout](<courses/09-deepagents-in-action/evidence/task8/demo-run-01/demo-stdout.txt>)。A1/A1c 的预期值由 `data/demo.csv` **独立手工计算**（window=5：6.0%/1000.0/1600.0/1.6；window=3：2.912621%/1300.0/1700.0/1.3076923076923077），与工具返回逐位一致。

### 2. 两次独立运行一致（确定性）

| 运行 | 方式 | 退出码 | 耗时 | 结果 |
|---|---|---:|---:|---|
| demo-run-01 | 作者本机，显式 --out/--root | 0 | 7.6s | 22/22 |
| demo-run-02-stranger | 新进程、默认参数、仓库外 cwd | 0 | 6.1s | 22/22 |

两份 summary.json 的 `checks`/`expected`/`env`/`fixed_input_fingerprints` **逐项相同**；`detail` 仅差运行特有的时间戳与 langgraph run_id。固定输入指纹（run_demo.py/tools.py/demo.csv/lab_common.py 的 sha256）两次相同。比对与哈希见 [registration.json](<courses/09-deepagents-in-action/evidence/task8/registration.json>)。

### 3. 环境与沙箱（如实记录）

- 依赖环境：既有 `.venv` 与 `requirements.lock.txt` 逐包一致（59/59，deepagents 0.7.24）。
- **本机沙箱无法复验"安装"这一步**：`python -m venv` 的 ensurepip 在 tempfile 清理 chmod 临时目录时被拒（WinError 5，与 Task6 记录同一环境限制，换 TMPDIR/TMP/TEMP 重试 3 次均失败）。因此"3 分钟"中安装环节未在本机复验；已复验的是：锁定环境＋陌生 cwd＋新进程＋默认参数＋退出码。已写入 registration.json 的 `protocol.venv_limitation`，不冒充。
- 证据转运：本机子进程只能写 TEMP、不能写 workspace；证据经宿住文件 API 原样转运，原文件 CRLF、仓库副本 LF，逐文件验证 LF 规范化后字节一致（注册文件内双份哈希）。

## 失败与保留（都是本轮真实发生的）

1. **A2b 发现了对作者不利的事实并把它变成检查项**：脚本化模型真去调用 window=10 时，`ValueError` 会**冲出 `agent.invoke()`**——本机 langgraph 1.2.15 的 `_default_handle_tool_errors` 直接 `raise`，工具异常不会变成模型可见的 error ToolMessage。这意味着 `agent.py` 目前**不 catch** 这个异常，真实使用中超界 window 会让 CLI 崩掉（待修，见下一步）。Demo 把"异常按精确消息传播"列为预期通过项（A2b），而不是藏起来。
2. **A1c 我的手算错了，工具是对的**：我最初把 window=3 的窗口算成第 4–9 行，实际 `rows[-6:]` 取第 5–10 行，工具返回 1300.0/1700.0/1.3076923076923077。修正的是**我的预期值**，不是工具。这个过程恰好证明"明确预期"是独立手工计算而非从输出倒推——若预期是从工具输出抄的，就永远发现不了这类分歧。
3. **stdout 日志第一次捕获成了 UTF-16**（pwsh `Tee-Object` 默认编码），读取失败后改用 `.NET WriteAllText`（UTF-8 无 BOM）重跑两次并留证；旧日志未保留，此处如实说明。
4. **段落 A 首次整体崩掉**：A2 原设计假设"工具异常会变成 error ToolMessage 供模型反应"，与实测机制不符，A 段 4 项检查全红。按"失败不静默跳过"的原则重构为 A2a/A2b/A2c 三项（工具层 raise／Agent 层传播／拒绝路径无调用），重跑 22/22。

## 证据与解释

| 本稿结论 | 证据对象 | 不能据此推出 |
|---|---|---|
| 22 项检查按固定输入与声明预期通过 | [demo-run-01/summary.json](<courses/09-deepagents-in-action/evidence/task8/demo-run-01/summary.json>) 的 `checks` 逐项 expected==actual | 真实模型会这样行为（scripted 只证机械链） |
| 陌生人视角可独立跑通 | [demo-run-02-stranger/demo-stdout.txt](<courses/09-deepagents-in-action/evidence/task8/demo-run-02-stranger/demo-stdout.txt>)：陌生 cwd、默认参数、退出码 0 | 任何机器任何版本都能跑通（只测了本机 Windows＋锁定依赖；安装环节未复验） |
| 两次运行确定一致 | registration.json 的 `determinism`：checks/expected/env/指纹逐项相同 | 统计显著或跨平台一致（同机两次重复） |
| 工具边界以精确异常拒绝 | summary.json `detail.段落 A.A2a_boundary_errors`（5 种坏输入同一消息） | 该边界能防住所有误用（只覆盖 window 参数） |
| 工具异常会冲出 invoke | `A2b_propagated_error` 与 langgraph 1.2.15 源码 `_default_handle_tool_errors` 直接 raise | 其他 SDK 版本行为相同（未测其他版本） |
| 审批中断发生在执行前 | C 段 `pre_resume_disk` 均为空、resume 后才有文件 | 审批能防住所有危险操作（C9：直接 backend 调用不受工具级权限管） |
| 记忆跨线程/跨进程注入、StateBackend 线程作用域 | B 段 B1–B6 与 Task6 E1 同族证据 | 文件记忆适合所有场景；记忆每 thread 只注入一次（Task6 探针） |

**核心解释：可复现性的关键不是"代码能跑"，而是"通过标准能被第三方判定"。** 固定输入让任何人拿到同一局面；事先声明的预期让通过/失败不依赖解释权；退出码让判定可以进 CI。这三件加起来，陌生人才可能 3 分钟跑通——否则他只能"看起来跑通了"。

## 能力边界（Advanced 档：敢说哪里不行）

1. **不证语义**：`ScriptedModel` 的回答文本是脚本；22/22 通过不说明真实模型会拒绝替代、会引用记忆。真实模型行为另见 Task6 [live-ds-01](<courses/09-deepagents-in-action/evidence/task6/live-ds-01/>)（每例 n=1）。
2. **样本与平台**：两次同机同版本重复；未测 Linux/macOS、未测其他 deepagents 版本；未复验 pip install。
3. **安全边界**：C9 恰证明权限是**工具级**闸门，绕过工具就绕过它；22/22 不是安全性证明。
4. **覆盖范围**：Demo 只覆盖文件系统三类工具与两个业务工具；respond 决定、SqliteSaver 跨进程审批排队未测（Task6 已列）。
5. **真实缺陷**：`agent.py` 不 catch 工具边界异常（A2b），超界 window 会崩 CLI——已知待修，未伪装成特性。

## 下一步修正

1. **修 caller 层**：`agent.py` 捕获工具边界 `ValueError`，转成"询问是否接受 5 条演示"的用户可读消息（与 system prompt 纪律对齐），并补一个对应检查项；另立版本，不改本轮冻结证据。
2. **同伴复现/互评**：把 [reproduce.md](<courses/09-deepagents-in-action/task8/reproduce.md>) 发给至少一位同伴，收回填写好的复现记录表与互评清单；只有真同伴的反馈才能兑现 Basic 档的"同伴复现或互评"。
3. **live 冒烟**：在锁定环境用真实模型跑一次 `agent.py` 固定问题，与 Demo 的 A1 预期值对照（真实模型＋确定性工具的组合验证）。
4. **发布边界**：投稿案例库、提 PR、提交表单均需另获用户确认；Task1–Task6 冻结文件继续不动。

## 结营交流：3 分钟怎么讲

1. **一句话**：Quanty 把算数关在确定性工具里，把记忆/审批/权限交给 Deep Agents 机制栈；我用一条离线命令验收它。
2. **跑**（30 秒）：`python task8/run_demo.py` → 22/22，退出码 0。
3. **讲方法论**（60 秒）：固定输入＋事先声明的预期＋退出码；A1 的 6.0%/1.6 是我按 CSV 手算的，不是从工具输出抄的。
4. **讲边界**（60 秒）：scripted 不证语义；A2b 工具异常会冲出 invoke（我的 CLI 现在会崩，待修）；C9 权限是工具级闸门。
5. **收尾**（30 秒）：复现协议和互评清单在 reproduce.md，欢迎照着 README 独立跑。
