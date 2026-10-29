# Task6｜记住用户偏好，执行前等待审批（第 8–9 章）

course_id: deepagents-in-action
task_id: Task6
source_id: courses/09-deepagents-in-action/evidence/task6/registration.json; courses/09-deepagents-in-action/task6/experiment-plan.json
execution_scope: 2026-10-27 本轮；本地开发环境（Python 3.12.14 / deepagents 0.7.24）＋ 远端 vm-60 主锁环境（Python 3.13.13 / deepagents 0.7.8）＋ 真实模型 DeepSeek-V4-Flash（AMD 端点）；非主网关部署，未改动 Task1–Task5 任何冻结文件

> **状态：三档目标全部有实验留证——离线机械链 24/24 × 2 环境通过，真实模型 5/5 检查通过；开篇问题（为什么新对话就忘）已用实验回答。** 样本量小（live 每例 n=1），不宣称统计显著；教程正文因本机出网受限未能在线逐字核对，实验设计以 SDK 源码与课程 notebook 目录为依据；未提交表单。

## 开篇问题的答案（先用一句话，再用实验证明）

**同一个对话里 Agent 记得你，是因为对话状态（消息和文件状态）保存在这个 thread 自己的 checkpoint 里，是 thread 作用域的；新对话 = 新 thread_id，从零开始，什么都看不到。要跨对话记住，必须把偏好写到对话之外的持久存储里：文件（`/memory/AGENTS.md`，由 MemoryMiddleware 在启动时注入 system prompt）或 Store（StoreBackend，按 namespace 跨线程持久化）。**

这个答案不是猜的：E1 的 H3 检查直接演示了"同线程记得、新线程失忆"（StateBackend），H1/H1b 演示了"文件在新线程/新进程里被注入"，两者对照就是机制本身。

## 验证什么

- **直觉**：记忆和审批是两套独立机制。记忆解决"写在哪里"：thread state 是对话内的，文件和 Store 是跨对话的。审批解决"什么时候执行"：HumanInTheLoopMiddleware 在工具执行**前**中断，人给四种决定（approve / edit / reject / respond），文件系统的 deny 是工具自己的硬闸门。
- **预测**：
  - H1：偏好写进 `/memory/AGENTS.md` 后，一个**全新 thread** 的 system prompt 会包含它（H1b：全新**进程**也会）；
  - H2（对照）：没有这个文件，新 thread 的 system prompt 不含偏好；
  - H3（开篇问题）：默认 StateBackend 下，同线程写文件同线程可见，**新线程不可见**；
  - H4/H5/H6：`interrupt_on={"write_file": True}` 时，approve 后按**原始参数**执行、reject 后**不执行**、edit 后**只按修改后参数**执行，且中断发生在执行**前**（磁盘在恢复前无文件）；
  - H7：同一 `InMemoryStore` + `namespace=(user_id,"files")` 下，alice 的记忆 alice 的新线程看得到，bob 看不到（用户间隔离）；
  - H8–H12：风险分层 `allow / interrupt / deny` 各自按预期生效，且 deny 规则声明在前则优先（first match wins）。
- **小实验**（全部留证，固定输入，seed 无关——scripted 是确定性的）：
  - 离线四组：E1 记忆（7 检查）、E2/E3/E4 审批（7 检查）、E5 Store 隔离（4 检查）、E6 风险分层（6 检查），共 **24 个机械检查**；
  - 真实模型 4 例：L1 写记忆（真实用户话语）、L2 **新进程**读记忆、L3 拒绝删除、L4 批准删除；
  - 两个机制探针：`ls` 对不存在的 root 的行为、记忆文件改动后同线程是否重新注入。
- **反例与边界**（本轮实测，都不是先验预期）：
  1. **直接调 backend 不受 permissions 保护**（H12）：`backend.write("/secrets/x.md")` 照常成功——权限只挂在 FilesystemMiddleware 的**工具级**，不挂 backend 级；
  2. **edit 改道 deny 路径仍被拒**（H11）：人把被中断的 `/drafts/ok.md` 编辑成 `/secrets/evil.md`，工具重入时仍过 deny 检查，两个文件都没落盘；
  3. **FilesystemBackend 不创建 root_dir**（探针）：root 不存在时 `ls("/")` 返回 `path_not_found`，`write_file` 却能顺手建目录——首跑 live L1 时真实模型因此陷进 ls 循环（失败记录见下）；
  4. **记忆注入每个 thread 只加载一次**（探针）：thread 内改磁盘文件，同线程再次 invoke **不会**重新注入；只有新 thread 才拿到新内容。

## 改了什么

1. **新增 Task6 实验代码**（未动 Task1–Task5 任何文件；哈希见[注册文件](<courses/09-deepagents-in-action/evidence/task6/registration.json>)）：
   - [lab_common.py](<courses/09-deepagents-in-action/task6/lab_common.py>)：确定性 ScriptedModel（步骤可以是"看到 system prompt 里有什么就答什么"的回调）+ Recorder（记录**真实**模型请求消息与工具事件）；
   - [run_memory_lab.py](<courses/09-deepagents-in-action/task6/run_memory_lab.py>)：E1 四阶段（写 / 新进程读 / 对照 / StateBackend 对照）；
   - [run_approval_lab.py](<courses/09-deepagents-in-action/task6/run_approval_lab.py>)：E2/E3/E4 三个独立案例（approve / reject / edit），各自 fresh root + fresh thread；
   - [run_store_lab.py](<courses/09-deepagents-in-action/task6/run_store_lab.py>)：E5 alice/bob 同 store 双向往返隔离；
   - [run_risk_lab.py](<courses/09-deepagents-in-action/task6/run_risk_lab.py>)：E6 四层风险规则 + 顺序变体 + 两个边界案例；
   - [run_live_lab.py](<courses/09-deepagents-in-action/task6/run_live_lab.py>)：真实模型四例，密钥只从远端 key 文件进进程，不进任何记录；
   - [probes/](<courses/09-deepagents-in-action/task6/probes/>)：两个机制探针脚本与输出。
   - [tests/](<courses/09-deepagents-in-action/task6/tests/>)：53 项回归测试（test_labs.py 29 项与 runner 检查 1:1；test_risk_scheme.py 24 项覆盖分级表与端到端闸门，含负向/对照）。
2. **live runner 的一次真实修复**：首跑 L1 失败（模型 4 次调用全是 `ls`，全被 `path_not_found` 打回，从未写文件）。根因不是模型不懂指令，是 FilesystemBackend 不建 root_dir（探针独立证实）。修复=在 runner 里显式 `mkdir` root 后重跑通过。**该次失败的原始 JSON 被重跑覆盖、未保留**，此处如实说明；保留的是探针证据与重跑后的完整证据。
3. **环境与沙箱**：本机子进程不能写 workspace（实测 PermissionError），所有实验输出写 TEMP/远端 /tmp，证据文件经宿住文件 API 原样转运，逐文件 sha256 核对一致（见[注册文件](<courses/09-deepagents-in-action/evidence/task6/registration.json>)）。远端主锁 0.7.8 与本地 0.7.24 **分开分母**，不混用。
4. **未做**：未改 Skill、未发布、未提交表单；Task5 全部证据与分享 ZIP 字节未动。

## 实际结果

### 1. 离线机械链（scripted，两环境分开）

| 实验 | 检查 | 本地 0.7.24 | 远端 0.7.8 主锁 |
|---|---|---:|---:|
| E1 跨线程文件记忆 | 7 | 7/7 | 7/7 |
| E2/E3/E4 审批（批准/拒绝/编辑） | 7 | 7/7 | 7/7 |
| E5 StoreBackend 用户隔离 | 4 | 4/4 | 4/4 |
| E6 风险分层与边界 | 6 | 6/6 | 6/6 |
| E7 风险分层审批方案（学余力，15 单位＋11 端到端） | 26 | 26/26 | 26/26 |
| **合计** | **50** | **50/50** | **50/50** |

证据：[offline-local-01/](<courses/09-deepagents-in-action/evidence/task6/offline-local-01/>)（四份 summary + 每条含真实 system prompt 原文）、[offline-remote-078-01/](<courses/09-deepagents-in-action/evidence/task6/offline-remote-078-01/>)。脚本化运行**零真实模型调用**，只证机械链，不证语义。

另有一套 pytest 回归：[tests/test_labs.py](<courses/09-deepagents-in-action/task6/tests/test_labs.py>) **53 项全部通过**（本地 0.7.24，含方案测试 24 项；输出见 [pytest-output.txt](<courses/09-deepagents-in-action/evidence/task6/offline-local-01/pytest-output.txt>)）：每个 runner 的检查名 1:1 镜像成测试，外加 H2/H3/H5/H6/H7b/H10/H10b/H11 负向与对照断言；冻结 runner 与 lab_common 字节未改（未发现 bug）。本机沙箱两个环境限制已绕行并留证：pytest 装在 DSH TEMP 目录经 PYTHONPATH 加载；tempfile.mkdtemp() 的 0o700 目录在本机丢写权限，测试助手探针验证后回退默认模式目录。

### 2. 真实模型（vm-60 主锁 0.7.8 + DeepSeek-V4-Flash，AMD 端点）

| 案例 | 内容 | 模型调用 | 结果 |
|---|---|---:|---|
| L1 写记忆 | "记住：我喜欢中文回答……写进 /memory/AGENTS.md" | 2 | `write_file` 落盘，内容为模型自组织的 Markdown（"# 用户偏好 - 用户喜欢用中文回答。- 回答要简洁。"） |
| L2 新进程读记忆 | "我喜欢什么语言的回答？" | 1 | 答"中文"并**引用记忆文件作为依据**；记录的 system prompt 确实含该文件内容 |
| L3 拒绝删除 | "把 /notes/todo.md 删掉" → interrupt → **reject** | 2 | 文件**原样存在**；工具 **start 事件都没出现**（从未执行） |
| L4 批准删除 | 同上 → interrupt → **approve** | 2 | 文件被删；模型确认"已删除" |

证据：[live-ds-01/](<courses/09-deepagents-in-action/evidence/task6/live-ds-01/>)（四份 summary，含每次真实请求）。**每例 n=1，temperature=0 不是确定性保证**；端点当日健康（首跑 ping 成功），但这是单日单窗口观察。

### 3. 机制边界（探针，两环境一致）

| 边界 | 实测 | 证据 |
|---|---|---|
| `ls` 与 root_dir | root 不存在时 `ls("/")`→`path_not_found`；`write_file` 却能建目录 | [probes/ls-root-probe-0.7.8.txt](<courses/09-deepagents-in-action/evidence/task6/probes/ls-root-probe-0.7.8.txt>) |
| 记忆重新注入 | thread 内改磁盘文件，同线程再 invoke **不**重新注入；新 thread 才注入 | [probes/mem-reload-boundary.txt](<courses/09-deepagents-in-action/evidence/task6/probes/mem-reload-boundary.txt>) |

### 4. 学余力项：按风险等级分层的审批方案（15 项单位分级＋11 项端到端，× 2 环境）

设计文档 [risk-tier-approval.md](<courses/09-deepagents-in-action/task6/risk-tier-approval.md>)，可复用策略模块 [risk_policy.py](<courses/09-deepagents-in-action/task6/risk_policy.py>)，实验 [run_risk_scheme_lab.py](<courses/09-deepagents-in-action/task6/run_risk_scheme_lab.py>)：

- **五层分级**：L0 只读自动放行 / L1 低风险写入（public 区）自动放行 / L2 中风险写入问人（approve、edit、reject）/ L3 难撤销（delete、敏感读、批量读到敏感区）问人但**只给 approve、reject**（不给 edit：改路径不降低删除的风险类）/ L4 禁止（secrets 路径、凭证样式内容）直接拒、不问人。
- **两条 fail-closed 默认**：未分类路径写入→L2 问人（SDK 默认是静默 allow，方案显式改掉）；不认识的工具→L2。
- **编译成 SDK 三层执行**：路径规则→FilesystemPermission（含 secrets 读的 interrupt 规则，让 SDK 自动生成**批量感知**谓词，补上无路径 grep／根目录 ls 的洞）；每工具决定集→interrupt_on（用户条目按工具名覆盖权限生成条目，实现分层收窄）；内容红线→RiskGateMiddleware（权限模型看不到内容，凭证样式写入在执行前被拒）。
- **审计三类事件缺一不可**（实验踩出来的）：执行判定（中间件）＋审批请求（见到 __interrupt__ 时）＋人的决定（resume 时）。被拒的调用不进执行阶段，只记执行就会漏掉一半事实。
- **结果**：单位分级表 15 例＋端到端 10 例（L1 直行不中断、L2/L3 中断且决定集正确、L4 路径/内容双拒且不中断、未分类 fail-closed、L3 上给 edit 决定被 SDK 拒绝、审计三类齐全），本地 0.7.24 与远端 0.7.8 主锁**逐项一致**。证据：[offline-local-01/risk-scheme-summary.json](<courses/09-deepagents-in-action/evidence/task6/offline-local-01/risk-scheme-summary.json>)、[offline-remote-078-01/risk-scheme-summary.json](<courses/09-deepagents-in-action/evidence/task6/offline-remote-078-01/risk-scheme-summary.json>)。

### 5. 失败与保留

- live L1 首跑：模型陷 ls 循环、未写文件（根因：探针 1）。修复后重跑通过；失败原始记录未保留，已如实说明。
- 其余运行 0 执行错误；远端与本地结果逐项一致，无版本分歧。

## 证据与解释

| 本稿结论 | 证据对象 | 不能据此推出 |
|---|---|---|
| 新 thread 能读到文件记忆（H1/H1b） | [memory-summary.json](<courses/09-deepagents-in-action/evidence/task6/offline-local-01/memory-summary.json>) 的 `thread_B_fresh_read.system_prompt_contains_preference` 与请求原文；远端同项在 offline-remote-078-01 | 真实模型一定"理解"记忆内容（scripted 只证注入）；文件记忆适合所有场景 |
| 没有文件就没有记忆（H2） | 同文件 `control_phase.system_prompt_contains_preference=false` | 记忆只能来自这一个路径（Store 也行，见 E5） |
| 开篇问题：state 是 thread 作用域（H3） | 同文件 `state_backend_phase`：同线程 `ls` 见 `pref.md`、新线程不见 | StateBackend"坏了"；它本来就不是跨 thread 存储 |
| 批准后执行、拒绝后不执行（H4/H5） | [approval-summary.json](<courses/09-deepagents-in-action/evidence/task6/offline-local-01/approval-summary.json>) 的 `pre_resume_disk`（空）/`post_resume_disk`（approve 有、reject 无） | 审批能防住所有危险操作（见边界 1/2） |
| edit 只执行修改后参数（H6） | 同文件 edit 案例 `disk_files == EDITED` 且 `!= PROPOSED` | 编辑后的调用跳过所有检查（H11 反例：仍过 deny） |
| 用户间隔离（H7） | [store-summary.json](<courses/09-deepagents-in-action/evidence/task6/offline-local-01/store-summary.json>)：bob 的 system prompt 与 `ls` 都看不到 alice 文件，反之亦然 | 任意 namespace 方案都安全；生产还要考虑 store 本身的访问控制 |
| 风险分层生效（H8–H10b） | [risk-summary.json](<courses/09-deepagents-in-action/evidence/task6/offline-local-01/risk-summary.json>)：public 无中断直写、drafts 中断后批准执行、secrets 拒写无中断、deny 声明在前则优先 | 这套规则适合所有工具（只覆盖了文件系统三类工具） |
| 直接 backend 调用不受权限保护（H12） | 同文件 `H12_direct_backend`：`backend.write("/secrets/…")` error=None 且落盘 | 有 permissions 就"安全"了；闸门在工具级，绕过工具就绕过它 |
| 真实模型端到端可用（L1–L4） | [live-ds-01](<courses/09-deepagents-in-action/evidence/task6/live-ds-01/>) 四份 summary | 该行为可复现、普遍成立（n=1/例；未做重复与换模型） |
| 教程正文依据 | 课程 notebook 目录 `ch08-long-term-memory`（跨线程记忆与文件隔离）、`ch09-human-in-the-loop`（工具审批与中断恢复）+ 本地 SDK 0.7.8/0.7.24 源码 | 与教程文字逐句一致（本机 web_fetch 提供方配置错误、pwsh 出网被禁，正文未能在线取阅） |

**核心解释：存储边界和审批边界是两条独立的线。**

- **存储边界**：`StateBackend`（默认）把文件放在 graph state 里，随 thread 生灭——这就是"新对话就忘"的机制根源；`FilesystemBackend` 把文件放在磁盘上，跨 thread 跨进程都在，但没有用户隔离；`StoreBackend` 把文件按 namespace 放进 Store，同 store 内用户间互相不可见。MemoryMiddleware 只负责"启动时把文件读进 system prompt"，而且**每个 thread 只读一次**（探针 2）——所以偏好写进文件只是半个机制，另一半是"下次是个新 thread"。
- **审批边界**：中断发生在 `after_model`、工具执行**之前**（L3 连工具 start 事件都没有，是最干净的证据）；人有四种决定，`respond` 本轮未测；`edit` 后的调用会**重入工具**再过一次 deny 检查（H11），所以"改道"改不出权限；permissions 是**工具级**闸门，直接拿 backend 用的人不受它管（H12）——审批保护的是"Agent 经工具出手"这条路，不是"所有代码路径"。

## 下一步修正

1. **live 重复与换模型**：每例补到 ≥3 次重复，并换一个模型（如远端 ollama qwen3:1.7b 或更大本地模型）观察 L2 是否仍引用记忆文件作答；换模型必须另立注册、不合并分母。
2. **审批的持久化边界未测**：本轮 MemorySaver 的中断状态随进程生灭。用 SqliteSaver 做一次"中断→进程退出→重启→批准→执行"，验证审批状态留在 checkpointer 里而不是工具里；这决定"人不在线时审批能不能排队"。
3. **respond 决定未测**：四种决定只测了三种；`respond`（人代替工具作答、不执行）值得补一个案例，尤其是它和 reject 的语义差别。
4. **风险分层只覆盖文件系统**：把分层模型扩展到 shell/网络类工具（若 harness 提供），并明确"哪一层绝不自动放行"；当前四层规则是教学配置，不是安全基线。
5. **教程原文核对**：网络恢复后对照 ch08/ch09 正文与配套 notebook，核对本轮 H 编号与教程示例的对应关系；若教程要求不同（如指定用 `edit_file` 而非 `write_file`），按另立版本补跑，不改本轮冻结证据。
6. **打卡物料**：本稿 + [实验讲解](<courses/09-deepagents-in-action/task6/实验讲解.md>) 为本地材料；表单未提交，发布/分享前另获用户确认。