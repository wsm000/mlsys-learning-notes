# 按风险等级分层的审批方案（Task6 Advanced+）

代码：[risk_policy.py](risk_policy.py)（策略模块，可直接 import）｜实验：[run_risk_scheme_lab.py](run_risk_scheme_lab.py)（11 项检查 × 2 环境全过）｜证据：[offline-local-01/risk-scheme-summary.json](../evidence/task6/offline-local-01/risk-scheme-summary.json)、[offline-remote-078-01/risk-scheme-summary.json](../evidence/task6/offline-remote-078-01/risk-scheme-summary.json)

## 1. 要解决什么问题

“执行前等待审批”如果对每个工具调用都弹窗，Agent 会被批准疲劳吞掉；如果都不弹，危险操作没人管。分层方案要回答三个问题：**哪个调用需要人看？人能看到什么？人能做什么决定？**

## 2. 分级（一个分类器说了算）

| 层 | 含义 | 例子 | 人是否需要看 | 人可做的决定 |
|---|---|---|---|---|
| L0 | 只读/可逆 | ls、read_file（非敏感）、glob、grep | 否 | — |
| L1 | 低风险写入 | 写 /public/** | 否（留审计） | — |
| L2 | 中风险写入 | 写 /drafts/**、写未分类路径、edit_file | **是** | approve / edit / reject |
| L3 | 高风险/难撤销 | delete、读 /secrets/**、批量读到敏感区 | **是** | approve / reject（**不给 edit**） |
| L4 | 禁止 | 写/删 /secrets/**、内容像凭证 | 否（问了也白问） | 不可批准，直接拒 |

分级依据四个维度：**操作类型**（读<写<删）、**路径敏感度**（public<drafts<secrets）、**可逆性**（新建<覆盖<删除）、**内容红线**（凭证样式）。内容红线优先级最高，deny 优先于 interrupt。

**两条 fail-closed 默认**（SDK 默认是“没有规则就允许”，方案显式改掉）：
- 未分类路径的写入 → L2（问人），不静默放行；
- 不认识的工具 → L2（问人）。

## 3. 编译成三层执行（scheme → SDK 参数）

`build_policy()` 把同一张分级表编译成 deepagents 的三件套：

1. **路径规则 → `FilesystemPermission`**：`/secrets/**` 读写删全部 deny；`/secrets/**` 的 read 用 interrupt——这条同时让 SDK 为 `ls/glob/grep` 自动生成**批量感知**的 `when` 谓词（无路径的 `grep`、根目录 `ls` 都会触发），补上手写谓词最容易漏的洞。
2. **每工具决定集 → `interrupt_on`**：用户提供的条目按工具名覆盖权限生成的条目，所以能把 L2 收窄成 `[approve, edit, reject]`、L3 收窄成 `[approve, reject]`。L3 不给 `edit` 的理由：改路径不降低“删除”的风险类，收窄决定面比给更多自由更安全。
3. **内容红线 → `RiskGateMiddleware`**：`FilesystemPermission` 只看路径看不到内容，凭证样式的写入由一个包住 `wrap_tool_call` 的中间件在执行前拒绝——这是权限模型表达不了的那一层。

## 4. 审计（三类事件，缺一不可）

实验里踩过才知道：只记“执行阶段的判定”会漏掉一半事实。完整审计要三类：

| 事件 | 记录者 | 内容 |
|---|---|---|
| 执行判定 | RiskGateMiddleware | 工具、参数摘要、层级、理由、是否执行、结果状态 |
| 审批请求 | runner（看到 `__interrupt__` 时） | 工具、层级、理由、当时允许的决定集 |
| 人的决定 | runner（resume 时） | 决定类型、理由 |

被拒绝的调用根本不进入执行阶段（决定在 `after_model` 处理），所以“执行审计”里不会有它——只有把审批请求和决定也记上，才能回答“谁在什么时候放行了什么”。

## 5. 实验验证的 11 项（两环境一致）

单位侧 15 例分级表 + 端到端 10 例：L1 直行不中断（S1）、L2 中断且决定集正确（S2）、L3 中断决定集更窄且拒绝后文件还在（S3）、L4 路径拒且不中断（S4）、L4 内容拒且不中断（S5）、未分类 fail-closed 中断（S6）、敏感读中断（S7）、根目录批量读保守中断（S8）、**L3 上给 edit 决定被 SDK 拒绝**（S9，决定集是强制执行的不是宣传）、审计三类齐全（S10）。

## 6. 边界（方案不承诺的）

- **权限是工具级**：直接 `backend.write()` 绕过全部三层（Task6 H12 已证）。方案保护的是“Agent 经工具出手”。
- **内容闸门不是 DLP**：只看 write/edit 的 content/new_string 参数，三条正则，漏报必然有；它是红线不是扫描器。
- **审计在进程内**：生产必须外送持久化 sink；中断状态随 checkpointer 生灭（MemorySaver 重启即失，跨会话排队要 SqliteSaver——列入待办）。
- **批量读的保守代价**：树里有 /secrets 时 `ls("/")` 永远中断。要把读保持在 L0，就限定 `path=/public` 或 `/drafts`。
- **只覆盖文件系统工具**：shell/网络类工具的分层是本方案的下一步，不是已完成项。
