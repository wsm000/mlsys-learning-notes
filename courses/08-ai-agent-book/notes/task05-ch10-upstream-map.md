# 第 10 章（多 Agent 协作）上游配套源码 · 机制对照与诚实边界

> 本文只回答三件事：**上游到底有什么项目**、**每个项目的核心机制落在哪个文件哪一行**、**它证明了什么、没证明什么**。
> 不重复 task05-ch10-multi-agent.md 的概念叙述；本文是"源码取证"性质的附录。

---

## 0. 取回信息（可复核）

| 项 | 值 |
| --- | --- |
| 仓库 | `bojieli/ai-agent-book`（《深入理解 AI Agent》） |
| 分支 | `main` |
| **commit sha** | `dbc046eb896ac4e39aa19c7774c8bf49583b89a6` |
| commit 时间 | 2026-09-30T03:03:28Z（"Update star history chart [skip ci]"） |
| 校验方式 | `git ls-remote https://github.com/bojieli/ai-agent-book.git refs/heads/main` 与 `git log -1` 双次一致 |
| 本地快照 | `courses/08-ai-agent-book/_upstream/<项目名>/` |
| 快照范围 | **只含源码 / README / 测试 / 示例配置**；已排除 `validation/`、`runs/`、图片、音频、`__pycache__` |
| 未取回 | 各项目的 `validation/**` 大批量产物（多 MB 的 campaign.json、judge_receipts、`runs/`）——本文凡引用其结果，均标注为"**读 README 自述，未独立复算**" |

**取回手段说明（诚实交代）**：本机 PowerShell 的 `Invoke-WebRequest`/`curl.exe` 因 schannel 无凭证失败；Python `urllib` 直连 `raw.githubusercontent.com` 在连续 ~200 次请求后被对端 reset（WinError 10054）。最终采用 `git clone --filter=blob:none --depth 1` + `git -c http.sslBackend=openssl` + sparse-checkout（`git ls-remote` 本身可走 schannel，故 sha 校验不受影响）。**没有关闭证书校验**——只是把 TLS 后端从 Windows schannel 换成 git 自带的 openssl。

---

## 1. 上游的项目组织方式（先纠正一个可能的误解）

任务/教材示意里常见的链路 **Planner → Researcher → Executor → Reviewer** 在上游 `chapter10/` **并不存在**。上游实际是**八个并列的独立子项目**，每个子项目自带 `README.md`、入口脚本、`requirements.txt`/`env.example` 和自己的 `tests/`：

```
chapter10/
├── README.md                     # 项目索引（下表来源）
├── multi-role-transfer/          # 共享上下文：角色切换 vs Skill
├── book-translation/             # 不共享上下文：管理者模式
├── autonomous-phone-registration/# 不共享上下文：Computer + Phone 双 Agent（自主启动）
├── talkact-reproduction/         # 同 10-3 的固定拓扑基线（仅 README + 校验脚本）
├── parallel-web-research/        # 不共享上下文：并行 worker + 级联终止
├── generative-agents/            # 斯坦福 AI 小镇复现指南（未取回，见 §7）
├── voice-werewolf/               # 法官状态机 + 信息权限
└── staged-system-prompt/         # 已归档：阶段化系统提示词（不属于当前编号实验）
```

「Planner/Researcher/Executor/Reviewer」式的**四段流水线**最接近的是 `book-translation`（Manager / Glossary / Translation / Proofreading），但它是**管理者编排 + 文件系统数据平面**，不是线性交接链；而 `multi-role-transfer` 里的 triage→research→data_analysis→writing 是**模型自主决策**的移交，顺序不由代码写死。两者都不是通用教程模板。

---

## 2. 总表：项目 → 实验号 → 核心机制 → 关键文件:行

| 项目 | 实验号 | 是否有正文对应 | 核心机制（一句话） | 关键文件:行 |
| --- | :--: | :--: | --- | --- |
| multi-role-transfer | 10-1 | ✅ 正文 p275–276 | 共享一条 `history`，只换 system prompt + 工具集 | `orchestrator.py:83-93, 250-252` |
| multi-role-transfer（Skill 臂） | 10-1 | ✅ | 静态前缀 + `load_skill` 把 SKILL.md 作为 tool result 追加 | `skill_orchestrator.py:165-171, 244-254` |
| book-translation | 10-2 | ✅ 正文 p285–287 | 四角色独立上下文，Manager 只存文件索引 | `agents.py:481-495, 560-576, 430-455` |
| book-translation（对照组） | 10-2 | ✅ | 单 Agent 一条不断增长的对话 | `agents.py:714-740` |
| autonomous-phone-registration | 10-3 | ✅ 正文 p287–289 | 模型在 `tool_choice=auto` 下自主 spawn Phone Agent + 异步总线 | `decision.py:165`, `orchestration.py:376-392, 395-425` |
| talkact-reproduction | 10-3（固定拓扑基线） | ⚠️ 无独立实验号 | 只有 README + `validate_campaign.py`，**无运行时代码** | `README.md`（机制不可从源码验证） |
| parallel-web-research | 10-4 | ✅ 正文 p289–292 | 并行 worker + 锁保护的"首个成功"结算 + 级联 terminate + ack | `agents.py:240-257, 266-290` |
| generative-agents | 10-5 | ✅ 正文 p292–299 | **未取回**（📖 类型：外部仓库复现指南） | — |
| voice-werewolf | 10-6 | ✅ 正文 p299–300 | 代码法官 + 三个投递原语实现信息权限 | `game.py:111-127, 154-181, 247-277` |
| staged-system-prompt | 无（已归档） | ❌ 已从第十章实验序列删除 | 阶段化 system prompt + 阶段化工具集 + 回退 | `agent.py:30-71, 306, 444-499` |

> 表格最后一列的行号均以本地快照文件为准，可用 `grep -n` 复核。

---

## 3. multi-role-transfer（实验 10-1）：共享上下文里的两种"换角色"

### 3.1 路径一：换掉 system prompt + 工具集

- 共享历史与 system prompt **物理分离**：`self.history` 不含 system 消息（`orchestrator.py:69`），每次调用前现拼：
  ```python
  # orchestrator.py:90-93
  def _messages_for_api(self):
      system_msg = {"role": "system", "content": ROLES[self.current_role].system_prompt}
      return [system_msg] + self.history
  ```
- 工具可见性同样按角色收窄：`_tools_for_current_role()` = 角色专属工具 + `transfer_to_agent`（`orchestrator.py:83-88`）。
- 移交是**工具调用**：模型调 `transfer_to_agent(target_role, reason)`，编排器只置一个 `pending_transfer`，等本轮所有 tool 消息都回填完之后才真正切换（`orchestrator.py:193-214` 与 `:245-252`）。这一"延迟切换"是 OpenAI tool_call/role 协议强制要求，不是随意选择。
- 拒绝自我移交：`target == self.current_role` 时返回"移交失败：你已经是 X 角色"（`orchestrator.py:196-202`）。
- 防死循环：同一 `(role, tool, arguments)` 签名第 3 次出现时追加系统提示（`orchestrator.py:232-238`），外加 `max_steps=20` 硬上限（`orchestrator.py:263-272`）。

### 3.2 路径二：Skill（静态前缀）

- 工具全集**故意固定**，注释写明了理由："changing tools at a role boundary would have the same prefix-cache consequence as changing the system prompt"（`skill_orchestrator.py:165-168`）。
- system prompt 也是固定字符串 + Skill 目录（`skill_orchestrator.py:97-120, 170-171`）。
- 角色选择靠 `load_skill(name)` 返回 SKILL.md 全文，作为 tool result 进入轨迹（`skill_orchestrator.py:218-243`）。
- **"看得到 ≠ 获准执行"**：工具 schema 全部可见，但 Harness 层用 `SKILL_TOOLS` 白名单硬拦（`skill_orchestrator.py:31-37, 244-254`）；第一步不加载 triage 也直接被拒（`skill_orchestrator.py:223-227`）。

### 3.3 证明了什么 / 没证明什么

**证明（代码层面确定成立）**
- 两条路径确实共享同一份用户/助手/tool 轨迹，差别只在"角色规程放哪"。
- 路径一在**结构上**让越界工具不可调用；路径二必须额外加白名单门，否则纯靠模型自觉。
- 两条路径都有真实的工具实现（`tools.py`），不是 mock。

**没证明**
- ❌ 没有证明 Skill 更省 token。README 自述的 v2 结果里，Skill 的未缓存输入 token **中位数反而 +6,855**、延迟 +4.368s（`README.md:640`）。
- ❌ 没有证明 Skill 质量更高。同处自述：确定性门禁 Skill 15/30 vs Transfer 2/30，但这是**某个模型某个配置下的有界结论**，README 自己就用 "bounded architecture results, not model-independent superiority claims" 限定（`README.md:640`）。
- ❌ 没有证明前缀缓存真的被复用。README 明确要求以服务商 `cached_tokens` 为准，前缀 hash 只是机制代理（`README.md:195-198, 714-716`）。
- ⚠️ **未验证（本轮）**：`validation/comparison/runs/exp10-1-qwen35flash-20260809-v2/` 的 campaign.json / judge.json 我**没有取回**，上面两个数字全部是 README 转述，没有独立复算。
- ⚠️ 本次快照**无法离线跑通**：`tests/` 只覆盖工具分发等单元行为；端到端对比需要 `OPENAI_API_KEY` + `TAVILY_API_KEY`。

---

## 4. book-translation（实验 10-2）：管理者模式的上下文账本

### 4.1 机制落点

| 机制 | 证据 |
| --- | --- |
| Manager 上下文里**不含正文**，只含 task/guide/plan/call_log/file_index/progress | `agents.py:481-488` |
| Manager 上下文体积用 tiktoken 实测（不靠估算） | `snapshot_manager()` `agents.py:491-495`，调用点 `:510, 508, 615, 650` |
| 译文只落盘，Manager 只记路径 | `agents.py:566-576`（写文件 + `file_index[name] = out_file`） |
| Glossary 读全书、产出后释放 | `agents.py:334-363`（独立 messages，只有 system+user 两条） |
| Translation 只看到"本章 + 术语表 + 指南" | `agents.py:374-393` |
| Proofreading 看到全部译文（唯一一个读全量的角色） | `agents.py:396-427`（`joined` 拼接所有译文） |
| Manager 决策只收"文件索引 + 报告摘要" | `agents.py:430-455` + 摘要裁剪 `:609-613` |
| 上下文隔离是**结构性**的，不是提示词要求 | `llm_chat()` 每次 `messages` 从零构造（`agents.py:242-248` 注释） |
| 编辑部指定术语强制覆盖模型结果 | `EDITORIAL_MANDATE` `agents.py:303-308`，强制写入 `:532-539` |

### 4.2 对照组是真正的对照组

`run_single_agent()`（`agents.py:668-741`）刻意构造"一条不断增长的对话"：每章译文以 `assistant` 消息**留在 messages 里**（`agents.py:733-734`，注释直接写"这正是上下文膨胀的来源"）。所以两臂的 token 差是**架构差**，不是提示词差。

### 4.3 证明了什么 / 没证明什么

**证明**
- Manager 上下文规模与书的长度**解耦**：它只随"章节数"增加几行记录，不随每章正文字数增长——这是由代码结构保证的（`agents.py:481-495`），不依赖运行结果。
- 同一份术语表确实被下发给每个 Translation 实例（`agents.py:374-381`），而单 Agent **完全没有术语表通道**（`agents.py:683-688` 只有 system + 逐章 user）。

**没证明（且 README 自己承认了负结果）**
- ❌ **多 Agent 并不全面更优**。README 自述 v4：Manager 上下文缩小 20.43×、翻译 token 减少 6.48×、匿名质量 4.654 > 4.481，但**墙钟慢 6.57%**（`README.md:121`）。
- ❌ **共享术语表没有换来通用术语一致性**：编辑部指定译法遵从率 75% vs 0%，但确定性字符串术语一致率反而 **50% vs 87.5%**（`README.md:119`）；且 `embedding` 没遵守指定译法"嵌入向量"（`README.md:121`）。
- ❌ **结构保真出现真实负结果**：两组都改动了代码 payload 并增加标题；管理者组还改动了插图目标（`README.md:123`）。
- ⚠️ **未验证（本轮）**：`validation/real_20260730T061500Z_v4/evidence.json` 未取回，上述数字为 README 转述。
- ⚠️ 4 章 `sample_book/` 的 demo 结果（`README.md:189-196`）规模太小，README 自己说"只用于教学预演"（`README.md:136`）。
- ⚠️ **本次快照无法复跑**：需要 ARK/OpenAI key；`tests/` 只用打桩 LLM 测空值与非法结构（`tests/test_null_glossary.py` 等）。

---

## 5. voice-werewolf（实验 10-6）：信息权限是"投递原语"问题

### 5.1 机制落点（这是全文最硬的一组断言）

法官**不是 LLM**，是 `Judge` 类（`werewolf/game.py:76-77`，类注释即"法官：编排 + 信息权限控制"）。信息权限不靠提示词"请你保密"，而靠**只有三个投递原语**，每个原语在投递的同时写审计日志：

| 原语 | 语义 | 行号 |
| --- | --- | --- |
| `broadcast()` | 进**所有玩家**（含已出局）上下文 | `game.py:111-115` |
| `private_send()` | 只进**指定单人**上下文 | `game.py:117-120` |
| `wolves_send()` | 只进**所有狼人**上下文 | `game.py:122-127` |

玩家侧对应的是**私有 memory 列表**（`werewolf/agent.py:114-115`，注释："别的 Agent 无法访问"），拼进 prompt 时只拼自己的（`agent.py:134-138, 140-149`）。

具体投放：
- 身份分配：每人只知自己的身份（`game.py:161-162`）；狼人额外收到队友名单（`game.py:164-166`，注释"这是信息不对称的关键"）。
- 预言家查验结果**只进预言家本人**（`game.py:264-266`）。
- 女巫得知今晚被刀者**只进女巫**（`game.py:288-289`）。
- 狼人夜间共识**只进狼人**（`game.py:239-240`）。

审计：`AuditLog` 每条记录带 `visible_to` 列表（`werewolf/audit.py:16-29`），收尾打印整张"谁看到了什么"表（`audit.py:31-43`）。**这是把"信息隔离"从口号变成可检查产物的关键设计。**

真人/模拟用户路径还有额外防泄漏：终端上帝视角身份表在真人局被隐藏（`game.py:167-175`），夜间私密日志只在允许者面前打印（`game.py:145-149`）。

### 5.2 模拟用户为什么"不能作弊"

`SimulatedUserPlayerAgent` 继承 `PlayerAgent`，因此**只拿到本席 memory**（`simulator.py:291-296`）；它被强制走工具调用（`tool_choice` 指定唯一工具，`simulator.py:328-340`），然后发言/选人必须经过 **TTS 合成 → ASR 转写**的真实音频回环（`simulator.py:271-273, 419`）。选人动作的转写解析结果与工具选择**不一致时直接抛错并留下 `simulator_action_mismatch` 证据**（`simulator.py:420-432`）——fail-closed，不是"猜一个"。

### 5.3 证明了什么 / 没证明什么

**证明（代码层面确定）**
- 信息隔离是**架构性质**：好人的 `memory` 里根本不存在"谁是狼人"这条数据，除非预言家验过（`agent.py:236-246` 的 `_known_wolves()` 只从自己 memory 里正则挖掘）。
- 信息隔离**可审计**：每条投放都留 `visible_to`。
- 语音通道**不能绕过**信息边界：模拟用户的"发言"必须过 audio，选人必须过 ASR 校验。

**没证明**
- ❌ 没有证明真人体验可用。README 明说自动模拟器的成功不能替代真人 VAD/打断体验（`README.md:100`）。
- ❌ 没有证明"AI 会玩"是稳定的。README 保留了三代失败：v1 把 `P1 is not` 误当弃权被严格复核否决、v3 两次含糊弃权被拒且狼人编造公开事件、直到 v11 才在同一局内全部通过（`README.md:98, 203-205`）。**这些失败被保留而不是删除，是这份材料可信度最高的地方。**
- ⚠️ **未验证（本轮）**：v11 的 `acceptance_report.json`（49KB）与 `game.log` **未取回**；"3 个完整回合、6 次音回环、13 个响应 ID、1650 音频 token"全部是 README 转述。
- ⚠️ **本次快照无法复跑**：需要 OpenRouter/OpenAI key + 本机 TTS + ASR；`--offline` 是确定性规则策略（`agent.py:110, 248-299`），只能验证**状态机与信息隔离**，不能验证 LLM 行为。

---

## 6. 另外三个项目（不在"必做四件套"里，但同属 chapter10）

### 6.1 autonomous-phone-registration（实验 10-3 主路径）

- **自主边界在哪**：`decision.py` 的注释写死"there is intentionally no Python `if len(fields)` decision"，`tool_choice="auto"`（`decision.py:82-87, 165`）。模型吐出 `initiate_phone_call_agent` 才创建 Phone Agent；没调用就抛错，**不允许代码"提前"创建**（`orchestration.py:384-388`）。
- **两个 Agent 怎么通信**：进程内 `MessageBus`，点对点队列 + 单调序号 + 单调时钟时间戳（`bus.py:15-58`）；敏感字段在日志与落盘 trace 中被 `<redacted>` 替换（`bus.py:44-53, 60-73`）。
- **并发与失败传播**：`asyncio.wait(FIRST_EXCEPTION)` + 显式取消未完成任务 + 关闭通道（`orchestration.py:395-425`）；浏览器填写失败会作为 `fill_error` 转给 Phone Agent，避免"页面没填上却对用户说注册成功"（`orchestration.py:294-304, 306-367`）。
- **边界**：证明的是"技术链路跑通"，README 明确"不等同于真人可用性或跨 NAT/TURN 测试"（`README.md:23`）。**未验证**：raw-v4 manifest 与 9/9 门禁均为 README 转述，本快照需要 Playwright + 真麦克风/WebRTC 才能跑。

### 6.2 parallel-web-research（实验 10-4）

- 这是 chapter10 里**唯一把"级联终止"写成可验证竞态**的项目：
  ```python
  # agents.py:240-257
  async def _settle(self, worker_id, profile):
      async with self._lock:
          if self._settled:
              self.duplicate_hits.append(worker_id); return
          self._settled, self.winner, self.profile = True, worker_id, profile
          self.expected_loser_acks = {...}                    # 只有"当时还在跑"的 worker 欠 ack
          await self.bus.send("coordinator", BROADCAST, "terminate", {...})
  ```
- 结算只允许一次（`_lock` + `_settled` 双保险），晚到的命中被记进 `duplicate_hits` 而不是覆盖赢家。
- 终态还聚合 `missing_loser_acks`、`resource_failures`、`contexts_closed`（`agents.py:295-323`），把"取消是否被确认""浏览器上下文是否真的关了"也纳入验收。
- 消息总线是仿 Redis Pub/Sub 的进程内实现（`message_bus.py:1-17, 92-140`），广播不回收给发送者（`message_bus.py:119-121`）。
- **边界**：证明的是"并发协调机制正确"，README 明说并行"不会自动解决同名对象、冲突资料或来源质量问题"（`README.md:13`）。**未验证**：1.872× 加速与 12/12 门禁为 README 转述。本项目已在本快照中（含源码），但同样需要 Playwright + 真实模型 key 才能跑。

### 6.3 staged-system-prompt（**已归档，不属于当前实验序列**）

- README 第 3 行与第 15-17 行明确：**"它已不属于当前正文的编号实验"**、"本实验已从当前第十章实验序列中删除；历史验收材料继续保留 `10-1` 标识"。**注意这个 `10-1` 与 multi-role-transfer 的 10-1 撞号，是历史遗留，不要混为一谈。**
- 机制：三阶段 `STAGE_PROMPTS`（`agent.py:30-64`）+ 三套工具集（`agent.py:67-71` + `tools.py:238-329`）；每次请求 `[system(当前阶段)] + history`（`agent.py:306`）；转换由"信号工具"触发（`agent.py:73-79, 369-385`）。
- 关键细节（容易被误读）：**阶段切换被推迟到本轮所有 tool 消息回填之后**（`agent.py:357-386` 的注释与 `pending_transition`）；回退用 `request_revision`，且有 `max_revisions` 安全阀（`agent.py:475-494`）。
- **边界**：它是"阶段化提示词"的完整可运行样例，但没有正文实验号，也**不能**用来支撑任何正文里的实验结论。

---

## 7. 未取回 / 未验证清单（不要当成已验证）

| 项 | 状态 | 原因 |
| --- | --- | --- |
| `chapter10/generative-agents/`（实验 10-5） | **未取回** | 索引标注为 📖"复现指南"，依赖需自行 clone 的外部仓库；本任务范围外 |
| 各项目 `validation/**`、`runs/**` | **未取回** | 体积大（单 campaign.json 5.8MB），且属于产物而非源码 |
| 所有"X/Y 门禁通过""加速 1.872×""上下文缩小 20.43×"等数字 | **未独立复算** | 均为各 README 转述；要复算必须取回对应 evidence/manifest 并联网 |
| 端到端运行（任何项目） | **未运行** | 本机无 `playwright`、无 OpenAI/ARK/Moonshot/OpenRouter/Tavily key；TTS/ASR 依赖本机 `say`/`espeak`/Whisper |
| 离线单元测试（`pytest`） | **未运行** | 快照缺 `pyproject.toml`/`uv.lock` 与部分 dev 依赖；`tests/` 原设计在仓库根环境跑。**不声称它们通过或失败。** |
| `talkact-reproduction/` | **只有文档** | 该目录下仅 `README.md` + `validate_campaign.py`，**没有运行时代码**；其"16/16 局、17/17 门禁"无法从源码验证 |

---

## 8. 一句话结论（给写作/复盘用）

1. 上游 chapter10 是**八个独立项目**，没有通用 Planner/Executor 教程目录；"角色切换 vs Skill""管理者模式""法官式信息权限"三件事分别由三个不同项目承载，机制各不相同。
2. 三个最值得引用的**结构性**结论（不依赖跑分，能指到行）：
   - 共享上下文换角色 = 换 system prompt + 工具集，history 不动（`orchestrator.py:90-93, 250-252`）；
   - 管理者模式的上下文可控性来自"正文只落盘、Manager 只存索引"（`agents.py:481-495, 566-576`）；
   - 信息权限要靠**投递原语 + 审计**，不能靠提示词自觉（`game.py:111-127` + `audit.py:16-43`）。
3. 上游最诚实的一点：**它自己保留并公布了负结果**——Skill 不比 Transfer 便宜、管理者模式比单 Agent 慢 6.57%、术语一致率反而更低、语音狼人杀三代验收失败。引用时务必带上这些限定。
