# 实验 10-4：N 路并行检索、消息总线、首个已验证成功结算与级联终止

> 对应第 10 章 10.4.4「管理者模式 · 并行协调形态」与实验 10-4。
> 代码：[`code/task5/parallel_web_research/`](../code/task5/parallel_web_research/README.md)
> 证据：[`evidence/task5/exp10-4/`](../evidence/task5/exp10-4/)
> 上游参考：[`_upstream/parallel-web-research/`](../_upstream/parallel-web-research/)（commit `dbc046e`）

## 1. 这次要证明什么，不证明什么

先划边界，避免把数字读多：

| 问题 | 本次能回答吗 | 依据 |
|---|---|---|
| 动态 N 路启动、独立上下文、状态表实时更新 | ✅ 能 | 15 路 parallel.json + 57 条 status_update |
| 首个已验证成功只结算一次（锁保护）、晚到命中不重复汇总 | ✅ 能（单测 + 真实运行） | `test_coordination.py` 5 个用例 + run1 的 `duplicate_hits` |
| 级联终止：恰好一次终止广播、未胜出 worker 安全点 ack | ✅ 能 | run1：1 次广播 / 2 个应 ack / 2 个实 ack / 0 缺失；复跑 cascade：1/3/3/0 |
| 失败隔离与超时不拖累同伴 | ✅ 能 | run1 中 1 个 HTTP500 + 1 个 MalformedPage，其余 13 路照常完成 |
| 资源关闭审计（上下文创建数 = 关闭数） | ✅ 能 | 三次运行全部 `contexts_leaked: 0` |
| **并行相对串行的加速比** | ⚠️ 只能给条件结论 | 真实网络 + 真实模型延迟的方差远大于调度收益，见 §4 |
| 浏览器抓取能力（Playwright 渲染） | ❌ 不证明 | 本机无 Playwright，改为 urllib 抓取 + 本地镜像回放 |

**适配说明（必须写清）**：上游用 Playwright + 真实大学网站；本机没有 Playwright，且多数外站 TLS 证书链有问题。
因此本实现用 `urllib` + `html.parser` 抽取"可见文本"，页面内容来自**一次真实抓取**后落地的本地镜像
（`corpus/`，含 Stanford 各院系页面与同名陷阱页），并在实验运行时用 `http.server` 起本地站点。
这证明的是**协调机制**，不是抓取能力。真实 HTTPS 的传输/TLS 代价单独标定在 `live_tls_calibration.json` 中，不与机制结论混用。

## 2. 架构：一个 Manager、一条总线、N 个同构 worker

```text
                    ┌──────────────── Coordinator（Manager）────────────────┐
                    │ 状态表 status_table   ·   asyncio.Lock 单次结算         │
                    │ duplicate/late_result  ·  恰好一次 terminate 广播        │
                    │ ack 对账（expected_loser_acks）· 资源关闭审计            │
                    └───────────────▲──────────────────┬────────────────────┘
            status_update/result/ack│                  │ task_assigned / terminate
                    ┌───────────────┴──────────────────▼────────────────────┐
                    │            MessageBus（信封：seq / ts / sender / target / type / payload）│
                    └───────────────┬──────────────────┬────────────────────┘
                    ┌───────────────▼──┐  ┌────────────▼─────┐        ┌──────────────┐
                    │ WorkerAgent-00   │  │ WorkerAgent-01   │  ...   │ WorkerAgent-14│
                    │ 独立抓取上下文    │  │ 独立抓取上下文    │        │ 独立抓取上下文 │
                    │ 证据约束抽取+门禁 │  │ 证据约束抽取+门禁 │        │ …             │
                    └──────────────────┘  └──────────────────┘        └──────────────┘
```

消息类型（全部落在同一个信封里，可落盘成 `events.jsonl` 重放）：
`task_assigned / status_update / target_found / not_found / worker_error / terminate / ack / resource_closed / late_result`。

**关键设计三点**（与上游的差异及原因，README 有完整列表）：

1. **terminate 只发给"仍在跑且还没有自己结论"的 worker**。上游把 terminate 广播给所有人，已结束的 worker
   不会回 ack，于是对账里会出现"缺失 ack"的假阳性。本实现先算出 `expected_loser_acks`，再定向投递；
   只有当收件人集合为空时才退回广播语义。
2. **晚到命中必须可见**。上游的 `target_found` 在结论已定后静默丢弃；本实现锁内记 `late_result` 并计入
   `duplicate_hits`，让"只结算一次"变成可观测事实而不是"看起来没发生"。
3. **门禁分两级**：字面命中直接放行（确定性、可复现）；字面不命中但页面主题命中的，仍调用模型，由**逐字证据**
   （evidence 必须能在渲染文本中原样找到）裁决。没有这一步，"姓名不在页面上"会被模型凭参数记忆编造成"找到了"。

## 3. 真实运行：run1 与级联复跑

### 3.1 run1（`exp10-4-local-run1`，查询「Provost of Stanford University」）

| 指标 | 值 |
|---|---|
| 启动 worker 数 | 15 |
| 并行墙钟 | 11.687 s |
| 串行墙钟（同一批站点、同一抽取函数，命中即停） | 11.796 s |
| 结果 | found：**Jenny Martinez**，evidence 逐字命中 "Jenny Martinez is Stanford's fourteenth provost." |
| 胜出 worker | agent-09 |
| terminate 广播次数 | **1** |
| 应 ack / 实 ack / 缺失 | 2 / 2 / **0**（agent-01、agent-13，均在 `terminate_safe_point` 应答） |
| 失败隔离 | agent-12 HTTP500、agent-14 MalformedPage，其余 13 路不受影响 |
| 未找到 | 10 路（确定性门禁直接判 not_found） |
| status_update 总数 | 57 |
| 上下文审计 | 并行 15 创建 / 15 关闭 / 泄漏 0；串行 2 / 2 / 0 |
| duplicate_hits | 0（本次无晚到命中，路径由单测覆盖） |

### 3.2 级联复跑（`lead-cascade-check2`，由 Lead 在修复后重跑）

| 指标 | 值 |
|---|---|
| worker 数 | 5（同一目标页的多个查询 URL，制造近乎同时命中） |
| 结果 | found（agent-02 胜出，17.860 s） |
| terminate 广播 | **1** |
| 应 ack / 实 ack / 缺失 | 3 / 3 / **0** |
| 上下文审计 | 5 / 5 / 泄漏 0 |
| 未胜出的 agent-00、agent-01、agent-03 | 状态 **已终止**，stop_reason `terminate_safe_point` |

> `smoke-cascade-4` 是更早的一次级联冒烟（1 次广播 / 3 应 ack / 3 实 ack / 4 上下文全关闭），结论一致。

### 3.3 修复记录：一个真实的"指标口径"bug

Lead 复核时发现：runner 交付前最后一次改动后，`test_cascade_terminates_exactly_once_and_losers_ack` 失败
（`assert 3 == 1`）。查证后确认**不是并发 bug**：

- `terminate` 的实现是"每个收件人一条信封"，所以总线上出现 3 条 TERMINATE 信封是**正确行为**；
- `terminate_broadcasts`（结算触发了几次级联）始终是 1；
- 出问题的是 `terminate_event_count` 这一派生指标——它直接数了信封个数，把"1 次级联"记成了"3 次终止"。

修复：把这个指标改成**按结算原因去重**统计（`len({payload.reason} ...)`），并在注释里写明它回答的是
"触发了几次级联"，不是"发了几条消息"。修完后 16 个用例全绿（`pytest_offline.txt`，Lead 亲自复跑）。

这件事本身值得记一笔：**多 Agent 系统里最容易被误读的不是机制，而是派生指标**。"广播了 1 次"和"信道上出现
了 3 条终止消息"是两个都要能分开回答的问题，混成一个字段，就会在回归测试里制造假失败、在生产里制造假告警。

## 4. 加速比为什么只有 1.009×：一个必须解释的负结果

run1 实测并行 11.687 s、串行 11.796 s，加速比 **1.009×**。这不是"并行没用"，而是这个条件下**测不出调度收益**：

1. **串行提前终止**：串行路径命中即停，本次第 2 个站点（provost-office）就命中，串行只访问了 2/15 个站点，
   总共也只需 ~11.8 s。而并行必须等"最快命中者"完成，本次是 agent-09。
2. **单次模型调用延迟主导且方差大**：页面抓取只要 0.016–0.031 s（本地镜像），而一次证据约束抽取要 5.5–6.3 s，
   且抖动用秒计。并行只把"抓取"并行掉了，抽取延迟是并行的——所以墙钟≈最慢的那个有效调用。
3. **并行不是免费的**：15 路同时打同一个模型端点，排队与重试会吃掉一部分收益。

**结论边界**：在本机这套"本地镜像 + 远程模型"的配置下，10-4 的加速比**不是可靠的观测量**。
要测调度本身，必须把延迟来源换成可控的固定值——所以补了一个**纯调度桩对照**（`schedule_control.json`）：
把抓取/抽取替换成固定 0.5 s + 1.0 s 的桩，8 站点并行 1.515 s vs 串行 12.125 s（**8.003×**，接近理论上限 8×）；
4 站点且胜者在第 2 位时并行 1.516 s vs 串行 3.031 s（**1.999×**，与"串行只需访问 2 个站点"的预期一致）。

> 两个数字必须分开读：`8.003×` 是**并发调度正确性的证据**（不是真实世界的加速承诺）；
> `1.009×` 是**真实条件下测不到收益的证据**。把前者说成后者，就是在用可控桩冒充生产性能。


## 4b. 真实站点直连（L1）：并行反而更慢（2026-10-26 追加）

上游用 Playwright + 真实大学网站。本机没有 Playwright，于是加了一条 `--level live_unverified` 路径：
worker 用 `urllib` 直接访问真实 HTTPS 站点（TLS 校验关闭——本机代理的证书链过不了校验，这一点必须写明）。
跑了两次，8 个真实站点，查询仍是「Provost of Stanford University」：

| 运行 | 并行墙钟 | 串行墙钟 | 串行访问站点数 | 结果 | 失败隔离 | 上下文审计 |
|---|---|---|---|---|---|---|
| exp10-4-live-L1 | 69.985 s | 41.968 s | 2 | not_found | HTTP403 x1、Timeout x3 | 8/8 关闭，泄漏 0 |
| exp10-4-live-L1b | 130.000 s | 13.375 s | 2 | not_found | HTTP403 x1、Timeout x3 | 8/8 关闭，泄漏 0 |

**读数与解释**：

1. **加速比是 0.6x 和 0.103x，即并行比串行更慢。** 原因是这一批真实站点要么 403、要么超时（4/8 失败），
   串行在第 2 个站点就拿到结果并停止；并行要等所有 8 路跑完（含 3 路超时），墙钟被最慢的失败者决定。
   **这正是 10.4.4 里"级联终止"要解决的问题，但本次没有胜出者，所以终止没有发生。**
2. **机制层依然全部通过**：8 个上下文全部关闭、泄漏 0；4 个失败被隔离并分类，没有拖垮其他 worker。
3. **这一层不能用来论证抓取能力**：TLS 校验被关闭、真实站点有 403/超时，它证明的是协调机制在真实网络抖动下不出错，
   不是"能抓到真实数据"。

**把三层放在一起看**（这是本次对 10-4 最完整的回答）：

| 层次 | 站点来源 | 延迟来源 | 并行 vs 串行 | 证明了什么 |
|---|---|---|---|---|
| 本地镜像 | 一次真实抓取后回放 | 远程模型抽取 | 1.009x | 机制（结算/终止/ack/审计）在可控条件下成立 |
| 纯调度桩 | 固定延迟桩 | 无模型 | 8.003x / 1.999x | 调度器本身正确，接近理论上限 |
| 真实站点 L1 | 真实 HTTPS | 真实网络 + 模型 | 0.6x / 0.103x | 机制在真实抖动下仍成立，但**并行不保证更快** |

**结论**：并行化的收益取决于"被并行的那段是不是瓶颈"。本机的瓶颈是远程模型调用（5-6 s 且方差大），
抓取只占 0.02 s，所以把抓取并行化省不下时间；一旦真实网络里出现超时和 403，并行还要等最慢的失败者，反而更慢。
**这是对"并行一定更快"最直接的反例，也是 10-4 这类实验最容易被误读的地方。**
## 5. 单元测试覆盖（16 个用例，离线可跑）

```powershell
cd courses/08-ai-agent-book/code/task5/parallel_web_research
.venv-task3\Scripts\python.exe -m pytest tests -q    # 16 passed
```

覆盖的机制：

| 机制 | 用例 |
|---|---|
| 并发结算只认第一个 | `test_concurrent_settle_only_first_wins` |
| 晚到命中记 duplicate，不重复汇总 | `test_late_hit_after_settlement_is_duplicate_not_resummary` |
| 级联恰好一次 + 三个败者 ack + 状态为已终止 | `test_cascade_terminates_exactly_once_and_losers_ack` |
| 无人命中时不得广播 terminate | `test_no_hits_means_no_terminate_broadcast` |
| 结算前已结束的 worker 不应被计入 ack 对账 | `test_worker_already_finished_before_settlement_is_not_expected_to_ack` |
| 失败隔离与分类（HTTP500 / TLS） | `test_failed_sites_are_isolated_and_classified` 等 |

## 6. 与第 10 章正文的对应

| 正文概念 | 本实现中的位置 |
|---|---|
| 10.4.2 一、消息传递：统一信封让链路可追溯 | `message_bus.py`（seq + ts 单调，可落盘重放） |
| 10.4.2 二、状态查询：别用拉取接口 | 全部走 `status_update` 推送 + Manager 状态表 |
| 10.4.2 三、执行终止：优雅优先、级联向下 | `terminate` → 安全点 → `ack` → 关闭上下文 → `resource_closed` |
| 10.4.2 四、资源与调度：并发上限、超时 | 每站点 timeout + worker 看门狗 |
| 10.4.4 并行协调：第一个**已验证**成功 | `settle()` 只在锁下生效，且此前必须过逐字证据门禁 |
| 10.5.1 并发冲突 | `duplicate_hits` / `late_result` 让竞争可见 |

## 7. 未完成与遗留

- **没有跑成完整的 `--mode full` 正式运行**：尝试过一次 15 路并行 + 串行 + 级联的合并运行，跑到 60+ 分钟仍未收尾
  （远程模型端点偶发 502，重试退避把尾延迟拉长）。最终交付用的是已完成且可复核的 run1 + 两次级联 + 桩对照 +
  单元测试，没有为了"看起来完整"而伪造一次全量运行。
- **Playwright 路径未验证**：本地镜像替代真实浏览器渲染，"渲染截图作为审核证据"这条在本次没有复现。
- `schedule_control` 的第二次对照组（4 站点）出现 `missing_loser_acks: ["agent-03"]`：桩运行里胜出
  worker 结算后，另一个 worker 已自行结束未收到 terminate，属对账口径问题（与 §3.3 同类），已保留原始记录不再修饰。
## 8. 追加：L1 真实网络等级（`--level live_unverified`）

L0（本地镜像）能控制变量，但回答不了"换回真实网络还成立吗"。因此新增 L1 等级：worker 直接访问真实 HTTPS 站点。

| 指标 | L1（60s 超时） | L1b（120s 超时） |
|---|---|---|
| 站点 / 并行 | 8 路 | 8 路 |
| 并行 / 串行墙钟 | 69.985 s / 41.968 s | 130.0 s / 13.375 s |
| 加速比 | 0.600× | 0.103× |
| 结果 | not_found | not_found |
| 上下文创建/关闭 | 8/8，泄漏 0 | 8/8，泄漏 0 |
| 失败 | HTTP403×1 + Timeout×3 | HTTP403×1 + Timeout×3 |

**结论**：机制层（状态表、失败隔离、资源关闭）在真实网络下**没有退化**；退化的是**吞吐与成功率**。

- 串行只访问 2 个站点（13.4 s）就结束，并行必须等 8 路全部落地（含重试退避），墙钟被最坏路径主导
  ——管理者模式在真实条件下的典型形态：**并行省平均延迟，赔最坏延迟**。
- L0 里永远出不来的失败出现了：HTTP403 反爬、真实链路超时、JS 渲染页只抽到 12 字符骨架。
- **本轮没有任何一路命中，因此 L1 下的级联终止"未被触发"**，不能据此说 L1 的终止机制已验证。

复现：

```powershell
cd courses/08-ai-agent-book/code/task5/parallel_web_research
python run_experiment.py --mode full --no-cascade --level live_unverified --timeout 120 --run-id exp10-4-live-L1b
```

> L1 关闭了 TLS 校验（本机代理证书链无法通过校验），因此它证明的是"协调机制在真实网络延迟下仍然成立"，
> 不证明"抓取能力等价于浏览器渲染"。
