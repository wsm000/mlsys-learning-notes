# 实验 10-4：N 路并行检索 + 消息总线 + 首个已验证成功结算 + 级联终止（本机离线版）

本目录是把 `courses/08-ai-agent-book/_upstream/parallel-web-research/`（上游用 Playwright + 真实大学网站）
移植到**本机可离线复现**的版本。

## 30 秒跑起来

```powershell
cd courses/08-ai-agent-book/code/task5/parallel_web_research

# 1) 协调机制的全部单元测试（不需要网络、不需要模型；16 个用例）
C:\Users\86150\Documents\ChatGPT\cs336复习\hello-gpu\.venv-task3\Scripts\python.exe -m pytest tests -q

# 2) 完整实验：并行 15 路 / 串行基线 / 级联竞争，写证据到 evidence/task5/exp10-4/<run-id>/
python run_experiment.py --mode full --run-id exp10-4-local-v1

# 3) 只跑级联竞争（约 20 秒）
python run_experiment.py --mode cascade --run-id exp10-4-cascade-v1
```

模型端点默认 `http://127.0.0.1:8789/v1/responses`（model `gpt-5.6-sol`），可用 `--llm-endpoint/--model` 覆盖。

## 代码地图

| 文件 | 作用 | 上游对应 |
|---|---|---|
| `message_bus.py` | 信封总线：seq/ts、点对点 + 广播、可落盘事件流 | `message_bus.py` |
| `runtime.py` | 抓取运行时：`PageRuntime`/`FetchContext` 代替 BrowserPool/context；本地镜像 `http.server`；四种抓取等级 | `BrowserPool` + Playwright |
| `worker.py` | 同构 worker：独立上下文、安全点终止、ack、资源关闭上报 | `agents.py::WorkerAgent` |
| `coordinator.py` | 状态表、`asyncio.Lock` 单次结算、duplicate、恰好一次 terminate、审计 | `agents.py::Coordinator` |
| `serial_runner.py` | 串行基线（复用同一个 WorkerAgent，保证"同一抽取函数"） | `run_sequential` |
| `llm_adapter.py` | Responses API 客户端 + 两级确定性门禁 + 逐字证据校验 + 重试退避 | `llm.py` |
| `sites.py` | 站点/语料配置与真值标注 | `sources.py` |
| `run_experiment.py` | 主入口：并行 → 串行 → 级联，落盘 events/summary/observations/receipts | `demo.py` + `run_official_experiment.py` |
| `corpus/` | 一次真实抓取落地的页面（Stanford 各院系 + 同名陷阱页） | 真实网站 |
| `tools/` | 语料抓取脚本、真实 HTTPS/TLS 代价标定脚本 | — |

## 与上游的实现差异（都是为了能离线复现 + 修掉真实并发下才暴露的竞态）

1. **没有 Playwright**：用 `urllib` + `html.parser` 抓"可见文本"，页面内容是一次真实抓取后本地回放。
2. **terminate 广播的收件人**：只发给"仍在跑且还没自己出结论"的 worker；上游把 terminate 广播给所有人，
   已结束的 worker 不回 ack，对账里会出现不存在的"缺失 ack"。
3. **晚到的命中会被记录**：上游的 `target_found` 在结论已定后静默丢弃；本实现锁内记 `late_result`，
   汇总里给出 `duplicate_hits`，让"只结算一次"可观测。
4. **门禁分两级**：字面命中直接放行；字面不命中但页面主题词命中时，仍调用模型并由逐字证据裁决，
   避免"字面匹配漏召回"。

详细结论与实测数字见 `../../notes/task05-ch10-parallel-orchestration.md`。
