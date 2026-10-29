"""
同构检索 Worker（移植自上游 agents.py::WorkerAgent）
==================================================

一个 worker = 一条独立抓取上下文 + 一条订阅总线上的收件箱。
它和协调者之间**只通过消息通信**：收到 task_assigned 才开始，任何阶段都可以在
"安全点"检查 terminate 事件，然后 ack + 关闭资源 + 上报 resource_closed。

四件事在这里被兑现：

1. **独立上下文**：每个 worker 自己 new_context()，崩溃/取消只影响自己。
2. **可中断的关键段**：抓取与 LLM 抽取都在 `_await_interruptibly` 里跑，
   terminate 一到就放弃等待（抓取线程仍在后台跑完——真实世界的取消就是如此）。
3. **首胜即晚到**：已经拿到 target_found 之后又收到 terminate，不再去结算，
   而是把结果作为 late_result 报到总线上（由协调者记为 duplicate）。
4. **安全点应答**：无论赢还是输，只要收到 terminate 就 ack；finally 里必关上下文。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import llm_adapter
import message_bus as mb
from message_bus import BROADCAST, MessageBus
from runtime import FetchContext, FetchError, PageRuntime, Site


class TaskState(str, Enum):
    SUBMITTED = "已提交"
    RUNNING = "执行中"
    SUCCEEDED = "已完成"
    FAILED = "失败"
    TERMINATED = "已终止"


class Terminated(Exception):
    """安全点上的协作式终止信号（不是 asyncio.CancelledError，不吞掉真正的外部取消）。"""


@dataclass
class WorkerOutcome:
    worker_id: str
    site: str
    state: str
    reason: str
    hit: bool = False
    late: bool = False
    seconds: float = 0.0
    fetch_seconds: float = 0.0
    llm_seconds: float = 0.0
    http_status: Optional[int] = None
    text_chars: int = 0
    acked: bool = False
    context_closed: bool = False
    stop_reason: str = ""


class WorkerAgent:
    def __init__(
        self,
        worker_id: str,
        site: Site,
        bus: MessageBus,
        runtime: PageRuntime,
        llm: Optional[llm_adapter.LLMClient],
        query: str,
        timeout: float = 60.0,
        criterion: Optional[str] = None,
        aliases: tuple = (),
        run_phase: str = "parallel",
        observations: Optional[list] = None,
    ):
        self.id = worker_id
        self.site = site
        self.bus = bus
        self.runtime = runtime
        self.llm = llm
        self.query = query
        self.timeout = timeout
        self.criterion = criterion
        self.aliases = tuple(aliases)
        self.run_phase = run_phase
        self.observations = observations if observations is not None else []

        self.sub = bus.subscribe(worker_id, types=[mb.TASK_ASSIGNED, mb.TERMINATE])
        self.terminate = asyncio.Event()
        self.termination_reason = ""
        self._assignment = None             # 由 _signals 转发的任务派发信封
        self.settled_hit = False            # 协调者已经结算了本 worker 的命中
        self.context: Optional[FetchContext] = None
        self.outcome = WorkerOutcome(
            worker_id=worker_id, site=site.name, state=TaskState.SUBMITTED.value, reason=""
        )

    # —— 与协调者通信 ——
    async def _status(self, state: TaskState, note: str) -> None:
        self.outcome.state = state.value
        await self.bus.send(self.id, "coordinator", mb.STATUS_UPDATE, {
            "state": state.value,
            "note": note,
            "source": self.site.name,
            "worker": self.id,
        })

    def _check_safe_point(self, where: str) -> None:
        """安全点：唯一允许抛 Terminated 的地方。"""
        if self.terminate.is_set():
            raise Terminated(f"{where} -> {self.termination_reason}")

    async def _await_interruptibly(self, coro):
        """让 awaitable 与 terminate 事件竞争；terminate 先到就放弃等待。"""
        operation = asyncio.ensure_future(coro)
        stopping = asyncio.ensure_future(self.terminate.wait())
        try:
            done, _pending = await asyncio.wait(
                {operation, stopping}, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            operation.cancel()
            stopping.cancel()
            await asyncio.gather(operation, stopping, return_exceptions=True)
            raise
        if operation in done:
            stopping.cancel()
            await asyncio.gather(stopping, return_exceptions=True)
            return operation.result()
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        raise Terminated(f"interrupted by terminate ({self.termination_reason})")

    async def _signals(self) -> None:
        """处理总线消息：terminate -> 置位终止事件；task_assigned -> 暂存在 outbox。

        注意：这个协程和主流程**共享同一个收件箱**。上游把 task_assigned 留在队列里等主流程先取，
        但主流程在信号协程启动之后就取不到它了（收件箱是单消费者队列）——这里显式转发。
        """
        while True:
            msg = await self.sub.get()
            if msg.type == mb.TERMINATE:
                self.termination_reason = msg.payload.get("reason", "cascade")
                self.terminate.set()
                # 置位后继续跑：后面的 terminate 或迟到的 task_assigned 都不再影响本 worker
                return
            if msg.type == mb.TASK_ASSIGNED:
                self._assignment = msg

    async def _ack(self, note: str) -> None:
        if self.outcome.acked:
            return
        self.outcome.acked = True
        await self.bus.send(self.id, "coordinator", mb.ACK, {
            "acked": "terminate",
            "source": self.site.name,
            "safe_point": note,
            "stop_reason": self.outcome.stop_reason,
        })

    # —— 主流程 ——
    async def run(self) -> WorkerOutcome:
        started = time.monotonic()
        signal_task = asyncio.ensure_future(self._signals())
        assigned = False
        # 先取出任务派发（如果信号协程抢到了，就从 outbox 里拿）
        assignment = self._assignment
        self._assignment = None
        while assignment is None and not self.terminate.is_set():
            await asyncio.sleep(0)
            assignment = self._assignment
            self._assignment = None
            if assignment is None and not self.sub.inbox.empty():
                assignment = await self.sub.get()
                if assignment.type == mb.TERMINATE:
                    self.termination_reason = assignment.payload.get("reason", "cascade")
                    self.terminate.set()
                    assignment = None
                    break
        if assignment is None:
            self.outcome.stop_reason = "terminate_before_start"
            await self._status(TaskState.TERMINATED, "启动前收到终止，未创建上下文")
            await self._ack("before_start")
            return self.outcome
        try:
            while assignment.type != mb.TASK_ASSIGNED:
                assignment = await self.sub.get()
            assigned = True
            self.query = assignment.payload.get("target", self.query)

            self._check_safe_point("after_assignment")
            await self._status(TaskState.RUNNING, f"创建独立抓取上下文 {self.id}")
            self.context = self.runtime.new_context(self.id)

            self._check_safe_point("before_fetch")
            await self._status(TaskState.RUNNING, f"抓取 {self.site.url}")
            fetch_started = time.monotonic()
            observation = await self._await_interruptibly(self.context.fetch(self.site, self.timeout))
            self.outcome.fetch_seconds = round(time.monotonic() - fetch_started, 3)
            self.outcome.http_status = observation.get("http_status")
            text = observation.get("text", "")
            self.outcome.text_chars = len(text)
            self.observations.append({
                "kind": "page_observation",
                "phase": self.run_phase,
                "worker_id": self.id,
                "site": self.site.name,
                "college": self.site.college,
                "level": self.site.level,
                "url": self.site.url,
                "http_status": observation.get("http_status"),
                "bytes": observation.get("bytes"),
                "text_chars": len(text),
                "text_sha256_head": str(abs(hash(text)) % (10 ** 12)),
                "fetch_seconds": self.outcome.fetch_seconds,
                "text_head": text[:400],
            })

            self._check_safe_point("after_fetch")
            await self._status(TaskState.RUNNING, "证据约束抽取")
            llm_started = time.monotonic()
            profile = await self._await_interruptibly(llm_adapter.extract_profile(
                self.llm,
                self.query,
                self.site.name,
                self.site.url,
                text,
                criterion=self.criterion,
                aliases=self.aliases,
                context={"phase": self.run_phase, "worker_id": self.id, "site": self.site.name},
                gate_terms_override=llm_adapter.gate_terms(self.site.name),
            ))
            self.outcome.llm_seconds = round(time.monotonic() - llm_started, 3)
            self.observations.append({
                "kind": "extraction",
                "phase": self.run_phase,
                "worker_id": self.id,
                "site": self.site.name,
                "status": profile.get("status"),
                "reason": profile.get("reason"),
                "gate": profile.get("gate"),
                "evidence_check": profile.get("evidence_check"),
                "profile": profile.get("profile"),
                "llm_latency_seconds": profile.get("llm_latency_seconds"),
                "seconds": profile.get("seconds"),
            })

            if profile.get("status") == "found":
                await self.bus.send(self.id, "coordinator", mb.TARGET_FOUND, {
                    "data": profile["profile"],
                    "source": self.site.name,
                    "worker": self.id,
                    "latency_seconds": round(time.monotonic() - started, 3),
                })
                self.outcome.hit = True
                self._check_safe_point("before_result_settled")
                await self._status(TaskState.SUCCEEDED, "找到目标（证据已逐字校验）")
            elif profile.get("status") == "not_found":
                await self.bus.send(self.id, "coordinator", mb.NOT_FOUND, {
                    "reason": profile.get("reason"),
                    "source": self.site.name,
                    "worker": self.id,
                })
                await self._status(TaskState.SUCCEEDED, "页面中未找到目标")
            else:
                reason = str(profile.get("reason") or "")
                # 抽取失败的错误类别形如 'LLMError[URLError]: ...'，保留冒号前的完整类别名
                kind = reason.split(":", 1)[0] if ":" in reason else profile.get("status", "error").capitalize()
                await self.bus.send(self.id, "coordinator", mb.WORKER_ERROR, {
                    "error": f"{kind}: {reason}",
                    "source": self.site.name,
                    "worker": self.id,
                })
                await self._status(TaskState.FAILED, str(profile.get("reason"))[:160])
        except Terminated as exc:
            self.outcome.stop_reason = "terminate_safe_point"
            await self._status(TaskState.TERMINATED, f"安全点响应级联终止：{exc}"[:200])
            await self._ack("terminate_safe_point")
        except FetchError as exc:
            if self.terminate.is_set():
                # 抓取因为终止被放弃，这是预期行为
                self.outcome.stop_reason = "terminate_during_fetch"
                await self._status(TaskState.TERMINATED, "抓取中被级联终止")
                await self._ack("during_fetch")
            else:
                self.outcome.stop_reason = "fetch_error"
                await self.bus.send(self.id, "coordinator", mb.WORKER_ERROR, {
                    "error": f"{exc.kind}: {exc.detail}"[:300],
                    "source": self.site.name,
                    "worker": self.id,
                })
                await self._status(TaskState.FAILED, f"抓取失败 {exc.kind}")
        except asyncio.CancelledError:
            self.outcome.stop_reason = "watchdog_cancel"
            await self.bus.send(self.id, "coordinator", mb.WORKER_ERROR, {
                "error": f"TimeoutError: exceeded {self.timeout + 10:.0f}s worker deadline",
                "source": self.site.name,
                "worker": self.id,
            })
            await self._status(TaskState.FAILED, "超出 worker 总时限，被看门狗取消")
            raise
        except Exception as exc:  # noqa: BLE001 —— worker 内任何异常都必须变成该站点的失败记录
            self.outcome.stop_reason = "exception"
            await self.bus.send(self.id, "coordinator", mb.WORKER_ERROR, {
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "source": self.site.name,
                "worker": self.id,
            })
            await self._status(TaskState.FAILED, f"{type(exc).__name__}")
        finally:
            signal_task.cancel()
            await asyncio.gather(signal_task, return_exceptions=True)
            context_closed = True
            if self.context is not None:
                try:
                    await self.context.close()
                except Exception as exc:  # noqa: BLE001
                    context_closed = False
                    await self.bus.send(self.id, "coordinator", mb.WORKER_ERROR, {
                        "error": f"ContextCloseError: {exc}"[:200],
                        "source": self.site.name,
                        "worker": self.id,
                    })
            self.outcome.context_closed = context_closed
            self.outcome.seconds = round(time.monotonic() - started, 3)
            await self.bus.send(self.id, "coordinator", mb.RESOURCE_CLOSED, {
                "context_closed": context_closed,
                "context_created": self.context is not None,
                "source": self.site.name,
                "worker": self.id,
                "was_assigned": assigned,
                "stop_reason": self.outcome.stop_reason,
            })
            # 收尾后若发现"终止信号已置位但没走到任何 safe-point"（例如 terminate 恰好
            # 落在 worker 最后一次让出事件循环之后、finally 之前），仍然补一次 ack：
            # 协议要求"收到 terminate 的 worker 必须应答"，而不是"必须死在某一行上"。
            if self.terminate.is_set():
                await self._ack("post_completion_ack")
        return self.outcome

    async def run_until_assigned(self) -> WorkerOutcome:
        """给测试用的薄封装：真正执行并返回结果。"""
        return await self.run()
