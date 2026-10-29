"""
协调者（移植自上游 agents.py::Coordinator，修掉了它的两个竞态）
=============================================================

它负责：动态 N 路启动、维护状态表、错误隔离、**加锁的首次命中结算**、
恰好一次 terminate 广播、未胜出 worker 的 ack 对账、资源关闭审计。

与上游的差异（都是"真实并发才会暴露"的问题）：

1. **terminate 不发给"已经出结论"的 worker。**
   上游在结算时一次性固定 expected_loser_acks，但 terminate 是广播，会同时送达
   已经发出 not_found / worker_error 的 worker——它们不回 ack（它们已经跑完了），
   于是对账里会凭空多出一个"缺失 ack"。

2. **worker 收到 terminate 后仍可能把结果发上来（late_result）。**
   上游只把它当普通 target_found 处理，结果被静默丢弃，没有任何记录；
   本实现在锁内额外记一条 late_result，并在汇总里给出 duplicate_hits 与
   "首个命中 vs 晚到命中"的时序差。这样"只结算一次"是**可观测**的，而不是靠假设。

3. **结算前先看目标站点是否已经结束。**
   not_found / worker_error 先于 target_found 到达时，该 worker 不再欠 ack。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import message_bus as mb
from message_bus import BROADCAST, MessageBus
from runtime import PageRuntime
from worker import TaskState, WorkerAgent


@dataclass
class TaskRecord:
    worker_id: str
    source_name: str
    state: TaskState = TaskState.SUBMITTED
    note: str = ""
    updates: int = 0
    first_seen: float = field(default_factory=time.monotonic)
    last_update: float = field(default_factory=time.monotonic)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker": self.worker_id,
            "source": self.source_name,
            "state": self.state.value,
            "note": self.note,
            "updates": self.updates,
            "seconds_since_start": round(self.last_update - self.first_seen, 3),
            "history": self.history,
        }


class Coordinator:
    def __init__(self, bus: MessageBus, query: str, runtime: PageRuntime, criterion: Optional[str] = None):
        self.bus = bus
        self.query = query
        self.runtime = runtime
        self.criterion = criterion
        self.sub = bus.subscribe("coordinator", types=None)
        self.workers: List[WorkerAgent] = []
        self.table: Dict[str, TaskRecord] = {}

        self._lock = asyncio.Lock()
        self.settled = False
        self.winner: Optional[str] = None
        self.profile: Optional[Dict[str, Any]] = None
        self.settle_seq: Optional[int] = None
        self.settle_ts: Optional[float] = None
        self.duplicate_hits: List[Dict[str, Any]] = []
        self.terminate_broadcast_count = 0
        self.terminate_recipients: List[str] = []

        self.acks: Dict[str, Dict[str, Any]] = {}
        self.expected_loser_acks: set = set()
        self.excluded_from_ack: Dict[str, str] = {}
        self.errors: Dict[str, str] = {}
        self.not_found: Dict[str, str] = {}
        self.closed: set = set()
        self.live_workers: set = set()      # 已报到、尚未关闭上下文的 worker
        self.resource_failures: Dict[str, str] = {}
        self.timeline: List[Dict[str, Any]] = []

    # —— 注册 ——
    def add_worker(self, worker: WorkerAgent) -> None:
        self.workers.append(worker)
        self.table[worker.id] = TaskRecord(worker.id, worker.site.name)

    # —— 首次命中结算（整个实验最关键的一把锁）——
    async def settle(self, worker_id: str, profile: Dict[str, Any], latency: Optional[float] = None) -> str:
        async with self._lock:
            if self.settled:
                self.duplicate_hits.append({
                    "worker_id": worker_id,
                    "latency_seconds": latency,
                    "seconds_after_winner": None if self.settle_ts is None else
                        round(mb.now() - self.settle_ts, 3),
                })
                await self.bus.send("coordinator", BROADCAST, mb.LATE_RESULT, {
                    "worker": worker_id,
                    "winner": self.winner,
                    "note": "hit arrived after settlement; recorded as duplicate, not re-summarized",
                })
                return "duplicate"
            self.settled = True
            self.winner = worker_id
            self.profile = profile
            self.settle_ts = mb.now()
            self.settle_seq = (self.bus.history[-1].seq if self.bus.history else 0)
            # 只有"仍在跑、还没自己出结论"的 worker 才需要 ack
            candidates = []
            for worker in self.workers:
                wid = worker.id
                if wid == worker_id:
                    continue
                if wid in self.not_found:
                    self.excluded_from_ack[wid] = "already_not_found"
                    continue
                if wid in self.errors:
                    self.excluded_from_ack[wid] = "already_failed"
                    continue
                if wid in self.closed:
                    self.excluded_from_ack[wid] = "already_closed"
                    continue
                candidates.append(wid)
            self.expected_loser_acks = set(candidates)
            self.terminate_recipients = list(candidates)
            self.terminate_broadcast_count += 1
            # 有明确收件人时只发给这些人：广播给"已经结束"的 worker 没有意义，
            # 而且会让"应 ack / 实 ack"的对账失去意义。收件人集合为空时才用广播语义。
            target = BROADCAST if not candidates else None
            if target is not None:
                await self.bus.send("coordinator", BROADCAST, mb.TERMINATE, {
                    "reason": f"target_found_by_{worker_id}",
                    "winner": worker_id,
                    "recipients_expect_ack": candidates,
                })
            else:
                for wid in candidates:
                    await self.bus.send("coordinator", wid, mb.TERMINATE, {
                        "reason": f"target_found_by_{worker_id}",
                        "winner": worker_id,
                        "recipients_expect_ack": candidates,
                    })
            return "winner"

    def closed_mode_drained(self) -> bool:
        """所有 worker 都上报过 resource_closed，且收件箱为空。"""
        return len(self.closed) >= len(self.workers) and self.sub.inbox.empty()

    # —— 事件循环 ——
    async def _apply(self, env: mb.Envelope) -> None:
        rec = self.table.get(env.sender_id)
        if env.type == mb.STATUS_UPDATE:
            if rec:
                rec.state = TaskState(env.payload["state"])
                rec.note = env.payload.get("note", "")
                rec.updates += 1
                rec.last_update = time.monotonic()
                rec.history.append({
                    "ts": round(env.ts, 3),
                    "state": rec.state.value,
                    "note": rec.note,
                })
                self.timeline.append({"ts": round(env.ts, 3), "kind": "status",
                                      "worker": env.sender_id, "note": rec.note,
                                      "state": rec.state.value})
        elif env.type == mb.TARGET_FOUND:
            verdict = await self.settle(
                env.sender_id, env.payload.get("data") or {}, env.payload.get("latency_seconds")
            )
            if verdict == "winner":
                self.timeline.append({"ts": round(env.ts, 3), "kind": "winner",
                                      "worker": env.sender_id})
        elif env.type == mb.ACK:
            self.acks[env.sender_id] = {
                "safe_point": env.payload.get("safe_point"),
                "stop_reason": env.payload.get("stop_reason"),
                "ts": round(env.ts, 3),
            }
            self.timeline.append({"ts": round(env.ts, 3), "kind": "ack", "worker": env.sender_id})
        elif env.type == mb.WORKER_ERROR:
            self.errors[env.sender_id] = env.payload.get("error", "unknown")
        elif env.type == mb.NOT_FOUND:
            self.not_found[env.sender_id] = env.payload.get("reason", "not found")
        elif env.type == mb.RESOURCE_CLOSED:
            self.closed.add(env.sender_id)
            self.live_workers.discard(env.sender_id)
            if not env.payload.get("context_closed", False):
                self.resource_failures[env.sender_id] = "context did not close"
            self.timeline.append({"ts": round(env.ts, 3), "kind": "resource_closed",
                                  "worker": env.sender_id})

    async def run(self, watchdog_extra: float = 10.0) -> Dict[str, Any]:
        started = time.monotonic()
        for worker in self.workers:
            self.live_workers.add(worker.id)
            await self.bus.send("coordinator", worker.id, mb.TASK_ASSIGNED, {
                "target": self.query,
                "url": worker.site.url,
                "task_id": worker.id,
            })
        tasks = [
            asyncio.ensure_future(asyncio.wait_for(w.run(), timeout=w.timeout + watchdog_extra))
            for w in self.workers
        ]
        hard_deadline = time.monotonic() + max(
            (w.timeout + watchdog_extra for w in self.workers), default=1.0
        ) + 5.0
        while True:
            # 终止条件：所有 worker 都已关闭上下文，**并且**收件箱已经排空。
            # （只按 closed 计数退出会漏掉排在 resource_closed 之后的事件——真实并发下
            #   terminate 广播、ack、resource_closed 的相对顺序并不由 closed 计数决定。）
            drained = self.closed_mode_drained()
            if len(self.closed) >= len(self.workers) and drained:
                break
            if time.monotonic() > hard_deadline:
                break
            env = await self.sub.get_with_timeout(0.2)
            if env is None:
                if all(t.done() for t in tasks) and self.sub.inbox.empty():
                    break
                continue
            await self._apply(env)
        # 收尾：把 worker 结束后仍在队列里的事件全部应用（ack 可能晚于 resource_closed）
        while not self.sub.inbox.empty():
            await self._apply(self.sub.inbox.get_nowait())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for worker, res in zip(self.workers, results):
            if isinstance(res, BaseException) and worker.id not in self.errors:
                self.errors[worker.id] = f"{type(res).__name__}: {res}"[:200]

        failure_types: Dict[str, int] = {}
        for error in self.errors.values():
            kind = error.split(":", 1)[0].strip()
            failure_types[kind] = failure_types.get(kind, 0) + 1
        missing_acks = sorted(self.expected_loser_acks - set(self.acks))
        unexpected_acks = sorted(set(self.acks) - self.expected_loser_acks)
        winners_terminate_ts = next(
            (e.ts for e in self.bus.history if e.type == mb.TERMINATE), None
        )
        return {
            "query": self.query,
            "outcome": "found" if self.winner else "not_found",
            "winner": self.winner,
            "profile": self.profile,
            "workers_launched": len(self.workers),
            "duplicate_hits": self.duplicate_hits,
            "late_results": self.bus.count(mb.LATE_RESULT),
            "terminate_broadcasts": self.terminate_broadcast_count,
            # 一次"级联终止"可能在总线上表现为多条同事件信封（每个收件人一条）。
            # 这里要断言的是"结算触发了几次级联"，不是"信封个数"。
            "terminate_event_count": len({e.payload.get("reason") for e in self.bus.events_of_type(mb.TERMINATE)}),
            "terminate_ts": None if winners_terminate_ts is None else round(winners_terminate_ts, 3),
            "terminate_recipients": self.terminate_recipients,
            "expected_loser_acks": sorted(self.expected_loser_acks),
            "acks": self.acks,
            "missing_loser_acks": missing_acks,
            "unexpected_acks": unexpected_acks,
            "excluded_from_ack": self.excluded_from_ack,
            "errors": self.errors,
            "failure_summary": {"count": len(self.errors), "by_type": failure_types},
            "not_found": self.not_found,
            "not_found_count": len(self.not_found),
            "status_table": {wid: rec.to_dict() for wid, rec in self.table.items()},
            "status_updates_total": sum(r.updates for r in self.table.values()),
            "resource_closed_reports": len(self.closed),
            "resource_failures": self.resource_failures,
            "contexts": {
                "created": self.runtime.contexts_created,
                "closed": self.runtime.contexts_closed,
                "leaked": self.runtime.contexts_created - self.runtime.contexts_closed,
                "audit_ok": self.runtime.contexts_created == self.runtime.contexts_closed,
            },
            "bus_events": len(self.bus.history),
            "parallel_seconds": round(time.monotonic() - started, 3),
            "timeline": self.timeline,
        }
