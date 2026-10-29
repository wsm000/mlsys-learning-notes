"""
进程内异步消息总线（Message Bus）
================================

实验 10-4 的通信底座，移植自上游 chapter10/parallel-web-research/message_bus.py，
但把"只在终端打印"换成了**可落盘、可断言的事件流**。

语义：
- 每条消息是一个 Envelope 信封：sender_id / target / type / payload / seq / ts。
- seq 全局单调递增（跨所有发送方），ts 是相对总线启动的单调秒数，
  两者一起构成"可重放的因果顺序"：任意两条线上的事件都能排出先后。
- target 为 BROADCAST 时投递给所有订阅者（按上游语义，不回投给发送者自己）。
- 每个订阅者一个 asyncio.Queue 收件箱，投递即入队，不做业务判断。
- history 保存全部信封，run_experiment 把它写成 events.jsonl 证据。

设计取舍（与上游一致）：
- await asyncio.sleep(0) 让出事件循环，使订阅者在同一步内就能拿到消息，
  这样"先广播 terminate、后检查 terminate 事件"的顺序在协程调度上是确定的。
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

BROADCAST = "*"

_seq_counter = itertools.count(1)
_START_TIME = time.monotonic()

# 消息类型常量：集中声明，避免各模块拼写漂移
TASK_ASSIGNED = "task_assigned"
STATUS_UPDATE = "status_update"
TARGET_FOUND = "target_found"
NOT_FOUND = "not_found"
WORKER_ERROR = "worker_error"
TERMINATE = "terminate"
ACK = "ack"
RESOURCE_CLOSED = "resource_closed"
LATE_RESULT = "late_result"


def reset_sequence() -> None:
    """测试用：重置全局序号（保证每次运行的 events.jsonl 从 #1 开始）。"""
    global _seq_counter, _START_TIME
    _seq_counter = itertools.count(1)
    _START_TIME = time.monotonic()


def now() -> float:
    return time.monotonic() - _START_TIME


@dataclass
class Envelope:
    sender_id: str
    target: str
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    seq: int = field(default_factory=lambda: next(_seq_counter))
    ts: float = field(default_factory=now)

    def short(self) -> str:
        tgt = "ALL" if self.target == BROADCAST else self.target
        try:
            body = json.dumps(self.payload, ensure_ascii=False, default=str)
        except Exception:
            body = str(self.payload)
        if len(body) > 90:
            body = body[:87] + "..."
        return (
            f"[t={self.ts:6.2f}s #{self.seq:<3}] {self.sender_id:>10} -> {tgt:<10} | "
            f"{self.type:<15} | {body}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": round(self.ts, 6),
            "sender_id": self.sender_id,
            "target": self.target,
            "type": self.type,
            "payload": self.payload,
        }


class Subscription:
    def __init__(self, owner_id: str, types: Optional[Iterable[str]] = None):
        self.owner_id = owner_id
        self.types = set(types) if types is not None else None
        self.inbox: "asyncio.Queue[Envelope]" = asyncio.Queue()

    def accepts(self, env: Envelope) -> bool:
        return self.types is None or env.type in self.types

    async def get(self) -> Envelope:
        return await self.inbox.get()

    def drain_nowait(self) -> List[Envelope]:
        out = []
        while not self.inbox.empty():
            out.append(self.inbox.get_nowait())
        return out

    async def get_with_timeout(self, timeout: float) -> Optional[Envelope]:
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class MessageBus:
    def __init__(self, verbose: bool = False):
        self._subs: Dict[str, List[Subscription]] = {}
        self.verbose = verbose
        self.history: List[Envelope] = []

    def subscribe(self, owner_id: str, types: Optional[Iterable[str]] = None) -> Subscription:
        sub = Subscription(owner_id, types)
        self._subs.setdefault(owner_id, []).append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        lst = self._subs.get(sub.owner_id, [])
        if sub in lst:
            lst.remove(sub)

    def subscriber_ids(self) -> List[str]:
        return sorted(self._subs)

    async def publish(self, env: Envelope) -> int:
        self.history.append(env)
        if self.verbose:
            print("  BUS " + env.short())
        delivered = 0
        for owner_id, sub_list in self._subs.items():
            if env.target != BROADCAST and owner_id != env.target:
                continue
            if env.target == BROADCAST and owner_id == env.sender_id:
                continue
            for sub in sub_list:
                if sub.accepts(env):
                    sub.inbox.put_nowait(env)
                    delivered += 1
        await asyncio.sleep(0)
        return delivered

    async def send(
        self, sender_id: str, target: str, type: str, payload: Optional[Dict[str, Any]] = None
    ) -> Envelope:
        env = Envelope(sender_id=sender_id, target=target, type=type, payload=payload or {})
        await self.publish(env)
        return env

    # —— 事件统计便捷方法 ——
    def events_of_type(self, type: str) -> List[Envelope]:
        return [e for e in self.history if e.type == type]

    def count(self, type: str) -> int:
        return len(self.events_of_type(type))

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in self.history)
