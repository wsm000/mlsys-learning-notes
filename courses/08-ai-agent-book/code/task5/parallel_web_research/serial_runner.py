"""
串行 vs 并行的对照组
====================

要求：**同一批站点、同一抽取函数**，只有调度方式不同，才允许比较墙钟时间。
所以串行路径复用 WorkerAgent 本体（只用一个 worker，顺序跑），而不是另写一套逻辑——
否则"同一抽取函数"就只是口号。

串行语义与上游一致：**一旦找到就不再访问后面的站点**（这正是串行最吃亏的地方：
它必须从第一个站点开始一个个试）。为了让两边分母一致，结果里同时给出
"串行访问了几站 / 并行启动了几站"，以及"按胜出站点序号折算的解释性加速比"。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import llm_adapter
import message_bus as mb
from message_bus import MessageBus
from runtime import PageRuntime, Site
from worker import TaskState, WorkerAgent


async def run_sequential(
    sites: List[Site],
    query: str,
    runtime: PageRuntime,
    llm: llm_adapter.LLMClient,
    timeout: float = 60.0,
    criterion: Optional[str] = None,
    aliases: tuple = (),
    run_phase: str = "serial",
    observations: Optional[list] = None,
) -> Dict[str, Any]:
    observations = observations if observations is not None else []
    started = time.monotonic()
    results: List[Dict[str, Any]] = []
    winner_index = None
    stop_reason = "exhausted"
    for index, site in enumerate(sites):
        bus = MessageBus(verbose=False)
        worker = WorkerAgent(
            worker_id=f"serial-{index:02d}",
            site=site,
            bus=bus,
            runtime=runtime,
            llm=llm,
            query=query,
            timeout=timeout,
            criterion=criterion,
            aliases=aliases,
            run_phase=run_phase,
            observations=observations,
        )
        await bus.send("coordinator", worker.id, mb.TASK_ASSIGNED, {
            "target": query, "url": site.url, "task_id": worker.id,
        })
        site_started = time.monotonic()
        outcome = await asyncio.wait_for(worker.run(), timeout=timeout + 10)
        site_seconds = round(time.monotonic() - site_started, 3)
        results.append({
            "index": index,
            "site": site.name,
            "level": site.level,
            "state": outcome.state,
            "hit": outcome.hit,
            "seconds": site_seconds,
            "fetch_seconds": outcome.fetch_seconds,
            "llm_seconds": outcome.llm_seconds,
            "http_status": outcome.http_status,
            "text_chars": outcome.text_chars,
            "context_closed": outcome.context_closed,
        })
        if outcome.hit:
            winner_index = index
            stop_reason = "found"
            break
    total = round(time.monotonic() - started, 3)
    return {
        "outcome": "found" if winner_index is not None else "not_found",
        "sites_visited": len(results),
        "sites_configured": len(sites),
        "winner_index": winner_index,
        "stop_reason": stop_reason,
        "seconds": total,
        "results": results,
        "contexts": {
            "created": runtime.contexts_created,
            "closed": runtime.contexts_closed,
            "leaked": runtime.contexts_created - runtime.contexts_closed,
            "audit_ok": runtime.contexts_created == runtime.contexts_closed,
        },
    }
