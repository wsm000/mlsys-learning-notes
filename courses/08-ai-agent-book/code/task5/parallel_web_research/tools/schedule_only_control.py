"""
纯调度桩对照（**机制验证**，不是真实测量）
==========================================

为什么需要它：本地真实运行里并行的墙钟被**单次 LLM 抽取延迟方差**主导（同一次运行里从 4 秒到 60 秒不等），
而串行又在第 2 个站点就命中并提前停下，于是"串行/并行"的比值落在 1.0 附近——这个数字既不能证明并行有效，也不能证伪。

这个脚本把"抓取 + 抽取"替换成**固定延迟桩**（都是真实的 asyncio 等待，不是假并行），只保留协调机制本身，
用来回答一个被剥离干净的问题：**在同样的每站点耗时下，并行调度相对串行调度能省多少墙钟？**

注意：它不测网络、不测模型、不测页面内容；它证明的是"调度器确实并发了 N 路"，而不是"在真实网络上能快多少"。

用法：python tools/schedule_only_control.py --output ../../../evidence/task5/exp10-4/schedule_control.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import message_bus as mb  # noqa: E402
from coordinator import Coordinator  # noqa: E402
from message_bus import MessageBus  # noqa: E402
from runtime import FetchContext, PageRuntime, Site  # noqa: E402
from serial_runner import run_sequential  # noqa: E402
from worker import WorkerAgent  # noqa: E402


class StubContext(FetchContext):
    """抓取桩：固定延迟后返回一段固定正文。"""

    async def fetch(self, site, timeout):
        await asyncio.sleep(site.latency_seconds)
        if site.fail == "llm_error":
            # 让抽取桩抛错：用一段特殊正文告诉桩"这一步要失败"
            return {"url": site.url, "http_status": 200, "bytes": 8,
                    "text": "LLM_ERROR", "seconds": site.latency_seconds, "level": "stub"}
        return {
            "url": site.url,
            "http_status": 200,
            "bytes": 32,
            "text": "STUB PAGE TEXT " + site.name,
            "seconds": site.latency_seconds,
            "level": "stub",
        }


class StubRuntime(PageRuntime):
    context_class = StubContext

    def __init__(self):
        super().__init__(corpus_dir=ROOT / "corpus", label="stub")


class StubExtractor:
    """与 llm_adapter.extract_profile 同签名：固定抽取耗时 + 按站点名判命中。"""

    def __init__(self, delay: float):
        self.delay = delay

    async def __call__(self, client=None, query=None, site_name="", url="", text="", **kwargs):
        # worker 以 llm_adapter.extract_profile(llm, query, site_name, url, text, ...) 的位置参数调用，
        # 这里用同样的位置顺序，llm 自己（self）已被 __call__ 的绑定吃掉。
        await asyncio.sleep(self.delay)
        if isinstance(text, str) and text.startswith("LLM_ERROR"):
            raise RuntimeError("LLMError: stub provider outage")
        if "--hit" in site_name:
            return {
                "found": True,
                "status": "found",
                "reason": "stub hit",
                "profile": {"name": "STUB", "evidence": "STUB PAGE TEXT " + site_name},
            }
        return {"found": False, "status": "not_found", "reason": "stub not found", "profile": None}


def make_sites(n: int, fetch_latency: float, windex: int, broken: tuple = ()):
    """n 个站点：抓取固定耗时；第 windex 个是命中页，broken 里的下标让抽取失败。"""
    sites = []
    for i in range(n):
        name = "s%02d--hit" % i if i == windex else "s%02d" % i
        sites.append(Site(
            name=name,
            college="Stub",
            url="http://stub/" + name,
            level="stub",
            expected="found" if i == windex else "not_found",
            latency_seconds=fetch_latency,
            fail="llm_error" if i in broken else None,
        ))
    return sites


class _ExtractorPatch:
    """把 worker 模块里的抽取函数换成桩，退出时还原。"""

    def __init__(self, extractor):
        self.extractor = extractor

    def __enter__(self):
        import worker
        self._module = worker
        self._original = worker.llm_adapter.extract_profile
        worker.llm_adapter.extract_profile = self.extractor
        return self

    def __exit__(self, *exc):
        self._module.llm_adapter.extract_profile = self._original
        return False


async def run_parallel(sites, query, criterion, extractor, timeout):
    mb.reset_sequence()
    runtime = StubRuntime()
    bus = MessageBus(verbose=False)
    coordinator = Coordinator(bus, query, runtime, criterion=criterion)
    for index, site in enumerate(sites):
        coordinator.add_worker(WorkerAgent(
            worker_id="agent-%02d" % index,
            site=site,
            bus=bus,
            runtime=runtime,
            llm=extractor,
            query=query,
            timeout=timeout,
            criterion=criterion,
            run_phase="stub_parallel",
        ))
    with _ExtractorPatch(extractor):
        result = await coordinator.run()
    result["runtime_audit"] = await runtime.shutdown()
    return result


async def run_serial(sites, query, criterion, extractor, timeout):
    runtime = StubRuntime()
    with _ExtractorPatch(extractor):
        return await run_sequential(
            sites, query, runtime, None, timeout=timeout,
            criterion=criterion, run_phase="stub_serial",
        )


async def main_async(n, fetch_latency, extract_delay, windex, broken=()):
    sites = make_sites(n, fetch_latency, windex, broken)
    query, criterion = "Stub Target", "Stub Target"
    extractor = StubExtractor(delay=extract_delay)
    par = await run_parallel(sites, query, criterion, extractor, timeout=30)
    ser = await run_serial(sites, query, criterion, extractor, timeout=30)
    speedup = None
    if par["parallel_seconds"]:
        speedup = round(ser["seconds"] / par["parallel_seconds"], 3)
    return {
        "kind": "schedule_only_control",
        "disclaimer": "固定延迟桩：只验证调度并发，不测网络与模型",
        "params": {
            "sites": n,
            "fetch_latency_seconds": fetch_latency,
            "extract_delay_seconds": extract_delay,
            "winner_index": windex,
            "per_site_serial_cost": round(fetch_latency + extract_delay, 3),
        },
        "parallel_seconds": par["parallel_seconds"],
        "serial_seconds": ser["seconds"],
        "measured_speedup_serial_over_parallel": speedup,
        "parallel_winner": par["winner"],
        "serial_winner_index": ser["winner_index"],
        "serial_sites_visited": ser["sites_visited"],
        "terminate_broadcasts": par["terminate_broadcasts"],
        "failure_summary": par["failure_summary"],
        "expected_loser_acks": par["expected_loser_acks"],
        "ack_delays_after_terminate": {
            wid: round(row["ts"] - (par["terminate_ts"] or 0.0), 3)
            for wid, row in par["acks"].items()
        },
        "ack_count": len(par["acks"]),
        "missing_loser_acks": par["missing_loser_acks"],
        "context_audit_parallel": par["runtime_audit"],
        "context_audit_serial": ser["contexts"],
        "status_updates_total": par["status_updates_total"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--fetch-latency", type=float, default=0.5)
    parser.add_argument("--extract-delay", type=float, default=1.0)
    parser.add_argument("--winner-index", type=int, default=7)
    parser.add_argument("--broken", default="", help="逗号分隔的站点下标：这些站点的抽取会失败")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    broken = tuple(int(x) for x in args.broken.split(",") if x.strip())
    result = asyncio.run(main_async(args.sites, args.fetch_latency, args.extract_delay,
                                    args.winner_index, broken))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        rows = json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            rows = [rows]
    rows.append(result)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
