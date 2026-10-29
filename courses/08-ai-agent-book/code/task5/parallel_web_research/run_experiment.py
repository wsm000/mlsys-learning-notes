"""
实验 10-4 主入口：并行 N 路检索 + 消息总线 + 首个成功结算 + 级联终止 + 串并行对比
================================================================================

用法（工作目录 = 本目录，即 code/task5/parallel_web_research）：

    python run_experiment.py --mode full --run-id exp10-4-local-v1
    python run_experiment.py --mode cascade --run-id exp10-4-cascade-v1   # 只跑级联/重复命中
    python run_experiment.py --mode calibration --source live_unverified   # 真实 HTTPS 的 TLS 惩罚

产出（默认写 ../../../evidence/task5/exp10-4/<run-id>/）：

  events.jsonl      总线事件流（seq/ts/sender/target/type/payload 全字段）
  summary.json      可重算笔记里所有数字的汇总（加速比、ack、资源审计、失败统计）
  parallel.json     并行原始结果（含状态表、时间线）
  serial.json       串行原始结果（同一批站点、同一抽取函数）
  observations.jsonl 每次页面抓取 + 每次抽取的原始观测
  llm_receipts.jsonl 每次 LLM 调用/失败的凭据（response_id、usage、latency）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import llm_adapter
import message_bus as mb
import sites as site_config
from coordinator import Coordinator
from message_bus import MessageBus
from runtime import MirrorServer, PageRuntime, Site
from serial_runner import run_sequential
from worker import WorkerAgent

HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_ROOT = HERE.parent.parent.parent / "evidence" / "task5" / "exp10-4"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_mirror_for(sites: List[Site]) -> MirrorServer:
    """按站点列表起本地镜像；半合成站点复用真实语料文件，并注入延迟/故障。"""
    routes: Dict[str, Dict[str, object]] = {}
    for site in sites:
        routes[site.name] = {
            "file": site.name.split("--")[0],
            "delay": site.latency_seconds,
            "fail": site.fail,
        }
    return MirrorServer(site_config.CORPUS_DIR, routes)


def build_cascade_sites(sites: List[Site]) -> List[Site]:
    """从主站点列表里派生一组"近同时命中"的竞争站点（正文都是真实抓取的页面）。"""
    by_name = {s.name: s for s in sites}
    base = by_name["stanford-provost-office"]
    trap = by_name["stanford-profiles-andrew-ng"]
    negative = by_name["stanford-cs-faculty"]

    def clone(src: Site, name: str, latency: float) -> Site:
        return Site(name=name, college=src.college, url=src.url, level="local_mirror",
                    expected=src.expected, latency_seconds=latency, fail=None)

    return [
        clone(base, "stanford-provost-office--fast-a", 0.0),
        clone(base, "stanford-provost-office--fast-b", 0.4),
        clone(base, "stanford-provost-office--slow-hit", 6.0),
        clone(trap, "stanford-profiles-andrew-ng--trap", 0.2),
        clone(negative, "stanford-cs-faculty--negative", 0.0),
    ]


def attach_urls(sites: List[Site], server: MirrorServer) -> None:
    for site in sites:
        site.url = server.url_for(site.name)


async def run_parallel_phase(
    sites: List[Site],
    query: str,
    criterion: str,
    timeout: float,
    phase: str,
    llm: llm_adapter.LLMClient,
    observations: List[Dict[str, Any]],
    receipts: List[Dict[str, Any]],
    verbose: bool = False,
) -> Dict[str, Any]:
    mb.reset_sequence()
    runtime = PageRuntime(corpus_dir=site_config.CORPUS_DIR, label=f"parallel-{phase}")
    bus = MessageBus(verbose=verbose)
    coordinator = Coordinator(bus, query, runtime, criterion=criterion)
    for index, site in enumerate(sites):
        coordinator.add_worker(WorkerAgent(
            worker_id=f"agent-{index:02d}",
            site=site,
            bus=bus,
            runtime=runtime,
            llm=llm,
            query=query,
            timeout=timeout,
            criterion=criterion,
            run_phase=phase,
            observations=observations,
        ))
    result = await coordinator.run()
    audit = await runtime.shutdown()
    result["runtime_audit"] = audit
    result["phase"] = phase
    result["bus_events_list"] = [env.to_dict() for env in bus.history]
    return result


async def run_serial_phase(
    sites: List[Site],
    query: str,
    criterion: str,
    timeout: float,
    llm: llm_adapter.LLMClient,
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    mb.reset_sequence()
    runtime = PageRuntime(corpus_dir=site_config.CORPUS_DIR, label="serial")
    result = await run_sequential(
        sites, query, runtime, llm, timeout=timeout, criterion=criterion,
        run_phase="serial", observations=observations,
    )
    result["runtime_audit"] = await runtime.shutdown()
    return result


def summarize(bundle: Dict[str, Any]) -> Dict[str, Any]:
    parallel = bundle.get("parallel")
    serial = bundle.get("serial")
    cascade = bundle.get("cascade")
    if parallel is None:
        return {
            "query": bundle["query"],
            "mode": "cascade_only",
            "cascade": {
                "winner": cascade["winner"],
                "outcome": cascade["outcome"],
                "duplicate_hits": cascade["duplicate_hits"],
                "terminate_broadcasts": cascade["terminate_broadcasts"],
                "terminate_ts": cascade["terminate_ts"],
                "acks": sorted(cascade["acks"]),
                "expected_loser_acks": cascade["expected_loser_acks"],
                "missing_loser_acks": cascade["missing_loser_acks"],
                "context_audit": cascade["runtime_audit"],
                "parallel_seconds": cascade["parallel_seconds"],
                "status_table": cascade["status_table"],
                "failure_summary": cascade["failure_summary"],
            } if cascade else None,
        }
    speedup = None
    if serial and parallel.get("parallel_seconds"):
        speedup = round(serial["seconds"] / parallel["parallel_seconds"], 3)
    winner_index = serial.get("winner_index") if serial else None
    explanatory = None
    if winner_index is not None and parallel.get("parallel_seconds"):
        numerator = sum(r["seconds"] for r in serial["results"][: winner_index + 1])
        explanatory = round(numerator / parallel["parallel_seconds"], 3)
    return {
        "query": bundle["query"],
        "parallel_seconds": parallel["parallel_seconds"],
        "serial_seconds": serial["seconds"] if serial else None,
        "measured_speedup_serial_over_parallel": speedup,
        "same_sites": bool(serial) and serial["sites_configured"] == parallel["workers_launched"],
        "sites_configured": parallel["workers_launched"],
        "serial_sites_visited": serial["sites_visited"] if serial else None,
        "serial_winner_index": winner_index,
        "explanatory_speedup_until_winner": explanatory,
        "winner": parallel["winner"],
        "outcome": parallel["outcome"],
        "duplicate_hits": parallel["duplicate_hits"],
        "terminate_broadcasts": parallel["terminate_broadcasts"],
        "terminate_event_count": parallel["terminate_event_count"],
        "expected_loser_acks": parallel["expected_loser_acks"],
        "ack_count": len(parallel["acks"]),
        "missing_loser_acks": parallel["missing_loser_acks"],
        "unexpected_acks": parallel["unexpected_acks"],
        "failure_summary": parallel["failure_summary"],
        "not_found_count": parallel["not_found_count"],
        "status_updates_total": parallel["status_updates_total"],
        "context_audit_parallel": parallel["runtime_audit"],
        "context_audit_serial": serial["runtime_audit"] if serial else None,
        "cascade": {
            "winner": cascade["winner"],
            "outcome": cascade["outcome"],
            "duplicate_hits": cascade["duplicate_hits"],
            "terminate_broadcasts": cascade["terminate_broadcasts"],
            "terminate_ts": cascade["terminate_ts"],
            "acks": sorted(cascade["acks"]),
            "expected_loser_acks": cascade["expected_loser_acks"],
            "missing_loser_acks": cascade["missing_loser_acks"],
            "context_audit": cascade["runtime_audit"],
            "parallel_seconds": cascade["parallel_seconds"],
        } if cascade else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"exp10-4-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_dir = Path(args.evidence_root).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    receipts: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    llm = llm_adapter.LLMClient(
        endpoint=args.llm_endpoint, model=args.model,
        timeout=args.llm_timeout, max_attempts=args.llm_attempts,
        receipt_sink=receipts.append,
    )

    live = getattr(args, "level", "local_mirror") == "live_unverified"
    if live:
        sites = site_config.build_live_sites()
        server = None
    else:
        sites = site_config.build_sites(level="local_mirror")
        server = build_mirror_for(sites)
    bundle: Dict[str, Any] = {"query": site_config.QUERY, "run_id": run_id,
                              "level": getattr(args, "level", "local_mirror")}
    started = time.monotonic()
    try:
        if server is not None:
            server.__enter__()
            attach_urls(sites, server)
        if server is not None:
            # 镜像服务器按站点名路由；派生出来的竞争站点需要额外注册路由
            for extra in build_cascade_sites(sites):
                server.httpd.RequestHandlerClass.routes[extra.name] = {
                    "file": extra.name.split("--")[0],
                    "delay": extra.latency_seconds,
                    "fail": None,
                }
        bundle["sites"] = [s.to_dict() for s in sites]
        bundle["notes"] = site_config.NOTES
        bundle["mirror_base"] = f"http://127.0.0.1:{server.port}" if server is not None else None

        if args.mode == "cascade":
            if live:
                raise SystemExit("--level live_unverified 目前只支持 --mode full（级联竞争需要镜像路由）")
            cascade_sites = build_cascade_sites(sites)
            attach_urls(cascade_sites, server)
            bundle["cascade"] = await run_parallel_phase(
                cascade_sites, site_config.QUERY, site_config.CRITERION, args.timeout,
                "cascade", llm, observations, receipts, verbose=args.verbose,
            )
        else:
            bundle["parallel"] = await run_parallel_phase(
                sites, site_config.QUERY, site_config.CRITERION, args.timeout,
                "parallel", llm, observations, receipts, verbose=args.verbose,
            )
            bundle["serial"] = await run_serial_phase(
                sites, site_config.QUERY, site_config.CRITERION, args.timeout, llm, observations,
            )
            if not args.no_cascade and not live:
                # 级联/重复命中实验：同一批真实语料，制造三种竞争形态
                #   fast-a / fast-b : 两个几乎同时（>1s 后）命中的快速页 -> 一个赢，另一个要么在锁内被判
                #                     duplicate，要么被 terminate 在安全点拦下
                #   slow-hit        : 6s 才回来、且本来会命中的页面 -> terminate 到达时它还在抓取，
                #                     必须在安全点 ack 并关闭上下文
                #   negative        : 永远不会命中，用来确认它也被正确终止/记账
                cascade_sites = build_cascade_sites(sites)
                attach_urls(cascade_sites, server)
                bundle["cascade"] = await run_parallel_phase(
                    cascade_sites, site_config.QUERY, site_config.CRITERION, args.timeout,
                    "cascade", llm, observations, receipts, verbose=args.verbose,
                )
    finally:
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass

    bundle["duration_seconds"] = round(time.monotonic() - started, 3)
    bundle["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "llm_endpoint": args.llm_endpoint,
        "llm_model": args.model,
        "worker_timeout_seconds": args.timeout,
        "level": ("live_unverified(L1): 真实 HTTPS 直连，TLS 校验关闭" if live
                  else "local_mirror(L0) + synthetic_local(L2)"),
        "playwright": False,
    }
    bundle["llm_stats"] = {
        "responses_ok": llm.calls,
        "attempts": llm.attempts,
        "failures": llm.failures,
        "input_tokens": llm.total_input_tokens,
        "output_tokens": llm.total_output_tokens,
        "max_concurrent_in_flight": llm.max_in_flight,
    }
    summary = summarize(bundle)
    bundle["summary"] = summary

    # —— 落盘 ——
    events: List[Dict[str, Any]] = []
    for key in ("parallel", "serial", "cascade"):
        phase = bundle.get(key)
        if not phase:
            continue
        for env in phase.get("bus_events_list", []):
            events.append({"phase": phase.get("phase", key), **env})
        phase.pop("bus_events_list", None)
    write_jsonl(out_dir / "events.jsonl", events)
    write_jsonl(out_dir / "observations.jsonl", observations)
    write_jsonl(out_dir / "llm_receipts.jsonl", receipts)
    write_json(out_dir / "summary.json", {"run_id": run_id, "generated_at": utc_now(), **summary})
    if "parallel" in bundle:
        write_json(out_dir / "parallel.json", bundle["parallel"])
        write_json(out_dir / "serial.json", bundle["serial"])
    if "cascade" in bundle:
        write_json(out_dir / "cascade.json", bundle["cascade"])
    write_json(out_dir / "run.json", {k: v for k, v in bundle.items() if k != "summary"})

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print(f"\n[evidence] {out_dir}")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "cascade"], default="full")
    parser.add_argument("--level", choices=["local_mirror", "live_unverified"], default="local_mirror",
                        help="local_mirror=本地镜像(默认)；live_unverified=真实 HTTPS 站点直连(关 TLS 校验)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    parser.add_argument("--timeout", type=float, default=40.0, help="每站点抓取超时（秒）")
    parser.add_argument("--llm-endpoint", default=os.environ.get("LLM_ENDPOINT", llm_adapter.DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", llm_adapter.DEFAULT_MODEL))
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-attempts", type=int, default=4)
    parser.add_argument("--no-cascade", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="打印每一条总线消息")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
