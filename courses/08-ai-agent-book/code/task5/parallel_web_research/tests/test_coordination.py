"""
离线可复现的测试：用假 LLM + 假页面运行时验证**协调机制**（不需要网络、不需要模型）。

覆盖验收要求：单次结算 / 重复命中 / 终止 ack / 错误隔离 / 资源关闭 / 状态表 / 门禁。
"""

from __future__ import annotations

import asyncio

import llm_adapter
import message_bus as mb
from coordinator import Coordinator
from message_bus import BROADCAST, MessageBus
from runtime import FetchContext, FetchError, PageRuntime, Site, fetch_site
from sites import CORPUS_DIR
from worker import TaskState, WorkerAgent

PROVOST_TEXT = (
    "Office of the Provost. The provost is responsible for administering the academic "
    "program, including instruction and research."
)


class FakeLLM:
    """假 LLM：按关键字返回固定结论；await asyncio.sleep 是真实等待，不是假装并行。"""

    def __init__(self, delay: float = 0.02, fail_on=()):
        self.delay = delay
        self.fail_on = set(fail_on)
        self.calls = 0

    async def complete(self, prompt, context=None):
        self.calls += 1
        await asyncio.sleep(self.delay)
        site = (context or {}).get("site", "")
        if site in self.fail_on:
            raise llm_adapter.LLMError("FakeDown", "simulated provider outage", 1)
        # 判定只看页面正文（PROVOST_TEXT 里才有这句话），与站点名无关
        if "provost is responsible" in prompt:
            text = '{"found": true, "name": "Office of the Provost", "role": "Provost", "evidence": "the provost is responsible for administering the academic program"}'
        else:
            text = '{"found": false, "reason": "the page lists no provost"}'
        return llm_adapter.LLMResult(text=text, usage={}, response_id="fake",
                                     latency_seconds=self.delay, attempts=1)


class _FakeContext(FetchContext):
    """抓取上下文替身：按站点名查正文/延迟/故障，用真实 asyncio 等待。"""

    async def fetch(self, site, timeout):
        runtime = self.runtime
        delay = runtime.delay_map.get(site.name, 0.0)
        fail = runtime.fail_map.get(site.name)
        text = runtime.text_map.get(site.name, "")
        if fail == "hang":
            try:
                await asyncio.sleep(timeout + 5)
            except asyncio.CancelledError:
                self.fetch_abandoned = True
                raise
        if fail:
            raise FetchError(fail, "injected failure")
        if delay:
            await asyncio.sleep(delay)
        return {"url": site.url, "http_status": 200, "bytes": len(text),
                "text": text, "seconds": delay, "level": site.level}


class FakeRuntime(PageRuntime):
    """不打网络的运行时：按站点名给正文/延迟/故障，其余（计数/关闭）沿用真实实现。"""

    context_class = _FakeContext

    def __init__(self, text_map, delay_map=None, fail_map=None):
        super().__init__(corpus_dir=CORPUS_DIR, label="fake")
        self.text_map = dict(text_map)          # 站点名 -> 正文
        self.delay_map = dict(delay_map or {})  # 站点名 -> 抓取耗时
        self.fail_map = dict(fail_map or {})    # 站点名 -> 注入的抓取故障


async def _run_parallel(text_map, delay_map=None, fail_map=None, llm=None, timeout=5.0, workers=None):
    mb.reset_sequence()
    runtime = FakeRuntime(text_map, delay_map, fail_map)
    bus = MessageBus(verbose=False)
    coordinator = Coordinator(bus, "Provost of Stanford University", runtime,
                              criterion="the current Provost of Stanford University")
    names = workers if workers is not None else list(text_map)
    for index, name in enumerate(names):
        coordinator.add_worker(WorkerAgent(
            worker_id=f"agent-{index:02d}",
            site=Site(name=name, college="Fake", url=f"http://fake/{name}", level="offline_fixture"),
            bus=bus,
            runtime=runtime,
            llm=llm if llm is not None else FakeLLM(delay=0.02),
            query="Provost of Stanford University",
            timeout=timeout,
            criterion="the current Provost of Stanford University",
        ))
    result = await coordinator.run()
    result["runtime_audit"] = await runtime.shutdown()
    result["bus"] = bus
    result["runtime"] = runtime
    return result


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ 单次结算 / 重复命中

def test_near_simultaneous_hits_settle_exactly_once():
    async def scenario():
        bus = MessageBus(verbose=False)
        runtime = FakeRuntime({})
        coordinator = Coordinator(bus, "target", runtime)
        verdicts = await asyncio.gather(
            coordinator.settle("agent-a", {"name": "target"}),
            coordinator.settle("agent-b", {"name": "target"}),
            coordinator.settle("agent-c", {"name": "target"}),
        )
        return coordinator, bus, verdicts
    coordinator, bus, verdicts = run(scenario())
    assert sorted(verdicts) == ["duplicate", "duplicate", "winner"]
    assert coordinator.winner is not None
    assert len(coordinator.duplicate_hits) == 2
    assert bus.count(mb.TERMINATE) == 1


def test_late_hit_after_settlement_is_duplicate_not_resummary():
    async def scenario():
        bus = MessageBus(verbose=False)
        runtime = FakeRuntime({})
        coordinator = Coordinator(bus, "target", runtime)
        first = await coordinator.settle("agent-a", {"name": "target"})
        second = await coordinator.settle("agent-b", {"name": "target"})
        third = await coordinator.settle("agent-a", {"name": "target"})
        return coordinator, bus, (first, second, third)
    coordinator, bus, verdicts = run(scenario())
    assert verdicts == ("winner", "duplicate", "duplicate")
    assert coordinator.profile == {"name": "target"}       # 汇总仍是第一个命中
    assert len(coordinator.duplicate_hits) == 2
    assert bus.count(mb.TERMINATE) == 1
    assert bus.count(mb.LATE_RESULT) == 2


# ------------------------------------------------------------------ 级联终止 / ack

def test_cascade_terminates_exactly_once_and_losers_ack():
    # 用中性站点名（不含 Stanford/provost 之类的主题词），让门禁只由正文决定
    text_map = {
        "alpha": PROVOST_TEXT,
        "beta": PROVOST_TEXT,
        "gamma": "this page mentions nothing relevant",
        "delta": "this page mentions nothing relevant",
    }
    delay_map = {"alpha": 0.01, "delta": 0.4, "beta": 0.8, "gamma": 0.9}
    result = run(_run_parallel(text_map, delay_map=delay_map, timeout=5.0))
    assert result["outcome"] == "found"
    assert result["winner"] == "agent-00"      # alpha（最快且命中）
    assert result["terminate_broadcasts"] == 1
    assert result["terminate_event_count"] == 1
    assert len(result["expected_loser_acks"]) == 3
    assert result["missing_loser_acks"] == []
    assert set(result["acks"]) == set(result["expected_loser_acks"])
    for wid in result["expected_loser_acks"]:
        assert result["acks"][wid]["safe_point"] == "terminate_safe_point" or result["acks"][wid]["safe_point"]
    assert result["contexts"]["audit_ok"] is True
    # 未胜出 worker 都停在了"已终止"，没有伪装成完成
    for wid in result["expected_loser_acks"]:
        assert result["status_table"][wid]["state"] == TaskState.TERMINATED.value


def test_no_hits_means_no_terminate_broadcast():
    text_map = {"a": "nothing", "b": "nothing", "c": "nothing"}
    result = run(_run_parallel(text_map))
    assert result["outcome"] == "not_found"
    assert result["terminate_broadcasts"] == 0
    assert result["expected_loser_acks"] == []
    assert result["acks"] == {}
    assert result["not_found_count"] == 3
    assert result["contexts"]["audit_ok"] is True


def test_worker_already_finished_before_settlement_is_not_expected_to_ack():
    text_map = {"fast-negative": "nothing", "fast-hit": PROVOST_TEXT, "slow-negative": "nothing"}
    delay_map = {"fast-negative": 0.0, "fast-hit": 0.3, "slow-negative": 1.2}
    result = run(_run_parallel(text_map, delay_map=delay_map, timeout=5.0))
    assert result["expected_loser_acks"] == ["agent-02"]
    assert result["excluded_from_ack"].get("agent-00") == "already_not_found"
    assert result["acks"]["agent-02"]["stop_reason"] == "terminate_safe_point"


# ------------------------------------------------------------------ 错误隔离

def test_failed_sites_are_isolated_and_classified():
    text_map = {"good": "nothing", "bad-http": "nothing", "bad-tls": "nothing"}
    fail_map = {"bad-http": "HTTP500", "bad-tls": "TLSError"}
    result = run(_run_parallel(text_map, fail_map=fail_map))
    assert result["outcome"] == "not_found"
    assert result["not_found_count"] == 1
    assert result["failure_summary"] == {"count": 2, "by_type": {"HTTP500": 1, "TLSError": 1}}
    assert result["contexts"]["audit_ok"] is True


def test_hanging_site_times_out_while_peers_finish_and_contexts_close():
    text_map = {"healthy": PROVOST_TEXT, "hang": "nothing"}
    delay_map = {"healthy": 0.0}
    fail_map = {"hang": "hang"}
    result = run(_run_parallel(text_map, delay_map=delay_map, fail_map=fail_map, timeout=0.3))
    assert result["outcome"] == "found"
    assert result["winner"] == "agent-00"
    assert result["runtime_audit"]["contexts_created"] == 2
    assert result["runtime_audit"]["contexts_closed"] == 2


def test_llm_outage_fails_only_that_site():
    text_map = {"llm-down": PROVOST_TEXT, "healthy": "nothing"}
    llm = FakeLLM(delay=0.0, fail_on={"llm-down"})
    result = run(_run_parallel(text_map, llm=llm))
    assert result["outcome"] == "not_found"
    # 失败类别保留到 provider 层错误名（LLMError[FakeDown]），便于定位是网络还是模型问题
    assert result["failure_summary"]["by_type"] == {"LLMError[FakeDown]": 1}
    assert result["not_found_count"] == 1
    assert result["contexts"]["audit_ok"] is True


def test_resource_audit_counts_created_equals_closed():
    text_map = {f"site-{i}": "nothing here" for i in range(5)}
    result = run(_run_parallel(text_map))
    audit = result["runtime_audit"]
    assert audit["contexts_created"] == 5
    assert audit["contexts_closed"] == 5
    assert audit["contexts_leaked"] == 0
    assert audit["open_now"] == 0


# ------------------------------------------------------------------ 状态表 / 总线

def test_status_table_updates_in_realtime_for_every_worker():
    text_map = {f"site-{i}": "nothing here" for i in range(4)}
    result = run(_run_parallel(text_map))
    assert result["workers_launched"] == 4
    assert len(result["status_table"]) == 4
    for record in result["status_table"].values():
        assert record["updates"] >= 2
        assert record["state"] == TaskState.SUCCEEDED.value
        assert record["history"]
    assert result["status_updates_total"] >= 8


def test_bus_point_to_point_and_broadcast_semantics():
    async def scenario():
        mb.reset_sequence()
        bus = MessageBus(verbose=False)
        alice = bus.subscribe("alice", types=None)
        bob = bus.subscribe("bob", types=["terminate"])
        await bus.send("coordinator", "alice", "task_assigned", {"x": 1})
        await bus.send("coordinator", BROADCAST, "terminate", {"reason": "test"})
        return bus, alice, bob
    bus, alice, bob = run(scenario())
    assert [e.seq for e in bus.history] == [1, 2]
    assert bus.history[0].ts <= bus.history[1].ts
    assert [e.type for e in alice.drain_nowait()] == ["task_assigned", "terminate"]
    assert [e.type for e in bob.drain_nowait()] == ["terminate"]


# ------------------------------------------------------------------ 抓取 / 门禁

def test_offline_fixture_fetch_reads_real_corpus():
    site = Site(name="stanford-provost-office", college="x", url="file://corpus",
                level="offline_fixture")
    obs = fetch_site(site, timeout=2.0, corpus_dir=CORPUS_DIR)
    assert obs["http_status"] == 200
    assert "Office of the Provost" in obs["text"]


def test_deterministic_name_gate_blocks_absent_target():
    assert llm_adapter.name_gate("Provost of Stanford University", "nothing here")["present"] is False
    assert llm_adapter.name_gate("Andrew Ng", "Andrew Ng\nSenior Director of Finance")["present"] is True


def test_evidence_must_be_verbatim_in_page_text():
    text = "Jenny Martinez is Stanford fourteenth provost."
    assert llm_adapter.verify_evidence("is Stanford fourteenth provost", text)["verbatim"] is True
    assert llm_adapter.verify_evidence("She became provost after a long career", text)["verbatim"] is False


def test_extract_profile_rejects_non_verbatim_evidence():
    class HallucinatingLLM:
        async def complete(self, prompt, context=None):
            return llm_adapter.LLMResult(
                text='{"found": true, "name": "Office of the Provost", "evidence": "a sentence that is nowhere on the page"}',
                usage={}, response_id="x", latency_seconds=0.0, attempts=1)

    async def scenario():
        return await llm_adapter.extract_profile(
            HallucinatingLLM(), "Provost of Stanford University", "site", "http://x", PROVOST_TEXT,
            criterion="the current Provost of Stanford University")
    result = run(scenario())
    assert result["status"] == "unsupported"
    assert result["found"] is False


def test_gate_short_circuits_without_spending_llm_call():
    llm = FakeLLM(delay=0.0)

    async def scenario():
        return await llm_adapter.extract_profile(
            llm, "Provost of Stanford University", "site", "http://x", "a page with no match")
    result = run(scenario())
    assert result["status"] == "not_found"
    assert llm.calls == 0
