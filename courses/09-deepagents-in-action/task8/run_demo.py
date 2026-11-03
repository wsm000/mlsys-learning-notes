"""Task8 可复现 Demo：固定输入 + 明确预期（离线，零真实模型调用，无需 API Key）。

一条命令验收 Quanty（量价研究助理）在本课程改造后的 Agent 机制链：

    段落 A  工具调用与确定性边界   （课程能力 1：tools + 拒绝替代/拒绝编造）
    段落 B  长期记忆               （课程能力 2：ch08 文件记忆跨线程/跨进程 + StateBackend 对照）
    段落 C  人工审批与风险分层     （课程能力 3：ch09 interrupt_on 三决定 + L0-L3 权限分层与两条边界）

方法论（本课程 Task5/Task6 一脉相承）：
  * 固定输入：用户话语、脚本化模型步骤、合成数据 data/demo.csv、线程 ID 全部写死；
    数据与关键源码的 sha256 在运行时打印，指纹变了就能发现。
  * 明确预期：每一项检查的预期值在本文件 EXPECTED 表中事先声明（不是跑完再写）；
    实际值与预期逐项比较，任何一项不符即以退出码 1 失败。
  * 离线确定性：模型由 ScriptedModel 驱动（步骤可依赖"实际收到的 system prompt"），
    全程零真实模型请求，因此与 temperature/seed 无关，任何人都能复现同一结果。

它证明什么：上述机制链在机械层面按预期工作（脚本化驱动下的确定行为）。
它不证明什么：真实模型会这样行为（语义层面）；那需要真实模型运行，
见 courses/09-deepagents-in-action/evidence/task6/live-ds-01/ 与 Task6 打卡稿。

用法：
    python task8/run_demo.py                     # 默认写到 ./demo-output
    python task8/run_demo.py --out DIR --root DIR
退出码：0 = 全部检查通过；1 = 有检查不符或运行错误。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

TASK8_DIR = Path(__file__).resolve().parent
COURSE_DIR = TASK8_DIR.parent
TASK6_DIR = COURSE_DIR / "task6"
for p in (str(TASK8_DIR), str(TASK6_DIR), str(COURSE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lab_common import (  # noqa: E402  (Task6 冻结骨架：确定性 ScriptedModel + Recorder)
    Recorder,
    ScriptedModel,
    env_record,
    now_iso,
    sha256_text,
    system_prompt_text,
    text_of,
    write_json,
)

# ---------------------------------------------------------------------------
# 固定输入（FIXED INPUTS）——全部写死，任何修改都会改变指纹
# ---------------------------------------------------------------------------

QUANTY_SYSTEM_PROMPT = (
    "你是量价研究助理。量价比较调用 analyze_price_volume，以工具返回为依据。"
    "例外：用户要求的window超出1–5时，不得调用较小window代替，不得宣称完成。"
    "应说明当前仅10条数据，比较最近N条与此前N条需要2N条，工具目前仅支持1–5。"
    "询问用户是否接受5条演示，获得明确同意前不调用替代方案，不修改工具代码。"
    "只有 DEMO 模拟数据，不可冒充真实行情。不支持的标的应明确告知。"
    "给出区间、涨幅、前后日均成交量、比值及局限。比较结果不能证明因果或预测。"
    "简单查询无需规划、委派或文件操作。"
)  # 与 courses/09-deepagents-in-action/agent.py 第 33-39 行逐字一致（冻结副本）

# 段落 A：工具调用
Q_TOOL = "分析 DEMO 最近5个交易日上涨是否伴随成交量放大，列出依据和局限。"
FINAL_TOOL = (
    "依据：DEMO 最近5个交易日收益 6.0%，前期日均成交量 1000.0，近期日均成交量 1600.0，"
    "量比 1.6，上涨伴随成交量放大。局限：这是合成教学数据，不是真实行情；"
    "描述性窗口比较，不证明因果，也不能预测。"
)
Q_BOUNDARY = "分析 DEMO 最近10个交易日上涨是否伴随成交量放大。"
FINAL_BOUNDARY = (
    "当前只有10条合成数据，比较最近10条与此前10条需要20条，而工具仅支持1–5。"
    "我不会用较小窗口代替，也不宣称完成。你是否接受用5条做演示？"
    "获得明确同意前我不调用工具。"
)
Q_RSI = "给我 DEMO 的 RSI(14)。"
FINAL_RSI = (
    "DEMO 只有10条收盘价，RSI(14) 需要15条，数据不足；我不编造数值。"
    "可指定 period=3 或 5 计算教学示例。"
)

# 段落 B：长期记忆
PREFERENCE = "用户偏好：我喜欢中文回答，回答要简洁。"
MEMORY_PATH = "/memory/AGENTS.md"
Q_MEM_WRITE = "记住：我喜欢中文回答，回答要简洁。"
Q_MEM_READ = "我喜欢什么语言的回答？"
Q_STATE_WRITE = "记住：我喜欢中文回答。"
Q_STATE_READ = "我之前让你记的偏好呢？"
STATE_PATH = "/notes/pref.md"

# 段落 C：审批与风险分层
TARGET = "/reports/report.md"
PROPOSED = "PROPOSED-CONTENT-原始提议"
EDITED = "EDITED-CONTENT-修改后参数"
Q_REPORT = "把报告写到 /reports/report.md"
RISK_RULES = [
    {"operations": ["read"], "paths": ["/**"], "mode": "allow"},
    {"operations": ["write"], "paths": ["/public/**"], "mode": "allow"},
    {"operations": ["write"], "paths": ["/drafts/**"], "mode": "interrupt"},
    {"operations": ["write"], "paths": ["/secrets/**"], "mode": "deny"},
]
ORDERED_RULES = [
    {"operations": ["read"], "paths": ["/**"], "mode": "allow"},
    {"operations": ["write"], "paths": ["/drafts/blocked/**"], "mode": "deny"},
    {"operations": ["write"], "paths": ["/drafts/**"], "mode": "interrupt"},
]

# ---------------------------------------------------------------------------
# 明确预期（EXPECTED）——运行前声明；实际值必须逐项等于预期值
# ---------------------------------------------------------------------------
# A1 的预期值由 data/demo.csv 手工计算（window=5）：
#   baseline=前5条 (1000+1100+900+1000+1000)/5=1000.0
#   recent =后5条 (1400+1500+1600+1700+1800)/5=1600.0
#   ratio  =1600/1000=1.6; return=106/100-1=6.0%
# A1c（window=3，rows[-6:] = 第5–10行）：
#   baseline=第5-7条 (1000+1400+1500)/3=1300.0
#   recent =第8-10条 (1600+1700+1800)/3=1700.0
#   ratio  =1700/1300; return=106/103-1=2.912621%
EXPECTED = {
    # 段落 A：工具调用与确定性边界
    "A1_tool_values_match_hand_computed": {
        "price_return_pct": 6.0,
        "baseline_mean_volume": 1000.0,
        "recent_mean_volume": 1600.0,
        "volume_ratio": 1.6,
        "rise_with_volume_expansion": True,
    },
    "A1b_final_answer_grounded_in_tool_return": True,
    "A1c_window3_values_match_hand_computed": {
        "price_return_pct": round((106 / 103 - 1) * 100, 6),
        "baseline_mean_volume": 3900 / 3,
        "recent_mean_volume": 5100 / 3,
        "volume_ratio": (5100 / 3) / (3900 / 3),
    },
    "A2a_tool_boundary_raises_with_exact_message": True,
    "A2b_agent_propagates_boundary_error": True,
    "A2c_refusal_without_tool_call_or_fabrication": True,
    "A3_insufficient_data_no_invented_rsi": True,
    # 段落 B：长期记忆
    "B1_memory_file_on_disk": True,
    "B2_fresh_thread_system_prompt_has_memory": True,
    "B3_fresh_process_system_prompt_has_memory": True,
    "B4_control_without_memory_file": True,
    "B5_state_same_thread_sees_file": True,
    "B6_state_fresh_thread_blind": True,
    # 段落 C：人工审批与风险分层
    "C1_approve_interrupts_then_executes_proposed": True,
    "C2_reject_does_not_execute": True,
    "C3_edit_executes_only_edited_args": True,
    "C4_public_write_auto_executes": True,
    "C5_drafts_write_asks_human_then_executes": True,
    "C6_secrets_write_denied_without_asking": True,
    "C7_deny_declared_first_wins": True,
    "C8_edit_into_deny_still_denied": True,
    "C9_direct_backend_call_not_gated": True,
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def tool_payloads(rec: Recorder, name: str) -> list:
    """从 Recorder 事件中取出某个工具的全部返回（解析 JSON 内容）。"""
    out = []
    for e in rec.events:
        if e["event"] != "tool_end":
            continue
        output = e.get("output")
        if not isinstance(output, dict) or output.get("name") != name:
            continue
        content = output.get("content")
        payload = None
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except Exception:
                payload = content
        elif isinstance(content, list):
            payload = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        else:
            payload = content
        out.append({"status": output.get("status", "success"), "payload": payload})
    return out


def disk_files(fs_root: Path) -> dict:
    if not fs_root.exists():
        return {}
    return {
        str(p.relative_to(fs_root)).replace("\\", "/"): p.read_text(encoding="utf-8")
        for p in fs_root.rglob("*")
        if p.is_file()
    }


def interrupt_payload(result) -> list:
    payload = []
    for item in result.get("__interrupt__", []) or []:
        value = getattr(item, "value", item)
        payload.append(value if isinstance(value, (dict, list)) else str(value))
    return payload


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 段落 A：工具调用与确定性边界（真实 tools.py + 脚本化模型）
# ---------------------------------------------------------------------------

def build_tool_agent(model):
    from deepagents import create_deep_agent

    from tools import analyze_price_volume, calculate_rsi  # courses/09-.../tools.py

    return create_deep_agent(
        model=model,
        tools=[analyze_price_volume, calculate_rsi],
        system_prompt=QUANTY_SYSTEM_PROMPT,
    )


def segment_a(root: Path) -> dict:
    from tools import analyze_price_volume

    actual = {}
    detail = {}

    def run_tool_case(tag: str, question: str, call_args: dict, final_text: str, thread_id: str):
        model = ScriptedModel(
            steps=[("analyze_price_volume", call_args), ("final", final_text)],
            tag=tag,
        )
        rec = Recorder()
        agent = build_tool_agent(model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
        )
        return rec, result

    # A1: window=5——脚本化模型调用真实工具，返回值必须等于手工计算值
    rec, result = run_tool_case("tool-normal", Q_TOOL, {"symbol": "DEMO", "window": 5}, FINAL_TOOL, "tool-A1")
    returns = [r for r in tool_payloads(rec, "analyze_price_volume") if r["status"] == "success"]
    payload = returns[0]["payload"] if returns else {}
    actual["A1_tool_values_match_hand_computed"] = {
        "price_return_pct": payload.get("price_return_pct"),
        "baseline_mean_volume": payload.get("baseline_mean_volume"),
        "recent_mean_volume": payload.get("recent_mean_volume"),
        "volume_ratio": payload.get("volume_ratio"),
        "rise_with_volume_expansion": payload.get("rise_with_volume_expansion"),
    }
    final_text = text_of(result["messages"][-1].content)
    actual["A1b_final_answer_grounded_in_tool_return"] = (
        "6.0" in final_text and "1.6" in final_text and "局限" in final_text
    )
    detail["A1_tool_return"] = payload
    detail["A1_final_message"] = final_text

    # A1c: window=3——第二组固定输入/精确预期，证明不是只对 window=5 特调
    rec3, result3 = run_tool_case(
        "tool-window3", "分析 DEMO 最近3个交易日的量价变化。", {"symbol": "DEMO", "window": 3}, FINAL_TOOL, "tool-A1c"
    )
    returns3 = [r for r in tool_payloads(rec3, "analyze_price_volume") if r["status"] == "success"]
    payload3 = returns3[0]["payload"] if returns3 else {}
    actual["A1c_window3_values_match_hand_computed"] = {
        "price_return_pct": payload3.get("price_return_pct"),
        "baseline_mean_volume": payload3.get("baseline_mean_volume"),
        "recent_mean_volume": payload3.get("recent_mean_volume"),
        "volume_ratio": payload3.get("volume_ratio"),
    }
    detail["A1c_tool_return"] = payload3

    # A2a: 工具层边界——超界 window 必须以 raise 拒绝，且消息精确（不夹带、不静默钳位）
    boundary_errors = []
    for bad_window in (10, 0, -1, "5", 5.0):
        try:
            analyze_price_volume(symbol="DEMO", window=bad_window)
            boundary_errors.append(f"window={bad_window!r}: NO EXCEPTION")
        except ValueError as exc:
            boundary_errors.append(f"window={bad_window!r}: {exc}")
        except Exception as exc:  # 其他异常类型也是不符合契约
            boundary_errors.append(f"window={bad_window!r}: {type(exc).__name__}: {exc}")
    expected_msg = "window must be an integer between 1 and 5"
    actual["A2a_tool_boundary_raises_with_exact_message"] = all(
        e == f"window={w!r}: {expected_msg}" for e, w in zip(boundary_errors, (10, 0, -1, "5", 5.0))
    )
    detail["A2a_boundary_errors"] = boundary_errors

    # A2b: Agent 层传播——脚本化模型真去调用 window=10 时，异常冲出 agent.invoke()
    # （机制边界：本机 langgraph 1.2.15 的 _default_handle_tool_errors 直接 raise，
    #  工具异常不会变成模型可见的 error ToolMessage；调用方必须自己接住。）
    model_b = ScriptedModel(
        steps=[
            ("analyze_price_volume", {"symbol": "DEMO", "window": 10}),
            ("final", FINAL_BOUNDARY),
        ],
        tag="tool-boundary",
    )
    propagated = None
    try:
        build_tool_agent(model_b).invoke(
            {"messages": [{"role": "user", "content": Q_BOUNDARY}]},
            config={"configurable": {"thread_id": "tool-A2"}, "callbacks": [Recorder()]},
        )
        propagated = "NO EXCEPTION"
    except ValueError as exc:
        propagated = str(exc)
    except Exception as exc:
        propagated = f"{type(exc).__name__}: {exc}"
    actual["A2b_agent_propagates_boundary_error"] = propagated == expected_msg
    detail["A2b_propagated_error"] = propagated

    # A2c: 拒绝路径——不调用工具、不替代、不宣称完成、不编造数值（机械验证"未发生调用"）
    model_c = ScriptedModel(steps=[("final", FINAL_BOUNDARY)], tag="tool-refusal")
    rec_c = Recorder()
    result_c = build_tool_agent(model_c).invoke(
        {"messages": [{"role": "user", "content": Q_BOUNDARY}]},
        config={"configurable": {"thread_id": "tool-A2c"}, "callbacks": [rec_c]},
    )
    final_c = text_of(result_c["messages"][-1].content)
    tool_calls_made = [
        e for e in rec_c.events if e["event"] in ("tool_start", "tool_end") and "analyze_price_volume" in json.dumps(e, ensure_ascii=False)
    ]
    actual["A2c_refusal_without_tool_call_or_fabrication"] = (
        not tool_calls_made
        and ("是否" in final_c or "接受" in final_c)
        and "6.0" not in final_c
        and "1.6" not in final_c
    )
    detail["A2c_final_message"] = final_c

    # A3: 数据不足——RSI(14) 需要 15 条而只有 10 条：返回 insufficient_data，不编造
    model_d = ScriptedModel(
        steps=[
            ("calculate_rsi", {"symbol": "DEMO", "period": 14}),
            ("final", FINAL_RSI),
        ],
        tag="tool-rsi",
    )
    rec_d = Recorder()
    result_d = build_tool_agent(model_d).invoke(
        {"messages": [{"role": "user", "content": Q_RSI}]},
        config={"configurable": {"thread_id": "tool-A3"}, "callbacks": [rec_d]},
    )
    rsi_returns = [r for r in tool_payloads(rec_d, "calculate_rsi") if r["status"] == "success"]
    rsi = rsi_returns[0]["payload"] if rsi_returns else {}
    final_d = text_of(result_d["messages"][-1].content)
    actual["A3_insufficient_data_no_invented_rsi"] = (
        rsi.get("status") == "insufficient_data"
        and rsi.get("rsi") is None
        and rsi.get("available_prices") == 10
        and rsi.get("required_prices") == 15
        and "不编造" in final_d
    )
    detail["A3_rsi_return"] = rsi
    detail["A3_final_message"] = final_d

    return actual, detail


# ---------------------------------------------------------------------------
# 段落 B：长期记忆（ch08）
# ---------------------------------------------------------------------------

def build_memory_agent(model, backend, checkpointer):
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        memory=[MEMORY_PATH],
        backend=backend,
        checkpointer=checkpointer,
    )


def memory_answer_step(preference):
    """脚本化回答依赖实际收到的 system prompt——机械证明注入，不证语义。"""

    def step(messages):
        prompt = system_prompt_text(messages)
        if preference in prompt:
            return ("final", "根据我的记忆，你喜欢中文回答。")
        return ("final", "我没有关于你的记忆。")

    return step


def segment_b(root: Path) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    actual = {}
    detail = {}

    # B1/B2: 线程 A 写记忆文件；全新线程 B（同 backend）的 system prompt 必须含偏好
    fs_root = fresh_dir(root / "fs_memory")
    backend = FilesystemBackend(root_dir=str(fs_root))
    checkpointer = MemorySaver()

    agent_a = build_memory_agent(
        ScriptedModel(
            steps=[
                ("write_file", {"file_path": MEMORY_PATH, "content": PREFERENCE}),
                ("final", "已记住：你喜欢中文回答。"),
            ],
            tag="mem-writer",
        ),
        backend,
        checkpointer,
    )
    result_a = agent_a.invoke(
        {"messages": [{"role": "user", "content": Q_MEM_WRITE}]},
        config={"configurable": {"thread_id": "mem-A"}, "callbacks": [Recorder()]},
    )
    disk_text = (
        (fs_root / "memory" / "AGENTS.md").read_text(encoding="utf-8")
        if (fs_root / "memory" / "AGENTS.md").exists()
        else None
    )
    actual["B1_memory_file_on_disk"] = disk_text == PREFERENCE
    detail["B1_disk_text"] = disk_text

    rec_b = Recorder()
    agent_b = build_memory_agent(ScriptedModel(steps=[memory_answer_step(PREFERENCE)], tag="mem-reader"), backend, checkpointer)
    result_b = agent_b.invoke(
        {"messages": [{"role": "user", "content": Q_MEM_READ}]},
        config={"configurable": {"thread_id": "mem-B-fresh"}, "callbacks": [rec_b]},
    )
    prompt_b = system_prompt_text(rec_b.requests[0]["messages"]) if rec_b.requests else ""
    actual["B2_fresh_thread_system_prompt_has_memory"] = PREFERENCE in prompt_b
    detail["B2_final_message"] = result_b["messages"][-1].content

    # B3: 全新进程（新建 agent 实例 + 新 checkpointer，同一磁盘 backend）
    agent_c = build_memory_agent(
        ScriptedModel(steps=[memory_answer_step(PREFERENCE)], tag="mem-reader-fresh-process"),
        FilesystemBackend(root_dir=str(fs_root)),
        MemorySaver(),
    )
    rec_c = Recorder()
    result_c = agent_c.invoke(
        {"messages": [{"role": "user", "content": Q_MEM_READ}]},
        config={"configurable": {"thread_id": "mem-C-fresh-process"}, "callbacks": [rec_c]},
    )
    prompt_c = system_prompt_text(rec_c.requests[0]["messages"]) if rec_c.requests else ""
    actual["B3_fresh_process_system_prompt_has_memory"] = PREFERENCE in prompt_c
    detail["B3_final_message"] = result_c["messages"][-1].content

    # B4: 对照——空 root，system prompt 不含偏好
    fs_control = fresh_dir(root / "fs_control")
    rec_d = Recorder()
    agent_d = build_memory_agent(
        ScriptedModel(steps=[memory_answer_step(PREFERENCE)], tag="mem-reader-control"),
        FilesystemBackend(root_dir=str(fs_control)),
        MemorySaver(),
    )
    agent_d.invoke(
        {"messages": [{"role": "user", "content": Q_MEM_READ}]},
        config={"configurable": {"thread_id": "mem-D-control"}, "callbacks": [rec_d]},
    )
    prompt_d = system_prompt_text(rec_d.requests[0]["messages"]) if rec_d.requests else ""
    actual["B4_control_without_memory_file"] = PREFERENCE not in prompt_d

    # B5/B6: 默认 StateBackend——同线程可见，新线程不可见（"新对话就忘"的机制根源）
    checkpointer_state = MemorySaver()
    rec_e = Recorder()
    agent_e = build_memory_agent(
        ScriptedModel(
            steps=[
                ("write_file", {"file_path": STATE_PATH, "content": PREFERENCE}),
                ("ls", {"path": "/notes"}),
                ("final", "同一线程内我已写下偏好。"),
            ],
            tag="state-writer",
        ),
        None,  # None -> 默认 StateBackend
        checkpointer_state,
    )
    agent_e.invoke(
        {"messages": [{"role": "user", "content": Q_STATE_WRITE}]},
        config={"configurable": {"thread_id": "mem-E-state"}, "callbacks": [rec_e]},
    )
    same_thread_ls = [e for e in rec_e.events if e["event"] == "tool_end" and "pref.md" in json.dumps(e, ensure_ascii=False)]
    actual["B5_state_same_thread_sees_file"] = bool(same_thread_ls)

    rec_f = Recorder()
    agent_f = build_memory_agent(
        ScriptedModel(
            steps=[
                ("ls", {"path": "/"}),
                ("final", "新线程里我没有看到任何文件。"),
            ],
            tag="state-reader",
        ),
        None,
        checkpointer_state,
    )
    agent_f.invoke(
        {"messages": [{"role": "user", "content": Q_STATE_READ}]},
        config={"configurable": {"thread_id": "mem-F-fresh"}, "callbacks": [rec_f]},
    )
    fresh_thread_ls = [e for e in rec_f.events if e["event"] == "tool_end" and "pref.md" in json.dumps(e, ensure_ascii=False)]
    actual["B6_state_fresh_thread_blind"] = not fresh_thread_ls

    return actual, detail


# ---------------------------------------------------------------------------
# 段落 C：人工审批与风险分层（ch09 + 学余力风险分层方案）
# ---------------------------------------------------------------------------

def build_approval_agent(model, backend, checkpointer):
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        backend=backend,
        checkpointer=checkpointer,
        interrupt_on={"write_file": True},
    )


def build_risk_agent(model, backend, checkpointer, rules):
    from deepagents import create_deep_agent
    from deepagents.middleware.filesystem import FilesystemPermission

    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        backend=backend,
        checkpointer=checkpointer,
        permissions=[FilesystemPermission(**r) for r in rules],
    )


def denied(case) -> bool:
    return any(
        "denied" in (m.get("content") or "").lower() or "permission" in (m.get("content") or "").lower()
        for m in case.get("tool_messages", [])
    )


def run_approval_case(name: str, decision: dict, fs_root: Path) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.types import Command
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = fresh_dir(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    checkpointer = MemorySaver()
    thread_id = f"approval-{name}"

    agent = build_approval_agent(
        ScriptedModel(
            steps=[
                ("write_file", {"file_path": TARGET, "content": PROPOSED}),
                ("final", f"案例 {name} 结束。"),
            ],
            tag=f"approval-{name}",
        ),
        backend,
        checkpointer,
    )
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": Q_REPORT}]},
        config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
    )
    payload = interrupt_payload(result)
    pre_disk = disk_files(fs_root)
    resumed = agent.invoke(
        Command(resume={"decisions": [decision]}),
        config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
    )
    return {
        "case": name,
        "decision": decision,
        "interrupted": bool(payload),
        "interrupt_payload": payload,
        "pre_resume_disk": pre_disk,
        "post_resume_disk": disk_files(fs_root),
        "final_message": resumed["messages"][-1].content,
        "events": rec.events,
    }


def run_risk_simple(name: str, path: str, content: str, fs_root: Path, rules=RISK_RULES) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = fresh_dir(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_risk_agent(
        ScriptedModel(
            steps=[
                ("write_file", {"file_path": path, "content": content}),
                ("final", f"案例 {name} 结束。"),
            ],
            tag=f"risk-{name}",
        ),
        backend,
        MemorySaver(),
        rules,
    )
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"写入 {path}"}]},
        config={"configurable": {"thread_id": f"risk-{name}"}, "callbacks": [rec]},
    )
    tool_messages = [
        e["output"] for e in rec.events
        if e["event"] == "tool_end" and isinstance(e.get("output"), dict) and e["output"].get("type") == "tool"
    ]
    return {
        "case": name,
        "path": path,
        "interrupted": bool(result.get("__interrupt__")),
        "tool_messages": tool_messages,
        "disk_files": disk_files(fs_root),
        "final_message": result["messages"][-1].content,
    }


def run_risk_interrupt_then(name: str, path: str, content: str, decision: dict, fs_root: Path, rules=RISK_RULES) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.types import Command
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = fresh_dir(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_risk_agent(
        ScriptedModel(
            steps=[
                ("write_file", {"file_path": path, "content": content}),
                ("final", f"案例 {name} 结束。"),
            ],
            tag=f"risk-{name}",
        ),
        backend,
        MemorySaver(),
        rules,
    )
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"写入 {path}"}]},
        config={"configurable": {"thread_id": f"risk-{name}"}, "callbacks": [rec]},
    )
    pre_disk = disk_files(fs_root)
    resumed = agent.invoke(
        Command(resume={"decisions": [decision]}),
        config={"configurable": {"thread_id": f"risk-{name}"}, "callbacks": [rec]},
    )
    tool_messages = [
        e["output"] for e in rec.events
        if e["event"] == "tool_end" and isinstance(e.get("output"), dict) and e["output"].get("type") == "tool"
    ]
    return {
        "case": name,
        "path": path,
        "decision": decision,
        "interrupted": bool(result.get("__interrupt__")),
        "pre_resume_disk": pre_disk,
        "tool_messages": tool_messages,
        "disk_files": disk_files(fs_root),
        "final_message": resumed["messages"][-1].content,
    }


def run_direct_backend_case(fs_root: Path) -> dict:
    from deepagents.backends import FilesystemBackend

    fs_root = fresh_dir(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    result = backend.write("/secrets/direct.md", "DIRECT-BACKEND-WRITE")
    return {
        "case": "direct_backend",
        "write_result": {
            "result_type": type(result).__name__,
            "error": getattr(result, "error", None),
            "path": getattr(result, "path", None),
        },
        "disk_files": disk_files(fs_root),
    }


def segment_c(root: Path) -> dict:
    actual = {}
    cases = {}

    cases["approve"] = run_approval_case("approve", {"type": "approve"}, root / "fs_approve")
    cases["reject"] = run_approval_case("reject", {"type": "reject", "message": "这份报告先不写"}, root / "fs_reject")
    cases["edit"] = run_approval_case(
        "edit",
        {"type": "edit", "edited_action": {"name": "write_file", "args": {"file_path": TARGET, "content": EDITED}}},
        root / "fs_edit",
    )
    approve_files = cases["approve"]["post_resume_disk"]
    reject_files = cases["reject"]["post_resume_disk"]
    edit_files = cases["edit"]["post_resume_disk"]

    actual["C1_approve_interrupts_then_executes_proposed"] = (
        cases["approve"]["interrupted"]
        and not cases["approve"]["pre_resume_disk"]
        and approve_files.get(TARGET.lstrip("/")) == PROPOSED
    )
    actual["C2_reject_does_not_execute"] = cases["reject"]["interrupted"] and not reject_files
    actual["C3_edit_executes_only_edited_args"] = (
        edit_files.get(TARGET.lstrip("/")) == EDITED and edit_files.get(TARGET.lstrip("/")) != PROPOSED
    )

    cases["L1_public_allow"] = run_risk_simple("L1_public_allow", "/public/note.md", "PUBLIC", root / "fs_public")
    cases["L2_drafts_interrupt_approve"] = run_risk_interrupt_then(
        "L2_drafts_interrupt_approve", "/drafts/d.md", "DRAFT", {"type": "approve"}, root / "fs_drafts"
    )
    cases["L3_secrets_deny"] = run_risk_simple("L3_secrets_deny", "/secrets/s.md", "SECRET", root / "fs_secrets")
    cases["L3b_deny_before_interrupt"] = run_risk_simple(
        "L3b_deny_before_interrupt", "/drafts/blocked/x.md", "BLOCKED", root / "fs_blocked", rules=ORDERED_RULES
    )
    cases["H11_edit_into_deny"] = run_risk_interrupt_then(
        "H11_edit_into_deny",
        "/drafts/ok.md",
        "DRAFT-OK",
        {"type": "edit", "edited_action": {"name": "write_file", "args": {"file_path": "/secrets/evil.md", "content": "EVIL"}}},
        root / "fs_edit_deny",
    )
    cases["H12_direct_backend"] = run_direct_backend_case(root / "fs_direct")

    actual["C4_public_write_auto_executes"] = (
        not cases["L1_public_allow"]["interrupted"] and cases["L1_public_allow"]["disk_files"].get("public/note.md") == "PUBLIC"
    )
    actual["C5_drafts_write_asks_human_then_executes"] = (
        cases["L2_drafts_interrupt_approve"]["interrupted"]
        and not cases["L2_drafts_interrupt_approve"]["pre_resume_disk"]
        and cases["L2_drafts_interrupt_approve"]["disk_files"].get("drafts/d.md") == "DRAFT"
    )
    actual["C6_secrets_write_denied_without_asking"] = (
        not cases["L3_secrets_deny"]["interrupted"] and denied(cases["L3_secrets_deny"]) and not cases["L3_secrets_deny"]["disk_files"]
    )
    actual["C7_deny_declared_first_wins"] = (
        not cases["L3b_deny_before_interrupt"]["interrupted"]
        and denied(cases["L3b_deny_before_interrupt"])
        and not cases["L3b_deny_before_interrupt"]["disk_files"]
    )
    actual["C8_edit_into_deny_still_denied"] = (
        cases["H11_edit_into_deny"]["interrupted"] and denied(cases["H11_edit_into_deny"]) and not cases["H11_edit_into_deny"]["disk_files"]
    )
    actual["C9_direct_backend_call_not_gated"] = (
        cases["H12_direct_backend"]["write_result"]["error"] is None
        and cases["H12_direct_backend"]["disk_files"].get("secrets/direct.md") == "DIRECT-BACKEND-WRITE"
    )

    return actual, cases


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_fixed_inputs(out: Path, root: Path) -> dict:
    demo_csv = COURSE_DIR / "data" / "demo.csv"
    fingerprints = {
        "run_demo.py": sha256_file(TASK8_DIR / "run_demo.py"),
        "tools.py": sha256_file(COURSE_DIR / "tools.py"),
        "data/demo.csv": sha256_file(demo_csv),
        "task6/lab_common.py": sha256_file(TASK6_DIR / "lab_common.py"),
    }
    print("=" * 74)
    print("Quanty 可复现 Demo —— 固定输入 + 明确预期（离线；零真实模型调用；无需 API Key）")
    print("=" * 74)
    env = env_record()
    print(f"[环境] python {env['python']} / deepagents {env.get('deepagents')} / langchain {env.get('langchain')} / langgraph {env.get('langgraph')}")
    print(f"[环境] 平台 {env['platform']}")
    print("[固定输入] 文件指纹（sha256，改动即可被发现）：")
    for name, digest in fingerprints.items():
        print(f"    {name:<22} {digest}")
    print(f"[固定输入] 用户话语：{Q_TOOL}")
    print(f"[固定输入] 用户话语：{Q_BOUNDARY}")
    print(f"[固定输入] 用户话语：{Q_RSI}")
    print(f"[固定输入] 记忆偏好：{PREFERENCE}  ->  {MEMORY_PATH}")
    print(f"[固定输入] 审批目标：{TARGET}（提议={PROPOSED} / 修改后={EDITED}）")
    print("[固定输入] 风险规则（声明序，首个匹配生效）：")
    for r in RISK_RULES:
        print(f"    {r['operations']} {r['paths']} -> {r['mode']}")
    print("[固定输入] 模型：ScriptedModel（确定性脚本，与 temperature/seed 无关）")
    print(f"[明确预期] {len(EXPECTED)} 项检查的预期值在本脚本 EXPECTED 表事先声明，逐项比对")
    print(f"[输出] out={out}")
    print(f"[输出] root={root}")
    return fingerprints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demo-output", help="证据输出目录（默认 ./demo-output）")
    parser.add_argument("--root", default=None, help="FilesystemBackend 沙箱根目录（默认 <out>/sandbox）")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out = Path(args.out).resolve()
    root = Path(args.root).resolve() if args.root else out / "sandbox"
    out.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    fingerprints = print_fixed_inputs(out, root)

    actual = {}
    detail = {}
    errors = []

    for label, fn in (
        ("段落 A：工具调用与确定性边界（能力 1：tools + 拒绝替代/拒绝编造）", lambda: segment_a(root)),
        ("段落 B：长期记忆（能力 2：ch08 文件记忆 + StateBackend 对照）", lambda: segment_b(root)),
        ("段落 C：人工审批与风险分层（能力 3：ch09 审批 + L0-L3 权限分层）", lambda: segment_c(root)),
    ):
        print()
        print("-" * 74)
        print(label)
        print("-" * 74)
        try:
            seg_actual, seg_detail = fn()
            actual.update(seg_actual)
            detail[label.split("：")[0]] = seg_detail
        except Exception as exc:  # 运行错误也是失败，不静默跳过
            errors.append({"segment": label, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[运行错误] {type(exc).__name__}: {exc}")

    print()
    print("=" * 74)
    print("验收结果（预期值事先声明；任何一项不符即失败）")
    print("=" * 74)
    passed = 0
    rows = []
    for check_id, expected in EXPECTED.items():
        value = actual.get(check_id, "<未运行>")
        seg_letter = check_id[0]
        segment_failed = any(e["segment"].startswith(f"段落 {seg_letter}") for e in errors)
        if segment_failed and value == "<未运行>":
            value = "<段落运行错误>"
        ok = (value == expected) and not segment_failed
        rows.append({"check": check_id, "expected": expected, "actual": value, "passed": bool(ok)})
        passed += 1 if ok else 0
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {check_id}")
        print(f"       预期: {json.dumps(expected, ensure_ascii=False)}")
        print(f"       实际: {json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value}")

    all_passed = passed == len(EXPECTED) and not errors
    print()
    print("=" * 74)
    print(f"总结：{passed}/{len(EXPECTED)} 项检查通过" + (f"；{len(errors)} 个段落运行错误" if errors else ""))
    print(f"结论：{'全部通过 —— 机制链按固定输入与明确预期工作' if all_passed else '存在不符 —— 见上方 FAIL 项'}")
    print(f"证据：{out / 'summary.json'}")
    print("提醒：scripted 只证机械链，不证真实模型语义（见 README「它不证明什么」）")
    print("=" * 74)

    summary = {
        "experiment": "task8-reproducible-demo",
        "at": now_iso(),
        "env": env_record(),
        "fixed_input_fingerprints": fingerprints,
        "expected": EXPECTED,
        "checks": rows,
        "passed": passed,
        "total": len(EXPECTED),
        "all_passed": all_passed,
        "segment_errors": errors,
        "detail": detail,
    }
    write_json(out / "summary.json", summary)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
