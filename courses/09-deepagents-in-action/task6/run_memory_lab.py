"""Task6 E1: cross-thread file memory (Lite + Basic) and the course question.

Hypotheses (see experiment-plan.json):
  H1: a preference written to /memory/AGENTS.md in thread A is injected into the
      system prompt of a FRESH thread B (different thread_id) on the same backend.
  H2 (control): without the file, a fresh thread's system prompt carries no preference.
  H3 (course question): with the default StateBackend, a file written in thread E
      is visible in thread E itself but NOT in a fresh thread F -- thread state is
      thread-scoped; that is why a new conversation forgets you.
  H1b: cross-PROCESS: a fresh process/agent still reads the file (--phase read).

Evidence: the recorder captures the REAL system prompt sent to the model each
turn; checks assert on recorded bytes, not on intentions.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_common import Recorder, ScriptedModel, env_record, now_iso, sha256_text, system_prompt_text, write_json

PREFERENCE = "用户偏好：我喜欢中文回答，回答要简洁。"
MEMORY_PATH = "/memory/AGENTS.md"


def build_agent(model, backend, checkpointer=None):
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        memory=[MEMORY_PATH],
        backend=backend,
        checkpointer=checkpointer,
    )


def memory_answer_step(preference):
    """Scripted answer that depends on the ACTUAL system prompt received."""

    def step(messages):
        prompt = system_prompt_text(messages)
        if preference in prompt:
            return ("final", "根据我的记忆，你喜欢中文回答。")
        return ("final", "我没有关于你的记忆。")

    return step


def run_write_phase(root: Path, out: Path) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = root / "fs"
    backend = FilesystemBackend(root_dir=str(fs_root))
    checkpointer = MemorySaver()
    records = {}

    # --- Thread A: write the preference into the agreed memory file ---
    model_a = ScriptedModel(
        steps=[
            ("write_file", {"file_path": MEMORY_PATH, "content": PREFERENCE}),
            ("final", "已记住：你喜欢中文回答。"),
        ],
        tag="writer",
    )
    agent_a = build_agent(model_a, backend, checkpointer)
    rec_a = Recorder()
    result_a = agent_a.invoke(
        {"messages": [{"role": "user", "content": "记住：我喜欢中文回答，回答要简洁。"}]},
        config={"configurable": {"thread_id": "mem-A"}, "callbacks": [rec_a]},
    )
    disk_text = (fs_root / "memory" / "AGENTS.md").read_text(encoding="utf-8") if (fs_root / "memory" / "AGENTS.md").exists() else None
    records["thread_A_write"] = {
        "thread_id": "mem-A",
        "final_message": result_a["messages"][-1].content,
        "disk_file_text": disk_text,
        "disk_matches_preference": disk_text == PREFERENCE,
        "requests": rec_a.requests,
        "events": rec_a.events,
    }

    # --- Thread B: FRESH thread, same backend; does the memory arrive? ---
    model_b = ScriptedModel(steps=[memory_answer_step(PREFERENCE)], tag="reader")
    agent_b = build_agent(model_b, backend, checkpointer)
    rec_b = Recorder()
    result_b = agent_b.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么语言的回答？"}]},
        config={"configurable": {"thread_id": "mem-B"}, "callbacks": [rec_b]},
    )
    prompt_b = system_prompt_text(rec_b.requests[0]["messages"]) if rec_b.requests else ""
    records["thread_B_fresh_read"] = {
        "thread_id": "mem-B",
        "final_message": result_b["messages"][-1].content,
        "system_prompt_contains_preference": PREFERENCE in prompt_b,
        "system_prompt_excerpt": prompt_b[:1200],
        "requests": rec_b.requests,
    }
    return records


def run_read_phase(root: Path, out: Path) -> dict:
    """Fresh PROCESS: new agent instance, same on-disk backend."""
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = root / "fs"
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(ScriptedModel(steps=[memory_answer_step(PREFERENCE)], tag="reader-fresh-process"), backend, MemorySaver())
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么语言的回答？"}]},
        config={"configurable": {"thread_id": "mem-C-fresh-process"}, "callbacks": [rec]},
    )
    prompt = system_prompt_text(rec.requests[0]["messages"]) if rec.requests else ""
    return {
        "thread_id": "mem-C-fresh-process",
        "final_message": result["messages"][-1].content,
        "system_prompt_contains_preference": PREFERENCE in prompt,
        "requests": rec.requests,
    }


def run_control_phase(root: Path, out: Path) -> dict:
    """Control: empty backend root -> no preference in the system prompt."""
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = root / "fs_control"
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(ScriptedModel(steps=[memory_answer_step(PREFERENCE)], tag="reader-control"), backend, MemorySaver())
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么语言的回答？"}]},
        config={"configurable": {"thread_id": "mem-D-control"}, "callbacks": [rec]},
    )
    prompt = system_prompt_text(rec.requests[0]["messages"]) if rec.requests else ""
    return {
        "thread_id": "mem-D-control",
        "final_message": result["messages"][-1].content,
        "system_prompt_contains_preference": PREFERENCE in prompt,
        "system_prompt_excerpt": prompt[:600],
        "requests": rec.requests,
    }


def run_state_backend_phase(root: Path, out: Path) -> dict:
    """H3: default StateBackend -- same thread remembers, fresh thread does not."""
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()

    # Thread E: write a file into STATE, then list it in the SAME thread.
    agent_e = build_agent(
        ScriptedModel(
            steps=[
                ("write_file", {"file_path": "/notes/pref.md", "content": PREFERENCE}),
                ("ls", {"path": "/notes"}),
                ("final", "同一线程内我已写下偏好。"),
            ],
            tag="state-writer",
        ),
        None,
        checkpointer,
    )  # None -> default StateBackend
    rec_e = Recorder()
    res_e = agent_e.invoke(
        {"messages": [{"role": "user", "content": "记住：我喜欢中文回答。"}]},
        config={"configurable": {"thread_id": "mem-E-state"}, "callbacks": [rec_e]},
    )
    ls_same_thread = [
        e for e in rec_e.events
        if e.get("name") == "ls" or (e["event"] == "tool_end" and isinstance(e.get("output"), dict) and e["output"].get("name") == "ls")
    ]

    # Thread F: FRESH thread on the same graph+checkpointer: is the file there?
    agent_f = build_agent(
        ScriptedModel(
            steps=[
                ("ls", {"path": "/"}),
                ("final", "新线程里我没有看到任何文件。"),
            ],
            tag="state-reader",
        ),
        None,
        checkpointer,
    )
    rec_f = Recorder()
    res_f = agent_f.invoke(
        {"messages": [{"role": "user", "content": "我之前让你记的偏好呢？"}]},
        config={"configurable": {"thread_id": "mem-F-fresh"}, "callbacks": [rec_f]},
    )
    ls_fresh_thread = [
        e for e in rec_f.events
        if e.get("name") == "ls" or (e["event"] == "tool_end" and isinstance(e.get("output"), dict) and e["output"].get("name") == "ls")
    ]

    return {
        "thread_E_same_thread": {
            "thread_id": "mem-E-state",
            "final_message": res_e["messages"][-1].content,
            "ls_events_same_thread": ls_same_thread,
            "same_thread_sees_file": any("pref.md" in json.dumps(e, ensure_ascii=False) for e in ls_same_thread),
        },
        "thread_F_fresh": {
            "thread_id": "mem-F-fresh",
            "final_message": res_f["messages"][-1].content,
            "ls_events_fresh_thread": ls_fresh_thread,
            "fresh_thread_sees_file": any("pref.md" in json.dumps(e, ensure_ascii=False) for e in ls_fresh_thread),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", required=True, help="sandbox root for backend files (TEMP)")
    parser.add_argument("--phase", choices=["write", "read", "control", "state", "all"], default="all")
    args = parser.parse_args()

    out = Path(args.out)
    root = Path(args.root)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase in ("write", "all") and (root / "fs").exists():
        shutil.rmtree(root / "fs")

    results = {}
    if args.phase in ("write", "all"):
        results["write_phase"] = run_write_phase(root, out)
    if args.phase in ("read", "all"):
        results["read_phase_fresh_process"] = run_read_phase(root, out)
    if args.phase in ("control", "all"):
        results["control_phase"] = run_control_phase(root, out)
    if args.phase in ("state", "all"):
        results["state_backend_phase"] = run_state_backend_phase(root, out)

    checks = {}
    if "write_phase" in results:
        checks["H1_disk_file_written"] = results["write_phase"]["thread_A_write"]["disk_matches_preference"]
        checks["H1_fresh_thread_system_prompt_has_preference"] = results["write_phase"]["thread_B_fresh_read"]["system_prompt_contains_preference"]
        checks["H1_fresh_thread_answer_uses_memory"] = "中文" in results["write_phase"]["thread_B_fresh_read"]["final_message"]
    if "read_phase_fresh_process" in results:
        checks["H1b_cross_process_system_prompt_has_preference"] = results["read_phase_fresh_process"]["system_prompt_contains_preference"]
    if "control_phase" in results:
        checks["H2_control_no_preference"] = not results["control_phase"]["system_prompt_contains_preference"]
    if "state_backend_phase" in results:
        checks["H3_same_thread_sees_file"] = results["state_backend_phase"]["thread_E_same_thread"]["same_thread_sees_file"]
        checks["H3_fresh_thread_blind"] = not results["state_backend_phase"]["thread_F_fresh"]["fresh_thread_sees_file"]

    summary = {
        "experiment": "task6-E1-memory",
        "at": now_iso(),
        "env": env_record(),
        "phase": args.phase,
        "preference_sha256": sha256_text(PREFERENCE),
        "checks": checks,
        "all_passed": all(checks.values()),
        "results": results,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"checks": checks, "all_passed": summary["all_passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())