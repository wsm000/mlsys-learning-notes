"""Task6 live experiments: real model, end-to-end memory and approval.

Separate denominator from the offline scripted labs. Runs on the remote main-lock
environment (vm-60, deepagents 0.7.8) against DeepSeek-V4-Flash (AMD endpoint)
or a local ollama model. Credentials are read from a key file and never recorded.

Cases:
  L1 (write):  real user asks the agent to remember a preference into
              /memory/AGENTS.md; verify the file on disk.
  L2 (read):   FRESH PROCESS + fresh thread; ask the preference; verify the
              answer uses the memory AND the recorded system prompt carried it.
  L3 (reject): delete /notes/todo.md -> interrupt -> reject -> file survives.
  L4 (approve): same -> approve -> file is gone.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_common import Recorder, env_record, now_iso, sha256_text, system_prompt_text, write_json

MEMORY_PATH = "/memory/AGENTS.md"
TODO_PATH = "/notes/todo.md"
TODO_CONTENT = "buy milk"


def build_model(args):
    from langchain_openai import ChatOpenAI

    if args.model == "deepseek":
        key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
        return ChatOpenAI(
            model="DeepSeek-V4-Flash",
            base_url="https://developer.amd.com.cn/radeon/api/v1",
            api_key=key,
            temperature=0,
            timeout=args.timeout,
            max_retries=0,
            max_tokens=2500,
        )
    return ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or "ollama",
        temperature=0,
        timeout=args.timeout,
        max_retries=0,
        max_tokens=2500,
    )


def build_agent(model, backend, checkpointer, interrupt_on=None):
    from deepagents import create_deep_agent

    kwargs = {}
    if interrupt_on is not None:
        kwargs["interrupt_on"] = interrupt_on
    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        memory=[MEMORY_PATH],
        backend=backend,
        checkpointer=checkpointer,
        **kwargs,
    )


def usage_of(rec) -> dict:
    total = {"input": 0, "output": 0, "total": 0}
    for resp in rec.responses:
        usage = (resp.get("message") or {}).get("usage_metadata") or {}
        for k in ("input", "output", "total"):
            total[k] += usage.get(f"{k}_tokens", 0) or 0
    return total


def run_write(args) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = Path(args.root) / "live-fs"
    if fs_root.exists():
        shutil.rmtree(fs_root)
    # FilesystemBackend does not create root_dir; read-only tools (ls) return
    # path_not_found until a write creates the tree. Create it explicitly.
    fs_root.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(build_model(args), backend, MemorySaver())
    rec = Recorder(budget=12)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "记住：我喜欢中文回答，回答要简洁。请调用 write_file 把这个偏好写进 /memory/AGENTS.md。"}]},
        config={"configurable": {"thread_id": "live-L1"}, "callbacks": [rec]},
    )
    mem_file = fs_root / "memory" / "AGENTS.md"
    disk_text = mem_file.read_text(encoding="utf-8") if mem_file.exists() else None
    return {
        "case": "L1_write",
        "thread_id": "live-L1",
        "final_message": result["messages"][-1].content,
        "disk_file_text": disk_text,
        "model_calls": len(rec.requests),
        "usage": usage_of(rec),
        "requests": rec.requests,
        "events": rec.events,
    }


def run_read(args) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    fs_root = Path(args.root) / "live-fs"
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(build_model(args), backend, MemorySaver())
    rec = Recorder(budget=12)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么语言的回答？请根据你的记忆回答，并说明依据。"}]},
        config={"configurable": {"thread_id": "live-L2-fresh-process"}, "callbacks": [rec]},
    )
    prompt = system_prompt_text(rec.requests[0]["messages"]) if rec.requests else ""
    final = result["messages"][-1].content or ""
    return {
        "case": "L2_read_fresh_process",
        "thread_id": "live-L2-fresh-process",
        "final_message": final,
        "answer_mentions_chinese": "中文" in final,
        "system_prompt_has_memory": "中文" in prompt,
        "model_calls": len(rec.requests),
        "usage": usage_of(rec),
        "requests": rec.requests,
    }


def run_delete(args, decision: dict, case: str, thread_id: str) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    fs_root = Path(args.root) / ("live-fs-" + case)
    if fs_root.exists():
        shutil.rmtree(fs_root)
    fs_root.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=str(fs_root))
    todo = fs_root / "notes" / "todo.md"
    todo.parent.mkdir(parents=True, exist_ok=True)
    todo.write_text(TODO_CONTENT, encoding="utf-8")

    agent = build_agent(build_model(args), backend, MemorySaver(), interrupt_on={"delete": True})
    rec = Recorder(budget=12)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "把 /notes/todo.md 删掉。"}]},
        config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
    )
    interrupted = bool(result.get("__interrupt__"))
    pre_exists = todo.exists()
    resumed = agent.invoke(
        Command(resume={"decisions": [decision]}),
        config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
    )
    return {
        "case": case,
        "thread_id": thread_id,
        "decision": decision,
        "interrupted": interrupted,
        "file_existed_before_resume": pre_exists,
        "file_exists_after_resume": todo.exists(),
        "final_message": resumed["messages"][-1].content,
        "model_calls": len(rec.requests),
        "usage": usage_of(rec),
        "requests": rec.requests,
        "events": rec.events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key-file", default="/home/simin/codex-task1-20261013/.mimo-key")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--phase", choices=["write", "read", "reject", "approve", "all"], default="all")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.phase in ("write", "all"):
        results["L1_write"] = run_write(args)
    if args.phase in ("read", "all"):
        results["L2_read"] = run_read(args)
    if args.phase in ("reject", "all"):
        results["L3_reject"] = run_delete(args, {"type": "reject", "message": "这个文件还有用，先别删"}, "reject", "live-L3")
    if args.phase in ("approve", "all"):
        results["L4_approve"] = run_delete(args, {"type": "approve"}, "approve", "live-L4")

    checks = {}
    if "L1_write" in results:
        text = results["L1_write"]["disk_file_text"] or ""
        checks["L1_memory_file_written"] = "中文" in text
    if "L2_read" in results:
        checks["L2_answer_uses_memory"] = results["L2_read"]["answer_mentions_chinese"]
        checks["L2_memory_in_system_prompt"] = results["L2_read"]["system_prompt_has_memory"]
    if "L3_reject" in results:
        c = results["L3_reject"]
        checks["L3_reject_keeps_file"] = c["interrupted"] and c["file_existed_before_resume"] and c["file_exists_after_resume"]
    if "L4_approve" in results:
        c = results["L4_approve"]
        checks["L4_approve_deletes_file"] = c["interrupted"] and c["file_existed_before_resume"] and not c["file_exists_after_resume"]

    summary = {
        "experiment": "task6-live",
        "at": now_iso(),
        "env": env_record(),
        "model": args.model,
        "checks": checks,
        "all_passed": all(checks.values()) if checks else None,
        "results": results,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"checks": checks, "all_passed": summary["all_passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())