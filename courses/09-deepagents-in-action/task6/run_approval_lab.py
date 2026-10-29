"""Task6 E2/E3/E4: tool approval -- approve executes, reject does not, edit runs edited args.

Hypotheses (see experiment-plan.json):
  H4 (Lite, approve): with interrupt_on={"write_file": True}, the graph pauses
     BEFORE the tool runs; resuming with {"type":"approve"} executes exactly the
     proposed args; the file lands on disk.
  H5 (Basic, reject): resuming with {"type":"reject"} does NOT execute the tool;
     no file appears; the model is told the call was not executed.
  H6 (Advanced, edit): resuming with {"type":"edit","edited_action":...} executes
     ONLY the edited args; disk content equals the edited content, not the
     originally proposed content.

Each case runs in a FRESH backend root and FRESH thread so disk effects are
unambiguous. Evidence includes the interrupt payload the human actually sees,
the pre-resume disk state (proof the gate is pre-execution), and post-resume
disk bytes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_common import Recorder, ScriptedModel, env_record, now_iso, write_json

PROPOSED_CONTENT = "PROPOSED-CONTENT-原始提议"
EDITED_CONTENT = "EDITED-CONTENT-修改后参数"
TARGET = "/reports/report.md"


def build_agent(model, backend, checkpointer):
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        backend=backend,
        checkpointer=checkpointer,
        interrupt_on={"write_file": True},
    )


def interrupt_payload(result) -> list:
    payload = []
    for item in result.get("__interrupt__", []) or []:
        value = getattr(item, "value", item)
        payload.append(value if isinstance(value, (dict, list)) else str(value))
    return payload


def disk_state(fs_root: Path) -> dict:
    reports = fs_root / "reports"
    if not reports.exists():
        return {"exists": False}
    files = {str(p.relative_to(fs_root)).replace("\\", "/"): p.read_text(encoding="utf-8") for p in reports.rglob("*") if p.is_file()}
    return {"exists": True, "files": files}


def run_case(name: str, decision: dict, fs_root: Path) -> dict:
    from deepagents.backends import FilesystemBackend
    from langgraph.types import Command
    from langgraph.checkpoint.memory import MemorySaver

    if fs_root.exists():
        shutil.rmtree(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    checkpointer = MemorySaver()
    thread_id = f"approval-{name}"

    model = ScriptedModel(
        steps=[
            ("write_file", {"file_path": TARGET, "content": PROPOSED_CONTENT}),
            ("final", f"案例 {name} 结束。"),
        ],
        tag=f"approval-{name}",
    )
    agent = build_agent(model, backend, checkpointer)
    rec = Recorder()

    # 1) model proposes the write; graph must pause BEFORE execution
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "把报告写到 /reports/report.md"}]},
        config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
    )
    payload = interrupt_payload(result)
    pre_disk = disk_state(fs_root)

    # 2) human decision
    resumed = agent.invoke(
        Command(resume={"decisions": [decision]}),
        config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
    )
    post_disk = disk_state(fs_root)
    final_messages = [m for m in resumed["messages"]]

    return {
        "case": name,
        "thread_id": thread_id,
        "decision": decision,
        "interrupted": bool(payload),
        "interrupt_payload": payload,
        "pre_resume_disk": pre_disk,
        "post_resume_disk": post_disk,
        "final_messages": [
            {"type": getattr(m, "type", "?"), "content": getattr(m, "content", None), "name": getattr(m, "name", None)}
            for m in final_messages[-4:]
        ],
        "events": rec.events,
        "requests_count": len(rec.requests),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    root = Path(args.root)
    out.mkdir(parents=True, exist_ok=True)

    cases = {
        "approve": run_case("approve", {"type": "approve"}, root / "fs_approve"),
        "reject": run_case("reject", {"type": "reject", "message": "这份报告先不写"}, root / "fs_reject"),
        "edit": run_case(
            "edit",
            {"type": "edit", "edited_action": {"name": "write_file", "args": {"file_path": TARGET, "content": EDITED_CONTENT}}},
            root / "fs_edit",
        ),
    }

    approve_files = cases["approve"]["post_resume_disk"].get("files", {})
    reject_files = cases["reject"]["post_resume_disk"].get("files", {})
    edit_files = cases["edit"]["post_resume_disk"].get("files", {})

    checks = {
        "H4_interrupted_before_execution": cases["approve"]["interrupted"]
        and not cases["approve"]["pre_resume_disk"].get("files"),
        "H4_approve_executes_proposed_args": approve_files.get(TARGET.lstrip("/")) == PROPOSED_CONTENT,
        "H4_approve_interrupt_payload_names_write_file": any(
            (a.get("name") if isinstance(a, dict) else None) == "write_file"
            for req in cases["approve"]["interrupt_payload"]
            if isinstance(req, dict)
            for a in req.get("action_requests", [])
        ),
        "H5_reject_does_not_execute": not reject_files,
        "H5_reject_still_interrupted": cases["reject"]["interrupted"],
        "H6_edit_executes_only_edited_args": edit_files.get(TARGET.lstrip("/")) == EDITED_CONTENT,
        "H6_edit_not_proposed_content": edit_files.get(TARGET.lstrip("/")) != PROPOSED_CONTENT,
    }

    summary = {
        "experiment": "task6-E234-approval",
        "at": now_iso(),
        "env": env_record(),
        "checks": checks,
        "all_passed": all(checks.values()),
        "cases": cases,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"checks": checks, "all_passed": summary["all_passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
