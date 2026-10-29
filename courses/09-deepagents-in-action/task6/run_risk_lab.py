"""Task6 E6 (Advanced): risk-tiered filesystem permissions and their boundaries.

Tier model (declaration order, first match wins):
  L0 read   /**           allow    (reads are free)
  L1 write  /public/**    allow    (low-risk writes auto-execute)
  L2 write  /drafts/**    interrupt(medium risk: human approval gate)
  L3 write  /secrets/**   deny     (high risk: hard block, not approvable)

Hypotheses (see experiment-plan.json):
  H8:  /public write executes with NO interrupt.
  H9:  /drafts write interrupts before execution; approve -> executes.
  H10: /secrets write is denied by the tool itself with NO interrupt.
  H10b: a deny rule declared BEFORE an interrupt rule wins for the overlapping
        path (first match in declaration order).
  H11 (boundary): an "edit" decision that redirects an interrupted write into a
        denied path is STILL denied -- edited calls re-enter the tool and hit
        its pre-execution deny check.
  H12 (boundary): direct backend.write() to a denied path SUCCEEDS -- permissions
        are enforced at the FilesystemMiddleware tool level, not the backend level.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_common import Recorder, ScriptedModel, env_record, now_iso, write_json

RULES = [
    {"operations": ["read"], "paths": ["/**"], "mode": "allow"},
    {"operations": ["write"], "paths": ["/public/**"], "mode": "allow"},
    {"operations": ["write"], "paths": ["/drafts/**"], "mode": "interrupt"},
    {"operations": ["write"], "paths": ["/secrets/**"], "mode": "deny"},
]


def build_agent(model, backend, checkpointer, rules):
    from deepagents import create_deep_agent
    from deepagents.middleware.filesystem import FilesystemPermission

    permissions = [FilesystemPermission(**r) for r in rules]
    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        backend=backend,
        checkpointer=checkpointer,
        permissions=permissions,
    )


def disk_files(fs_root: Path) -> dict:
    if not fs_root.exists():
        return {}
    return {
        str(p.relative_to(fs_root)).replace("\\", "/"): p.read_text(encoding="utf-8")
        for p in fs_root.rglob("*")
        if p.is_file()
    }


def tool_messages(rec) -> list:
    out = []
    for e in rec.events:
        if e["event"] == "tool_end" and isinstance(e.get("output"), dict) and e["output"].get("type") == "tool":
            out.append(e["output"])
    return out


def run_simple(name: str, path: str, content: str, fs_root: Path, rules=RULES) -> dict:
    """One-shot case: model proposes a write; observe interrupt/deny/execute."""
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    if fs_root.exists():
        shutil.rmtree(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(
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
    return {
        "case": name,
        "path": path,
        "interrupted": bool(result.get("__interrupt__")),
        "tool_messages": tool_messages(rec),
        "disk_files": disk_files(fs_root),
        "final_message": result["messages"][-1].content,
    }


def run_interrupt_then(name: str, path: str, content: str, decision: dict, fs_root: Path, rules=RULES) -> dict:
    """Interrupt case with a resume decision (approve or edit)."""
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    if fs_root.exists():
        shutil.rmtree(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(
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
    return {
        "case": name,
        "path": path,
        "decision": decision,
        "interrupted": bool(result.get("__interrupt__")),
        "pre_resume_disk": pre_disk,
        "tool_messages": tool_messages(rec),
        "disk_files": disk_files(fs_root),
        "final_message": resumed["messages"][-1].content,
    }


def run_direct_backend_case(fs_root: Path) -> dict:
    """H12: permissions are tool-level; a direct backend call is not gated."""
    from deepagents.backends import FilesystemBackend

    if fs_root.exists():
        shutil.rmtree(fs_root)
    backend = FilesystemBackend(root_dir=str(fs_root))
    result = backend.write("/secrets/direct.md", "DIRECT-BACKEND-WRITE")
    payload = {
        "result_type": type(result).__name__,
        "error": getattr(result, "error", None),
        "path": getattr(result, "path", None),
    }
    return {"case": "direct_backend", "write_result": payload, "disk_files": disk_files(fs_root)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    root = Path(args.root)
    out.mkdir(parents=True, exist_ok=True)

    cases = {}

    # L1: public write auto-executes
    cases["L1_public_allow"] = run_simple("L1_public_allow", "/public/note.md", "PUBLIC", root / "fs_public")

    # L2: drafts write interrupts; approve -> executes
    cases["L2_drafts_interrupt_approve"] = run_interrupt_then(
        "L2_drafts_interrupt_approve", "/drafts/d.md", "DRAFT", {"type": "approve"}, root / "fs_drafts"
    )

    # L3: secrets write denied, no interrupt
    cases["L3_secrets_deny"] = run_simple("L3_secrets_deny", "/secrets/s.md", "SECRET", root / "fs_secrets")

    # Order: deny declared before interrupt for an overlapping path
    ordered_rules = [
        {"operations": ["read"], "paths": ["/**"], "mode": "allow"},
        {"operations": ["write"], "paths": ["/drafts/blocked/**"], "mode": "deny"},
        {"operations": ["write"], "paths": ["/drafts/**"], "mode": "interrupt"},
    ]
    cases["L3b_deny_before_interrupt"] = run_simple(
        "L3b_deny_before_interrupt", "/drafts/blocked/x.md", "BLOCKED", root / "fs_blocked", rules=ordered_rules
    )

    # Boundary: edit redirects an interrupted write into a denied path
    cases["H11_edit_redirect_into_deny"] = run_interrupt_then(
        "H11_edit_redirect_into_deny",
        "/drafts/ok.md",
        "DRAFT-OK",
        {"type": "edit", "edited_action": {"name": "write_file", "args": {"file_path": "/secrets/evil.md", "content": "EVIL"}}},
        root / "fs_edit_deny",
    )

    # Boundary: direct backend write bypasses tool-level permissions
    cases["H12_direct_backend"] = run_direct_backend_case(root / "fs_direct")

    def denied(case) -> bool:
        return any(
            "denied" in (m.get("content") or "").lower() or "permission" in (m.get("content") or "").lower()
            for m in case["tool_messages"]
        )

    checks = {
        "H8_public_no_interrupt_and_executed": (
            not cases["L1_public_allow"]["interrupted"] and cases["L1_public_allow"]["disk_files"].get("public/note.md") == "PUBLIC"
        ),
        "H9_drafts_interrupts_then_approve_executes": (
            cases["L2_drafts_interrupt_approve"]["interrupted"]
            and not cases["L2_drafts_interrupt_approve"]["pre_resume_disk"]
            and cases["L2_drafts_interrupt_approve"]["disk_files"].get("drafts/d.md") == "DRAFT"
        ),
        "H10_secrets_denied_no_interrupt": (
            not cases["L3_secrets_deny"]["interrupted"]
            and denied(cases["L3_secrets_deny"])
            and not cases["L3_secrets_deny"]["disk_files"]
        ),
        "H10b_deny_before_interrupt_wins": (
            not cases["L3b_deny_before_interrupt"]["interrupted"]
            and denied(cases["L3b_deny_before_interrupt"])
            and not cases["L3b_deny_before_interrupt"]["disk_files"]
        ),
        "H11_edit_into_deny_still_denied": (
            cases["H11_edit_redirect_into_deny"]["interrupted"]
            and denied(cases["H11_edit_redirect_into_deny"])
            and not cases["H11_edit_redirect_into_deny"]["disk_files"]
        ),
        "H12_direct_backend_not_gated": (
            cases["H12_direct_backend"]["write_result"]["error"] is None
            and cases["H12_direct_backend"]["disk_files"].get("secrets/direct.md") == "DIRECT-BACKEND-WRITE"
        ),
    }

    summary = {
        "experiment": "task6-E6-risk-tiers",
        "at": now_iso(),
        "env": env_record(),
        "rules": RULES,
        "ordered_rules_variant": ordered_rules,
        "checks": checks,
        "all_passed": all(checks.values()),
        "cases": cases,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"checks": checks, "all_passed": summary["all_passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
