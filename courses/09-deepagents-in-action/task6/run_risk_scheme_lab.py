"""Task6 risk-scheme lab (Advanced+): verify the tiered approval scheme end-to-end.

Part A (unit): classify_call tier table -- deterministic, no agent.
Part B (e2e): a real graph built from risk_policy.build_policy() is driven by the
scripted model; each case checks the ACTUAL gate behavior:
  S1  L1 public write            -> no interrupt, executed
  S2  L2 draft write             -> interrupt with decisions [approve,edit,reject]
  S3  L3 delete                  -> interrupt with decisions [approve,reject]
  S4  L4 secret write            -> denied, no interrupt
  S5  L4 credential content      -> denied by the content guard, no interrupt
  S6  unclassified write         -> interrupt (fail-closed default)
  S7  L3 secret read             -> interrupt with decisions [approve,reject]
  S8  bulk read of the whole tree-> interrupt (conservative: root overlaps secrets)
  S9  L3 decision-set enforcement-> an "edit" decision on a delete raises ValueError
  S10 audit trail                -> every tool call recorded with tier + reasons
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_common import Recorder, ScriptedModel, env_record, now_iso, write_json

SECRET_FILE = "/secrets/k.txt"
SECRET_CONTENT = "top-secret"


def build_agent(model, backend, checkpointer, audit):
    from deepagents import create_deep_agent
    from risk_policy import build_policy

    policy = build_policy(audit)
    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        backend=backend,
        checkpointer=checkpointer,
        middleware=policy["middleware"],
        permissions=policy["permissions"],
        interrupt_on=policy["interrupt_on"],
    )


def record_approval(approvals: list, case_name: str, payload: list) -> None:
    """Record an approval REQUEST (what the human was shown) for the audit trail."""
    from risk_policy import classify_call

    for req in payload:
        if not isinstance(req, dict):
            continue
        for action in req.get("action_requests", []):
            verdict = classify_call(action.get("name"), action.get("args", {}) or {})
            decisions = None
            for cfg in req.get("review_configs", []):
                if cfg.get("action_name") == action.get("name"):
                    decisions = cfg.get("allowed_decisions")
            approvals.append(
                {
                    "kind": "approval_request",
                    "case": case_name,
                    "tool": action.get("name"),
                    "tier": verdict.tier,
                    "reasons": verdict.reasons,
                    "allowed_decisions": decisions,
                }
            )


def record_decision(approvals: list, case_name: str, decision: dict) -> None:
    approvals.append({"kind": "human_decision", "case": case_name, "decision": decision.get("type"), "message": decision.get("message")})


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


def interrupt_payload(result) -> list:
    payload = []
    for item in result.get("__interrupt__", []) or []:
        value = getattr(item, "value", item)
        payload.append(value if isinstance(value, (dict, list)) else str(value))
    return payload


def decisions_of(payload, tool_name):
    for req in payload:
        if not isinstance(req, dict):
            continue
        for cfg in req.get("review_configs", []):
            if cfg.get("action_name") == tool_name:
                return cfg.get("allowed_decisions")
    return None


def fresh_root(root: Path, name: str, seed_secret: bool = False) -> Path:
    fs_root = root / name
    if fs_root.exists():
        shutil.rmtree(fs_root)
    fs_root.mkdir(parents=True)
    if seed_secret:
        (fs_root / "secrets").mkdir()
        (fs_root / "secrets" / "k.txt").write_text(SECRET_CONTENT, encoding="utf-8")
    return fs_root


def run_simple(name, path, content, fs_root, audit, audit_approvals, extra_steps=None):
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    backend = FilesystemBackend(root_dir=str(fs_root))
    steps = [("write_file", {"file_path": path, "content": content})]
    steps += extra_steps or [("final", f"案例 {name} 结束。")]
    agent = build_agent(ScriptedModel(steps=steps, tag=f"scheme-{name}"), backend, MemorySaver(), audit)
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"写入 {path}"}]},
        config={"configurable": {"thread_id": f"scheme-{name}"}, "callbacks": [rec]},
    )
    payload = interrupt_payload(result)
    if payload:
        record_approval(audit_approvals, name, payload)
    return {
        "case": name,
        "interrupted": bool(payload),
        "interrupt_payload": payload,
        "tool_messages": tool_messages(rec),
        "disk_files": disk_files(fs_root),
        "final_message": result["messages"][-1].content,
    }


def run_delete_case(name, path, fs_root, audit, audit_approvals, decision=None):
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(
        ScriptedModel(
            steps=[("delete", {"file_path": path}), ("final", f"案例 {name} 结束。")],
            tag=f"scheme-{name}",
        ),
        backend,
        MemorySaver(),
        audit,
    )
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"删除 {path}"}]},
        config={"configurable": {"thread_id": f"scheme-{name}"}, "callbacks": [rec]},
    )
    payload = interrupt_payload(result)
    if payload:
        record_approval(audit_approvals, name, payload)
    case = {
        "case": name,
        "interrupted": bool(payload),
        "interrupt_payload": payload,
        "tool_messages": tool_messages(rec),
        "disk_files": disk_files(fs_root),
    }
    if decision is not None:
        record_decision(audit_approvals, name, decision)
        resumed = agent.invoke(
            Command(resume={"decisions": [decision]}),
            config={"configurable": {"thread_id": f"scheme-{name}"}, "callbacks": [rec]},
        )
        case["final_message"] = resumed["messages"][-1].content
        case["post_disk_files"] = disk_files(fs_root)
    return case


def run_read_case(name, path, fs_root, audit, audit_approvals):
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(
        ScriptedModel(steps=[("read_file", {"file_path": path}), ("final", f"案例 {name} 结束。")], tag=f"scheme-{name}"),
        backend,
        MemorySaver(),
        audit,
    )
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"读取 {path}"}]},
        config={"configurable": {"thread_id": f"scheme-{name}"}, "callbacks": [rec]},
    )
    payload = interrupt_payload(result)
    if payload:
        record_approval(audit_approvals, name, payload)
    return {
        "case": name,
        "interrupted": bool(payload),
        "interrupt_payload": payload,
        "tool_messages": tool_messages(rec),
    }


def run_bulk_case(name, tool_name, args, fs_root, audit, audit_approvals):
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    backend = FilesystemBackend(root_dir=str(fs_root))
    agent = build_agent(
        ScriptedModel(steps=[(tool_name, args), ("final", f"案例 {name} 结束。")], tag=f"scheme-{name}"),
        backend,
        MemorySaver(),
        audit,
    )
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "列一下目录"}]},
        config={"configurable": {"thread_id": f"scheme-{name}"}, "callbacks": [rec]},
    )
    payload = interrupt_payload(result)
    if payload:
        record_approval(audit_approvals, name, payload)
    return {"case": name, "interrupted": bool(payload), "interrupt_payload": payload}


def denied(case) -> bool:
    return any(
        "denied" in (m.get("content") or "").lower()
        or "blocked by risk policy" in (m.get("content") or "").lower()
        for m in case["tool_messages"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    root = Path(args.root)
    out.mkdir(parents=True, exist_ok=True)

    from risk_policy import classify_call

    # ---------- Part A: unit tier table ----------
    unit_cases = [
        ("ls", {"path": "/public"}, "L0"),
        ("read_file", {"file_path": "/public/a.md"}, "L0"),
        ("read_file", {"file_path": "/secrets/k.txt"}, "L3"),
        ("grep", {"path": None}, "L3"),
        ("glob", {"path": "/public", "pattern": "/secrets/**"}, "L3"),
        ("ls", {"path": "/"}, "L3"),  # root overlaps secrets: conservative
        ("write_file", {"file_path": "/public/a.md", "content": "x"}, "L1"),
        ("write_file", {"file_path": "/drafts/b.md", "content": "x"}, "L2"),
        ("write_file", {"file_path": "/other/x.md", "content": "x"}, "L2"),
        ("write_file", {"file_path": "/secrets/c.md", "content": "x"}, "L4"),
        ("write_file", {"file_path": "/public/p.md", "content": "api_key = sk-abcdef1234567890abcdef"}, "L4"),
        ("edit_file", {"file_path": "/drafts/d.md", "old_string": "a", "new_string": "b"}, "L2"),
        ("delete", {"file_path": "/notes/t.md"}, "L3"),
        ("delete", {"file_path": "/secrets/k.txt"}, "L4"),
        ("weird_tool", {"x": 1}, "L2"),
    ]
    unit_results = []
    for tool, tool_args, want in unit_cases:
        verdict = classify_call(tool, tool_args)
        unit_results.append({"tool": tool, "args": tool_args, "want": want, "got": verdict.tier, "verdict": verdict.as_dict()})

    # ---------- Part B: end-to-end ----------
    audit: list = []          # execution-phase verdicts (RiskGateMiddleware)
    approvals: list = []      # approval requests + human decisions
    cases = {}

    cases["S1_public_write"] = run_simple("S1_public_write", "/public/note.md", "PUBLIC", fresh_root(root, "s1"), audit, approvals)
    cases["S2_draft_write"] = run_simple("S2_draft_write", "/drafts/d.md", "DRAFT", fresh_root(root, "s2"), audit, approvals)
    # S3: seed the todo file first so reject-vs-execute is observable on disk
    s3_root = fresh_root(root, "s3", seed_secret=True)
    (s3_root / "notes").mkdir(parents=True, exist_ok=True)
    (s3_root / "notes" / "t.md").write_text("todo", encoding="utf-8")
    cases["S3_delete"] = run_delete_case("S3_delete", "/notes/t.md", s3_root, audit, approvals,
                                         decision={"type": "reject", "message": "先别删"})
    cases["S4_secret_write"] = run_simple("S4_secret_write", "/secrets/s.md", "SECRET", fresh_root(root, "s4"), audit, approvals)
    cases["S5_credential_content"] = run_simple(
        "S5_credential_content", "/public/p.md", "api_key = sk-abcdef1234567890abcdef", fresh_root(root, "s5"), audit, approvals
    )
    cases["S6_unclassified_write"] = run_simple("S6_unclassified_write", "/other/x.md", "X", fresh_root(root, "s6"), audit, approvals)
    cases["S7_secret_read"] = run_read_case("S7_secret_read", "/secrets/k.txt", fresh_root(root, "s7", seed_secret=True), audit, approvals)
    cases["S8_bulk_root"] = run_bulk_case("S8_bulk_root", "ls", {"path": "/"}, fresh_root(root, "s8", seed_secret=True), audit, approvals)

    # S9: an edit decision on an L3 delete must be rejected by the decision set
    s9_error = None
    try:
        run_delete_case("S9_edit_on_delete", "/notes/t.md", fresh_root(root, "s9"), audit, approvals,
                        decision={"type": "edit", "edited_action": {"name": "delete", "args": {"file_path": "/notes/t.md"}}})
    except ValueError as exc:
        s9_error = str(exc)
    except Exception as exc:  # noqa: BLE001
        s9_error = f"{type(exc).__name__}: {exc}"
    cases["S9_edit_on_delete"] = {"case": "S9_edit_on_delete", "error": s9_error}

    checks = {
        "A_unit_tier_table": all(u["got"] == u["want"] for u in unit_results),
        "S1_L1_public_no_interrupt_executed": (
            not cases["S1_public_write"]["interrupted"] and cases["S1_public_write"]["disk_files"].get("public/note.md") == "PUBLIC"
        ),
        "S2_L2_draft_interrupt_decisions": (
            cases["S2_draft_write"]["interrupted"]
            and decisions_of(cases["S2_draft_write"]["interrupt_payload"], "write_file") == ["approve", "edit", "reject"]
        ),
        "S3_L3_delete_interrupt_decisions_and_reject_keeps_file": (
            cases["S3_delete"]["interrupted"]
            and decisions_of(cases["S3_delete"]["interrupt_payload"], "delete") == ["approve", "reject"]
            and cases["S3_delete"].get("post_disk_files", {}).get("notes/t.md") == "todo"
        ),
        "S4_L4_secret_write_denied_no_interrupt": (
            not cases["S4_secret_write"]["interrupted"] and denied(cases["S4_secret_write"]) and not cases["S4_secret_write"]["disk_files"]
        ),
        "S5_L4_credential_content_denied_by_guard": (
            not cases["S5_credential_content"]["interrupted"]
            and any(
                rec.get("outcome") == "denied-by-content-guard" and rec.get("tier") == "L4"
                for rec in audit
                if rec.get("tool") == "write_file"
            )
            and not cases["S5_credential_content"]["disk_files"]
        ),
        "S6_unclassified_fails_closed_to_interrupt": cases["S6_unclassified_write"]["interrupted"],
        "S7_L3_secret_read_interrupt_decisions": (
            cases["S7_secret_read"]["interrupted"]
            and decisions_of(cases["S7_secret_read"]["interrupt_payload"], "read_file") == ["approve", "reject"]
        ),
        "S8_bulk_root_interrupts": cases["S8_bulk_root"]["interrupted"],
        "S9_edit_decision_on_L3_rejected": bool(cases["S9_edit_on_delete"]["error"]) and "not allowed" in cases["S9_edit_on_delete"]["error"],
        "S10_audit_trail_complete": (
            # (a) execution-phase verdicts recorded with tier + reasons
            len(audit) >= 3
            and all(rec.get("tier") and rec.get("reasons") for rec in audit)
            # (b) every interrupted case produced an approval REQUEST record
            and {a["case"] for a in approvals if a["kind"] == "approval_request"}
            >= {"S2_draft_write", "S3_delete", "S7_secret_read", "S8_bulk_root"}
            # (c) the human decision on S3 is recorded
            and any(a["kind"] == "human_decision" and a["case"] == "S3_delete" and a["decision"] == "reject" for a in approvals)
        ),
    }

    summary = {
        "experiment": "task6-risk-scheme",
        "at": now_iso(),
        "env": env_record(),
        "checks": checks,
        "all_passed": all(checks.values()),
        "unit_results": unit_results,
        "cases": cases,
        "audit_execution": audit,
        "audit_approvals": approvals,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"checks": checks, "all_passed": summary["all_passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
