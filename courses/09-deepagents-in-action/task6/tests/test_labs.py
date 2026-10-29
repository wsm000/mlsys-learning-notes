"""Pytest regression tests for the Task6 labs (course 09, deepagents-in-action).

Task6 covers long-term memory (ch08) and human-in-the-loop approval (ch09) of
the DeepAgents SDK. These tests import the four frozen evidence-producing
runners -- run_memory_lab, run_approval_lab, run_store_lab, run_risk_lab --
re-run their experiment functions, and assert exactly the checks each runner
asserts in its own main() summary. Check names are mirrored 1:1 (test_h1_...
mirrors check H1_...), so a green pytest run and a green runner summary mean
the same thing. A few extra negative/control assertions are marked NEGATIVE.

Host-sandbox discipline (inherited from Task5/Task6):
  * every backend root and evidence directory lives under a TEMP directory --
    subprocesses on this host cannot write inside the workspace;
  * tempfile.mkdtemp() creates 0o700 directories whose explicit DACL drops the
    sandbox write grant on this host (verified: writing inside such a
    directory raises PermissionError while os.mkdir()-created directories
    inherit the grant), so make_temp_dir() tries mkdtemp first and falls back
    to a default-mode directory under the system temp root. Writability is
    probe-verified either way, so the helper stays portable.

Recommended invocation (from the workspace root):
  <venv>/Scripts/python.exe -m pytest courses/09-deepagents-in-action/task6/tests/ -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import pytest

TASK6_DIR = Path(__file__).resolve().parent.parent
if str(TASK6_DIR) not in sys.path:
    sys.path.insert(0, str(TASK6_DIR))

import lab_common
import run_approval_lab
import run_memory_lab
import run_risk_lab
import run_store_lab

# --------------------------------------------------------------------------
# TEMP-directory helper (never writes inside the workspace)
# --------------------------------------------------------------------------

_CREATED_TEMP_DIRS: list[Path] = []


def _probe_writable(path: Path) -> None:
    probe = path / ".task6-write-probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def make_temp_dir(prefix: str) -> Path:
    """Return a fresh, writable TEMP directory (never inside the workspace)."""
    try:
        candidate = Path(tempfile.mkdtemp(prefix=prefix))
        _probe_writable(candidate)
        _CREATED_TEMP_DIRS.append(candidate)
        return candidate
    except OSError:
        # mkdtemp's 0o700 DACL is unwritable under this host sandbox.
        pass
    base = Path(tempfile.gettempdir())
    for index in range(10_000):
        candidate = base / f"{prefix}{os.getpid()}-{index:04d}"
        try:
            candidate.mkdir()  # default mode -> inherits the sandbox temp ACLs
        except FileExistsError:
            continue
        try:
            _probe_writable(candidate)
        except OSError:
            continue
        _CREATED_TEMP_DIRS.append(candidate)
        return candidate
    raise RuntimeError(f"could not create a writable temp directory under {base}")


@pytest.fixture(scope="session", autouse=True)
def _cleanup_temp_dirs():
    yield
    for path in reversed(_CREATED_TEMP_DIRS):
        shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------
# E1: cross-thread file memory (run_memory_lab)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def memory_root() -> Path:
    return make_temp_dir("task6-memory-")


@pytest.fixture(scope="module")
def memory_write_phase(memory_root) -> dict:
    """Thread A writes /memory/AGENTS.md; FRESH thread B reads it back."""
    return run_memory_lab.run_write_phase(memory_root, memory_root / "out-write")


@pytest.fixture(scope="module")
def memory_read_phase(memory_root, memory_write_phase) -> dict:
    """Fresh agent instance over the SAME on-disk backend root (H1b)."""
    return run_memory_lab.run_read_phase(memory_root, memory_root / "out-read")


@pytest.fixture(scope="module")
def memory_control_phase() -> dict:
    """Control: empty backend root -> no memory file, no preference."""
    root = make_temp_dir("task6-memory-control-")
    return run_memory_lab.run_control_phase(root, root / "out")


@pytest.fixture(scope="module")
def memory_state_phase() -> dict:
    """H3: default StateBackend -- same thread remembers, fresh thread does not."""
    root = make_temp_dir("task6-memory-state-")
    return run_memory_lab.run_state_backend_phase(root, root / "out")


class TestMemoryLab:
    def test_h1_disk_file_written(self, memory_write_phase):
        record = memory_write_phase["thread_A_write"]
        assert record["disk_matches_preference"] is True
        assert record["disk_file_text"] == run_memory_lab.PREFERENCE

    def test_h1_fresh_thread_system_prompt_has_preference(self, memory_write_phase):
        record = memory_write_phase["thread_B_fresh_read"]
        assert record["system_prompt_contains_preference"] is True
        # Re-derive from the RECORDED request bytes, exactly as the runner does.
        prompt = lab_common.system_prompt_text(record["requests"][0]["messages"])
        assert run_memory_lab.PREFERENCE in prompt

    def test_h1_fresh_thread_answer_uses_memory(self, memory_write_phase):
        record = memory_write_phase["thread_B_fresh_read"]
        assert "中文" in record["final_message"]

    def test_h1b_cross_process_system_prompt_has_preference(self, memory_read_phase):
        assert memory_read_phase["system_prompt_contains_preference"] is True
        assert memory_read_phase["thread_id"] == "mem-C-fresh-process"

    def test_h2_control_no_preference(self, memory_control_phase):
        # NEGATIVE control: without the memory file the prompt carries no preference.
        assert memory_control_phase["system_prompt_contains_preference"] is False
        assert run_memory_lab.PREFERENCE not in memory_control_phase["system_prompt_excerpt"]

    def test_h2_control_answer_reports_no_memory(self, memory_control_phase):
        # NEGATIVE control: the scripted reader reports "no memory" without the file.
        assert memory_control_phase["final_message"] == "我没有关于你的记忆。"

    def test_h3_same_thread_sees_file(self, memory_state_phase):
        record = memory_state_phase["thread_E_same_thread"]
        assert record["same_thread_sees_file"] is True

    def test_h3_fresh_thread_blind(self, memory_state_phase):
        # NEGATIVE: StateBackend state is thread-scoped; a fresh thread is blind.
        record = memory_state_phase["thread_F_fresh"]
        assert record["fresh_thread_sees_file"] is False
        assert "pref.md" not in json.dumps(record["ls_events_fresh_thread"], ensure_ascii=False)
        assert record["final_message"] == "新线程里我没有看到任何文件。"

    def test_memory_lab_all_checks_pass(
        self, memory_write_phase, memory_read_phase, memory_control_phase, memory_state_phase
    ):
        # Mirror of run_memory_lab.main()'s own check computation.
        checks = {
            "H1_disk_file_written": memory_write_phase["thread_A_write"]["disk_matches_preference"],
            "H1_fresh_thread_system_prompt_has_preference": memory_write_phase["thread_B_fresh_read"]["system_prompt_contains_preference"],
            "H1_fresh_thread_answer_uses_memory": "中文" in memory_write_phase["thread_B_fresh_read"]["final_message"],
            "H1b_cross_process_system_prompt_has_preference": memory_read_phase["system_prompt_contains_preference"],
            "H2_control_no_preference": not memory_control_phase["system_prompt_contains_preference"],
            "H3_same_thread_sees_file": memory_state_phase["thread_E_same_thread"]["same_thread_sees_file"],
            "H3_fresh_thread_blind": not memory_state_phase["thread_F_fresh"]["fresh_thread_sees_file"],
        }
        assert all(checks.values()), checks


# --------------------------------------------------------------------------
# E2/E3/E4: tool approval (run_approval_lab)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def approval_root() -> Path:
    return make_temp_dir("task6-approval-")


@pytest.fixture(scope="module")
def approval_approve(approval_root) -> dict:
    """H4 (Lite): approve executes exactly the proposed args."""
    return run_approval_lab.run_case("approve", {"type": "approve"}, approval_root / "fs_approve")


@pytest.fixture(scope="module")
def approval_reject(approval_root) -> dict:
    """H5 (Basic): reject does NOT execute the tool."""
    return run_approval_lab.run_case(
        "reject", {"type": "reject", "message": "这份报告先不写"}, approval_root / "fs_reject"
    )


@pytest.fixture(scope="module")
def approval_edit(approval_root) -> dict:
    """H6 (Advanced): edit runs ONLY the edited args."""
    return run_approval_lab.run_case(
        "edit",
        {
            "type": "edit",
            "edited_action": {
                "name": "write_file",
                "args": {"file_path": run_approval_lab.TARGET, "content": run_approval_lab.EDITED_CONTENT},
            },
        },
        approval_root / "fs_edit",
    )


class TestApprovalLab:
    def test_h4_interrupted_before_execution(self, approval_approve):
        assert approval_approve["interrupted"] is True
        assert not approval_approve["pre_resume_disk"].get("files")

    def test_h4_approve_executes_proposed_args(self, approval_approve):
        files = approval_approve["post_resume_disk"].get("files", {})
        assert files.get(run_approval_lab.TARGET.lstrip("/")) == run_approval_lab.PROPOSED_CONTENT

    def test_h4_approve_interrupt_payload_names_write_file(self, approval_approve):
        assert any(
            (action.get("name") if isinstance(action, dict) else None) == "write_file"
            for request in approval_approve["interrupt_payload"]
            if isinstance(request, dict)
            for action in request.get("action_requests", [])
        )

    def test_h5_reject_still_interrupted(self, approval_reject):
        assert approval_reject["interrupted"] is True

    def test_h5_reject_does_not_execute(self, approval_reject):
        # NEGATIVE: a rejected call must not create the report on disk.
        assert approval_reject["post_resume_disk"].get("exists") is False
        assert not approval_reject["post_resume_disk"].get("files")

    def test_h6_edit_executes_only_edited_args(self, approval_edit):
        files = approval_edit["post_resume_disk"].get("files", {})
        assert files.get(run_approval_lab.TARGET.lstrip("/")) == run_approval_lab.EDITED_CONTENT

    def test_h6_edit_not_proposed_content(self, approval_edit):
        # NEGATIVE: the originally proposed content must not be what lands on disk.
        files = approval_edit["post_resume_disk"].get("files", {})
        assert files.get(run_approval_lab.TARGET.lstrip("/")) != run_approval_lab.PROPOSED_CONTENT

    def test_approval_lab_all_checks_pass(self, approval_approve, approval_reject, approval_edit):
        # Mirror of run_approval_lab.main()'s own check computation.
        approve_files = approval_approve["post_resume_disk"].get("files", {})
        reject_files = approval_reject["post_resume_disk"].get("files", {})
        edit_files = approval_edit["post_resume_disk"].get("files", {})
        checks = {
            "H4_interrupted_before_execution": approval_approve["interrupted"]
            and not approval_approve["pre_resume_disk"].get("files"),
            "H4_approve_executes_proposed_args": approve_files.get(run_approval_lab.TARGET.lstrip("/")) == run_approval_lab.PROPOSED_CONTENT,
            "H4_approve_interrupt_payload_names_write_file": any(
                (action.get("name") if isinstance(action, dict) else None) == "write_file"
                for request in approval_approve["interrupt_payload"]
                if isinstance(request, dict)
                for action in request.get("action_requests", [])
            ),
            "H5_reject_does_not_execute": not reject_files,
            "H5_reject_still_interrupted": approval_reject["interrupted"],
            "H6_edit_executes_only_edited_args": edit_files.get(run_approval_lab.TARGET.lstrip("/")) == run_approval_lab.EDITED_CONTENT,
            "H6_edit_not_proposed_content": edit_files.get(run_approval_lab.TARGET.lstrip("/")) != run_approval_lab.PROPOSED_CONTENT,
        }
        assert all(checks.values()), checks


# --------------------------------------------------------------------------
# E5: StoreBackend per-user isolation (run_store_lab)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store_threads() -> dict:
    """Mirror of run_store_lab.main(): one shared InMemoryStore, two users.

    The store lab is purely in-memory (InMemoryStore + StoreBackend), so no
    filesystem backend root is needed; the TEMP discipline still applies to
    every other lab here.
    """
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()

    # alice thread A1: write her preference into the shared store (her namespace)
    a1 = run_store_lab.run_user_turn(
        store,
        "alice",
        "store-A1",
        steps=[
            ("write_file", {"file_path": run_store_lab.MEMORY_PATH, "content": run_store_lab.ALICE_PREF}),
            ("final", "alice 已写入偏好。"),
        ],
        tag="alice-writer",
        user_msg="记住：我喜欢中文回答，回答要简洁。",
    )

    # alice thread A2: FRESH thread, same user -> memory must arrive
    a2 = run_store_lab.run_user_turn(
        store,
        "alice",
        "store-A2",
        steps=[
            run_store_lab.memory_answer_step(
                run_store_lab.ALICE_PREF, "alice 的记忆：中文回答。", "alice 没有记忆。"
            )
        ],
        tag="alice-reader",
        user_msg="我的偏好是什么？",
    )
    a2["system_prompt_has_own_preference"] = (
        run_store_lab.ALICE_PREF in a2["system_prompt_excerpt"]
        or run_store_lab.ALICE_PREF in json.dumps(a2["requests"], ensure_ascii=False)
    )

    # bob thread B1: same store, different user -> must NOT see alice's file
    b1 = run_store_lab.run_user_turn(
        store,
        "bob",
        "store-B1",
        steps=[
            ("ls", {"path": "/"}),
            run_store_lab.memory_answer_step(
                run_store_lab.ALICE_PREF, "bob 看到了 alice 的偏好（不该发生）", "bob 没有看到 alice 的偏好。"
            ),
        ],
        tag="bob-reader",
        user_msg="我的偏好是什么？",
    )
    b1["system_prompt_has_alice_preference"] = run_store_lab.ALICE_PREF in json.dumps(
        b1["requests"], ensure_ascii=False
    )
    ls_outputs = [
        e
        for e in b1["events"]
        if e["event"] == "tool_end" and isinstance(e.get("output"), dict) and e["output"].get("name") == "ls"
    ]
    b1["ls_saw_alice_file"] = "AGENTS.md" in json.dumps(ls_outputs, ensure_ascii=False)

    # bob writes his own preference in his namespace; alice must not see it either
    b2 = run_store_lab.run_user_turn(
        store,
        "bob",
        "store-B2",
        steps=[
            ("write_file", {"file_path": run_store_lab.MEMORY_PATH, "content": run_store_lab.BOB_PREF}),
            ("final", "bob 已写入偏好。"),
        ],
        tag="bob-writer",
        user_msg="记住：I prefer English answers.",
    )
    a3 = run_store_lab.run_user_turn(
        store,
        "alice",
        "store-A3",
        steps=[
            run_store_lab.memory_answer_step(
                run_store_lab.BOB_PREF, "alice 看到了 bob 的偏好（不该发生）", "alice 没有看到 bob 的偏好。"
            )
        ],
        tag="alice-reader-2",
        user_msg="有别人的偏好吗？",
    )
    a3["system_prompt_has_bob_preference"] = run_store_lab.BOB_PREF in json.dumps(
        a3["requests"], ensure_ascii=False
    )

    return {"A1_write": a1, "A2_read": a2, "B1_read": b1, "B2_write": b2, "A3_read": a3}


class TestStoreLab:
    def test_h7a_alice_cross_thread_sees_own_memory(self, store_threads):
        assert store_threads["A2_read"]["system_prompt_has_own_preference"] is True

    def test_h7a_alice_answer_uses_memory(self, store_threads):
        assert "中文" in store_threads["A2_read"]["final_message"]

    def test_h7b_bob_cannot_see_alice_file(self, store_threads):
        # NEGATIVE: namespace isolation between users on a shared store.
        b1 = store_threads["B1_read"]
        assert b1["system_prompt_has_alice_preference"] is False
        assert b1["ls_saw_alice_file"] is False

    def test_h7b_alice_cannot_see_bob_file(self, store_threads):
        # NEGATIVE: the isolation runs both ways.
        assert store_threads["A3_read"]["system_prompt_has_bob_preference"] is False

    def test_store_lab_all_checks_pass(self, store_threads):
        # Mirror of run_store_lab.main()'s own check computation.
        checks = {
            "H7a_alice_cross_thread_sees_own_memory": store_threads["A2_read"]["system_prompt_has_own_preference"],
            "H7a_alice_answer_uses_memory": "中文" in store_threads["A2_read"]["final_message"],
            "H7b_bob_cannot_see_alice_file": not store_threads["B1_read"]["system_prompt_has_alice_preference"]
            and not store_threads["B1_read"]["ls_saw_alice_file"],
            "H7b_alice_cannot_see_bob_file": not store_threads["A3_read"]["system_prompt_has_bob_preference"],
        }
        assert all(checks.values()), checks


# --------------------------------------------------------------------------
# E6: risk-tiered permissions (run_risk_lab)
# --------------------------------------------------------------------------

ORDERED_RULES = [
    {"operations": ["read"], "paths": ["/**"], "mode": "allow"},
    {"operations": ["write"], "paths": ["/drafts/blocked/**"], "mode": "deny"},
    {"operations": ["write"], "paths": ["/drafts/**"], "mode": "interrupt"},
]


def _denied(case: dict) -> bool:
    """Mirror of run_risk_lab.main()'s denied() helper."""
    return any(
        "denied" in (m.get("content") or "").lower() or "permission" in (m.get("content") or "").lower()
        for m in case["tool_messages"]
    )


@pytest.fixture(scope="module")
def risk_root() -> Path:
    return make_temp_dir("task6-risk-")


@pytest.fixture(scope="module")
def risk_cases(risk_root) -> dict:
    cases = {}
    # L1: public write auto-executes
    cases["L1_public_allow"] = run_risk_lab.run_simple(
        "L1_public_allow", "/public/note.md", "PUBLIC", risk_root / "fs_public"
    )
    # L2: drafts write interrupts; approve -> executes
    cases["L2_drafts_interrupt_approve"] = run_risk_lab.run_interrupt_then(
        "L2_drafts_interrupt_approve", "/drafts/d.md", "DRAFT", {"type": "approve"}, risk_root / "fs_drafts"
    )
    # L3: secrets write denied, no interrupt
    cases["L3_secrets_deny"] = run_risk_lab.run_simple(
        "L3_secrets_deny", "/secrets/s.md", "SECRET", risk_root / "fs_secrets"
    )
    # Order: deny declared before interrupt for an overlapping path
    cases["L3b_deny_before_interrupt"] = run_risk_lab.run_simple(
        "L3b_deny_before_interrupt", "/drafts/blocked/x.md", "BLOCKED", risk_root / "fs_blocked", rules=ORDERED_RULES
    )
    # Boundary: edit redirects an interrupted write into a denied path
    cases["H11_edit_redirect_into_deny"] = run_risk_lab.run_interrupt_then(
        "H11_edit_redirect_into_deny",
        "/drafts/ok.md",
        "DRAFT-OK",
        {
            "type": "edit",
            "edited_action": {"name": "write_file", "args": {"file_path": "/secrets/evil.md", "content": "EVIL"}},
        },
        risk_root / "fs_edit_deny",
    )
    # Boundary: direct backend write bypasses tool-level permissions
    cases["H12_direct_backend"] = run_risk_lab.run_direct_backend_case(risk_root / "fs_direct")
    return cases


class TestRiskLab:
    def test_h8_public_no_interrupt_and_executed(self, risk_cases):
        case = risk_cases["L1_public_allow"]
        assert case["interrupted"] is False
        assert case["disk_files"].get("public/note.md") == "PUBLIC"

    def test_h9_drafts_interrupts_then_approve_executes(self, risk_cases):
        case = risk_cases["L2_drafts_interrupt_approve"]
        assert case["interrupted"] is True
        assert not case["pre_resume_disk"]
        assert case["disk_files"].get("drafts/d.md") == "DRAFT"

    def test_h10_secrets_denied_no_interrupt(self, risk_cases):
        # NEGATIVE: a denied write must not execute and must not interrupt.
        case = risk_cases["L3_secrets_deny"]
        assert case["interrupted"] is False
        assert _denied(case) is True
        assert not case["disk_files"]

    def test_h10b_deny_before_interrupt_wins(self, risk_cases):
        # NEGATIVE: first match in declaration order wins (deny beats interrupt).
        case = risk_cases["L3b_deny_before_interrupt"]
        assert case["interrupted"] is False
        assert _denied(case) is True
        assert not case["disk_files"]

    def test_h11_edit_into_deny_still_denied(self, risk_cases):
        # NEGATIVE boundary: an edited call re-enters the tool and is still denied.
        case = risk_cases["H11_edit_redirect_into_deny"]
        assert case["interrupted"] is True
        assert _denied(case) is True
        assert not case["disk_files"]

    def test_h12_direct_backend_not_gated(self, risk_cases):
        case = risk_cases["H12_direct_backend"]
        assert case["write_result"]["error"] is None
        assert case["disk_files"].get("secrets/direct.md") == "DIRECT-BACKEND-WRITE"

    def test_risk_lab_all_checks_pass(self, risk_cases):
        # Mirror of run_risk_lab.main()'s own check computation.
        checks = {
            "H8_public_no_interrupt_and_executed": not risk_cases["L1_public_allow"]["interrupted"]
            and risk_cases["L1_public_allow"]["disk_files"].get("public/note.md") == "PUBLIC",
            "H9_drafts_interrupts_then_approve_executes": risk_cases["L2_drafts_interrupt_approve"]["interrupted"]
            and not risk_cases["L2_drafts_interrupt_approve"]["pre_resume_disk"]
            and risk_cases["L2_drafts_interrupt_approve"]["disk_files"].get("drafts/d.md") == "DRAFT",
            "H10_secrets_denied_no_interrupt": not risk_cases["L3_secrets_deny"]["interrupted"]
            and _denied(risk_cases["L3_secrets_deny"])
            and not risk_cases["L3_secrets_deny"]["disk_files"],
            "H10b_deny_before_interrupt_wins": not risk_cases["L3b_deny_before_interrupt"]["interrupted"]
            and _denied(risk_cases["L3b_deny_before_interrupt"])
            and not risk_cases["L3b_deny_before_interrupt"]["disk_files"],
            "H11_edit_into_deny_still_denied": risk_cases["H11_edit_redirect_into_deny"]["interrupted"]
            and _denied(risk_cases["H11_edit_redirect_into_deny"])
            and not risk_cases["H11_edit_redirect_into_deny"]["disk_files"],
            "H12_direct_backend_not_gated": risk_cases["H12_direct_backend"]["write_result"]["error"] is None
            and risk_cases["H12_direct_backend"]["disk_files"].get("secrets/direct.md") == "DIRECT-BACKEND-WRITE",
        }
        assert all(checks.values()), checks
