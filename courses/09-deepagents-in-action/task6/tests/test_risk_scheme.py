"""Pytest regression tests for the risk-tiered approval scheme (Task6 Advanced+).

Mirrors run_risk_scheme_lab.py: the unit tier table (classifier only, no agent)
plus the load-bearing end-to-end gates. Sandbox discipline as in test_labs.py:
everything runs under TEMP; the workspace is never written by subprocesses.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import pytest

TASK6_DIR = Path(__file__).resolve().parent.parent
if str(TASK6_DIR) not in sys.path:
    sys.path.insert(0, str(TASK6_DIR))

from lab_common import Recorder, ScriptedModel
from risk_policy import build_policy, classify_call
from run_risk_scheme_lab import decisions_of, disk_files, fresh_root, run_bulk_case, run_read_case, run_simple

TIER_TABLE = [
    ("ls", {"path": "/public"}, "L0"),
    ("read_file", {"file_path": "/public/a.md"}, "L0"),
    ("read_file", {"file_path": "/secrets/k.txt"}, "L3"),
    ("grep", {"path": None}, "L3"),
    ("glob", {"path": "/public", "pattern": "/secrets/**"}, "L3"),
    ("ls", {"path": "/"}, "L3"),
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


@pytest.mark.parametrize(("tool", "args", "want"), TIER_TABLE)
def test_classifier_tier(tool, args, want):
    assert classify_call(tool, args).tier == want


def test_classifier_decision_sets():
    assert classify_call("write_file", {"file_path": "/drafts/x", "content": "x"}).decisions == ["approve", "edit", "reject"]
    assert classify_call("delete", {"file_path": "/notes/x"}).decisions == ["approve", "reject"]
    assert classify_call("write_file", {"file_path": "/public/x", "content": "x"}).decisions == []


def test_build_policy_shape():
    policy = build_policy([])
    assert [r.mode for r in policy["permissions"]] == ["interrupt", "deny", "deny"]
    assert set(policy["interrupt_on"]) == {"write_file", "edit_file", "delete", "read_file"}
    assert policy["interrupt_on"]["delete"]["allowed_decisions"] == ["approve", "reject"]
    assert policy["interrupt_on"]["write_file"]["allowed_decisions"] == ["approve", "edit", "reject"]


@pytest.fixture(scope="module")
def sandbox_root():
    base = Path(tempfile.gettempdir()) / f"task6-scheme-tests-{__import__('os').getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_e2e_l1_public_write_auto_executes(sandbox_root):
    audit: list = []
    approvals: list = []
    case = run_simple("t_public", "/public/note.md", "PUBLIC", fresh_root(sandbox_root, "t_public"), audit, approvals)
    assert not case["interrupted"]
    assert case["disk_files"].get("public/note.md") == "PUBLIC"
    assert approvals == []


def test_e2e_l2_draft_interrupt_decision_set(sandbox_root):
    audit: list = []
    approvals: list = []
    case = run_simple("t_draft", "/drafts/d.md", "DRAFT", fresh_root(sandbox_root, "t_draft"), audit, approvals)
    assert case["interrupted"]
    assert decisions_of(case["interrupt_payload"], "write_file") == ["approve", "edit", "reject"]
    assert not case["disk_files"]  # gate is pre-execution
    assert approvals and approvals[0]["tier"] == "L2"


def test_e2e_l3_delete_interrupt_decision_set(sandbox_root):
    audit: list = []
    approvals: list = []
    case = run_read_case("t_secret_read", "/secrets/k.txt", fresh_root(sandbox_root, "t_secret", seed_secret=True), audit, approvals)
    assert case["interrupted"]
    assert decisions_of(case["interrupt_payload"], "read_file") == ["approve", "reject"]


def test_e2e_l4_secret_write_denied_no_interrupt(sandbox_root):
    audit: list = []
    approvals: list = []
    case = run_simple("t_secret_write", "/secrets/s.md", "S", fresh_root(sandbox_root, "t_secret_write"), audit, approvals)
    assert not case["interrupted"]
    assert not case["disk_files"]
    assert any("denied" in (m.get("content") or "").lower() for m in case["tool_messages"])


def test_e2e_l4_credential_content_denied_by_guard(sandbox_root):
    audit: list = []
    approvals: list = []
    case = run_simple(
        "t_cred", "/public/p.md", "api_key = sk-abcdef1234567890abcdef", fresh_root(sandbox_root, "t_cred"), audit, approvals
    )
    assert not case["interrupted"]
    assert not case["disk_files"]
    assert any(r.get("outcome") == "denied-by-content-guard" and r.get("tier") == "L4" for r in audit)


def test_e2e_unclassified_fails_closed(sandbox_root):
    audit: list = []
    approvals: list = []
    case = run_simple("t_unclassified", "/other/x.md", "X", fresh_root(sandbox_root, "t_unclassified"), audit, approvals)
    assert case["interrupted"]
    assert approvals and approvals[0]["tier"] == "L2"


def test_e2e_bulk_root_read_interrupts(sandbox_root):
    audit: list = []
    approvals: list = []
    case = run_bulk_case("t_bulk", "ls", {"path": "/"}, fresh_root(sandbox_root, "t_bulk", seed_secret=True), audit, approvals)
    assert case["interrupted"]
