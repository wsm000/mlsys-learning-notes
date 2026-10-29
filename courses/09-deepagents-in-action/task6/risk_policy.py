"""Risk-tiered approval scheme for Task6 (Advanced+): 按风险等级分层的审批方案.

One classifier is the single source of truth for tiering. The scheme compiles it
into three enforcement layers of the DeepAgents SDK:

  1. path rules        -> FilesystemPermission (allow / interrupt / deny)
  2. per-tool decision -> interrupt_on InterruptOnConfig(allowed_decisions=..., when=...)
                          (user-supplied entries win per tool name over the
                          permission-generated ones, so decision sets can be
                          narrowed per tier)
  3. content rules     -> RiskGateMiddleware.wrap_tool_call (denies credential-like
                          content; FilesystemPermission cannot see content)

Tiers:
  L0 read-only / reversible            auto-allow, no human
  L1 low-risk write (public area)      auto-allow, audited
  L2 medium-risk write (drafts, new)   human: approve / edit / reject
  L3 high-risk / hard to reverse       human: approve / reject   (no edit: editing a
                                       delete does not lower its risk class)
  L4 forbidden                          deny -- not approvable, no interrupt asked

Fail-closed defaults: an unclassified write path or an unknown tool tiers UP to
L2 (ask the human) instead of being allowed silently. Bulk reads (ls/glob/grep)
whose search subtree could overlap /secrets/** are tiered L3, mirroring the SDK
bulk predicate (pathless calls fire; glob's absolute pattern redirects the root).

Boundaries (verified in run_risk_scheme_lab.py, not claimed beyond):
  * permissions are tool-level; direct backend calls bypass everything (H12);
  * the content guard sees only write/edit content args; it is not a DLP scanner;
  * the audit trail is in-process; a production scheme must ship it to a durable sink.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from wcmatch import glob as wcglob

_WC_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR

# ---------------------------------------------------------------------------
# Tier table
# ---------------------------------------------------------------------------

TIERS: dict[str, dict[str, Any]] = {
    "L0": {"label": "只读/可逆", "human": False, "decisions": []},
    "L1": {"label": "低风险写入", "human": False, "decisions": []},
    "L2": {"label": "中风险写入", "human": True, "decisions": ["approve", "edit", "reject"]},
    "L3": {"label": "高风险/难撤销", "human": True, "decisions": ["approve", "reject"]},
    "L4": {"label": "禁止", "human": False, "decisions": [], "mode": "deny"},
}

READ_TOOLS = {"ls", "read_file", "glob", "grep"}
WRITE_TOOLS = {"write_file", "edit_file"}
BULK_TOOLS = {"ls", "glob", "grep"}

CREDENTIAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "openai-style key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws access key id"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|credential)s?\s*[:=]"), "credential assignment"),
]


@dataclass
class Decision:
    """Tier verdict for one tool call."""

    tier: str
    mode: str  # "allow" | "interrupt" | "deny"
    decisions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"tier": self.tier, "mode": self.mode, "decisions": self.decisions, "reasons": self.reasons}


def _globmatch(path: str, pattern: str) -> bool:
    return bool(wcglob.globmatch(path, pattern, flags=_WC_FLAGS))


def _bulk_overlaps_secrets(tool_name: str, args: dict) -> bool:
    """Bulk read whose search subtree could overlap /secrets/** (SDK-consistent).

    Mirrors deepagents.middleware._fs_interrupt: a pathless bulk call fires
    unconditionally; "." / "" / "./" collapse to the whole tree; and glob's
    absolute pattern redirects the search root independently of path.
    """
    from deepagents.backends.utils import _glob_anchor, _paths_overlap

    args = args or {}
    raw_path = args.get("path")
    if raw_path is None:
        return True  # pathless bulk call can touch anything
    if not isinstance(raw_path, str):
        return False
    if raw_path in (".", "", "./"):
        return True
    if _paths_overlap(_glob_anchor(raw_path), "/secrets"):
        return True
    if tool_name == "glob":
        pattern = args.get("pattern")
        if isinstance(pattern, str):
            posix = pattern.replace("\\", "/")
            if posix.startswith("/"):
                return _paths_overlap(_glob_anchor(pattern), "/secrets")
            if ".." in PurePosixPath(posix).parts:
                return True  # relative pattern with .. can climb out of path
    return False


def classify_call(tool_name: str, args: dict) -> Decision:
    """Tier one tool call. Content rules outrank path rules; deny outranks interrupt."""
    args = args or {}
    path = args.get("file_path") or args.get("path") or ""
    content = args.get("content") or args.get("new_string") or ""

    # --- L4 content gate (highest precedence; permissions cannot see content) ---
    for pattern, label in CREDENTIAL_PATTERNS:
        if isinstance(content, str) and pattern.search(content):
            return Decision("L4", "deny", [], [f"content matches {label}"])

    # --- L4 path rules ---
    if tool_name in WRITE_TOOLS and _globmatch(path, "/secrets/**"):
        return Decision("L4", "deny", [], ["write/edit under /secrets/**"])
    if tool_name == "delete" and _globmatch(path, "/secrets/**"):
        return Decision("L4", "deny", [], ["delete under /secrets/**"])

    # --- reads ---
    if tool_name in READ_TOOLS:
        if tool_name in BULK_TOOLS:
            if _bulk_overlaps_secrets(tool_name, args):
                return Decision("L3", "interrupt", ["approve", "reject"], ["bulk read subtree may overlap /secrets/**"])
            return Decision("L0", "allow", [], ["bulk read outside sensitive area"])
        if _globmatch(path, "/secrets/**"):
            return Decision("L3", "interrupt", ["approve", "reject"], ["read under /secrets/**"])
        return Decision("L0", "allow", [], ["read-only"])

    # --- delete ---
    if tool_name == "delete":
        return Decision("L3", "interrupt", ["approve", "reject"], ["delete is hard to reverse"])

    # --- writes ---
    if tool_name in WRITE_TOOLS:
        if _globmatch(path, "/public/**"):
            return Decision("L1", "allow", [], ["public-area write"])
        if _globmatch(path, "/drafts/**"):
            return Decision("L2", "interrupt", ["approve", "edit", "reject"], ["draft-area write"])
        return Decision("L2", "interrupt", ["approve", "edit", "reject"], ["unclassified path fails closed to approval"])

    # --- unknown tool: fail closed ---
    return Decision("L2", "interrupt", ["approve", "reject"], ["unknown tool fails closed to approval"])


# ---------------------------------------------------------------------------
# Enforcement layers
# ---------------------------------------------------------------------------


def _when(tool_name: str):
    """Build the HITL when-predicate: interrupt iff the classifier says L2/L3."""

    def predicate(request) -> bool:
        args = request.tool_call.get("args", {}) or {}
        return classify_call(tool_name, args).mode == "interrupt"

    return predicate


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def make_gate(audit: list):
    """Content-based L4 gate + audit trail, as a langchain AgentMiddleware."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    class _RiskGate(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            call = request.tool_call
            verdict = classify_call(call.get("name"), call.get("args", {}) or {})
            record = {
                "at": time.time(),
                "tool": call.get("name"),
                "args_digest": _digest(call.get("args", {})),
                **verdict.as_dict(),
                "executed": False,
            }
            if verdict.tier == "L4" and any("content" in reason for reason in verdict.reasons):
                record["outcome"] = "denied-by-content-guard"
                audit.append(record)
                return ToolMessage(
                    content=f"Blocked by risk policy ({verdict.tier}): {'; '.join(verdict.reasons)}",
                    name=call.get("name"),
                    tool_call_id=call.get("id"),
                    status="error",
                )
            result = handler(request)
            record["executed"] = True
            record["outcome"] = "executed"
            status = getattr(result, "status", None)
            if status is not None:
                record["result_status"] = status
            audit.append(record)
            return result

    return _RiskGate()


def build_policy(audit: list | None = None) -> dict:
    """Compile the scheme into create_deep_agent arguments.

    Returns {"permissions", "interrupt_on", "middleware", "audit"}; pass the
    returned pieces straight into create_deep_agent(...).
    """
    from deepagents.middleware.filesystem import FilesystemPermission
    from langchain.agents.middleware import InterruptOnConfig

    audit = audit if audit is not None else []
    permissions = [
        # L3 reads on sensitive paths; the SDK auto-generates bulk-aware
        # predicates for ls/glob/grep from this rule (pathless bulk fires).
        FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="interrupt"),
        # L4 path denies (tool-level, pre-execution).
        FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
        FilesystemPermission(operations=["delete"], paths=["/secrets/**"], mode="deny"),
    ]
    interrupt_on = {
        "write_file": InterruptOnConfig(allowed_decisions=["approve", "edit", "reject"], when=_when("write_file")),
        "edit_file": InterruptOnConfig(allowed_decisions=["approve", "edit", "reject"], when=_when("edit_file")),
        "delete": InterruptOnConfig(allowed_decisions=["approve", "reject"], when=_when("delete")),
        "read_file": InterruptOnConfig(allowed_decisions=["approve", "reject"], when=_when("read_file")),
    }
    return {
        "permissions": permissions,
        "interrupt_on": interrupt_on,
        "middleware": [make_gate(audit)],
        "audit": audit,
    }
