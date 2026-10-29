"""Shared harness for Task6 offline labs: deterministic scripted model + recorder.

Discipline (inherited from Task5): offline scripted runs prove the mechanical
chain only. The scripted model is NOT evidence of semantic model behavior; live
model runs are a separate denominator. All evidence is written under a
caller-supplied output directory (TEMP on this host; transferred verbatim).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


def msg_record(message) -> dict:
    if isinstance(message, dict):
        return message
    record = {"type": getattr(message, "type", type(message).__name__), "content": text_of(message.content)}
    for key in ("id", "name", "tool_calls", "tool_call_id", "status"):
        value = getattr(message, key, None)
        if value is not None:
            record[key] = value
    return record


class Recorder(BaseCallbackHandler):
    """Record real model requests/responses and tool events for evidence."""

    raise_error = True

    def __init__(self, budget: int = 12):
        self.budget = budget
        self.requests = []
        self.responses = []
        self.events = []

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        if len(self.requests) >= self.budget:
            raise RuntimeError(f"predeclared {self.budget}-model-call budget exhausted")
        tools = kwargs.get("invocation_params", {}).get("tools") or kwargs.get("tools", [])
        self.requests.append(
            {
                "at": now_iso(),
                "run_id": str(run_id),
                "messages": [msg_record(m) for m in messages[0]],
                "tools_schema_sha256": sha256_text(json.dumps(tools, ensure_ascii=False, sort_keys=True)) if tools else None,
            }
        )

    def on_llm_end(self, response, *, run_id, **kwargs):
        for group in response.generations:
            for generation in group:
                self.responses.append({"at": now_iso(), "run_id": str(run_id), "message": msg_record(generation.message)})

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.events.append(
            {"event": "tool_start", "at": now_iso(), "run_id": str(run_id), "name": (serialized or {}).get("name"), "input": input_str}
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
        update = getattr(output, "update", None)
        if hasattr(output, "type"):
            value = msg_record(output)
        elif isinstance(update, dict):
            value = {"kind": "Command", "messages": [msg_record(m) for m in update.get("messages", [])]}
        else:
            value = text_of(output)
        self.events.append({"event": "tool_end", "at": now_iso(), "run_id": str(run_id), "output": value})


class ScriptedModel(BaseChatModel):
    """Deterministic driver. Steps are (tool_name, args) or ("final", text).

    A step may also be a callable(messages) -> action so a scripted answer can
    depend on what the model ACTUALLY receives (e.g. memory injected into the
    system prompt). This proves injection mechanically; it is not a claim about
    real-model semantics.
    """

    steps: list = Field(default_factory=list)
    cursor: int = 0
    bound_names: list = Field(default_factory=list)
    tag: str = "scripted"

    @property
    def _llm_type(self) -> str:
        return f"task6-scripted-{self.tag}"

    def bind_tools(self, tools, **kwargs):
        self.bound_names = [t.name if hasattr(t, "name") else t.get("function", t).get("name") for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.cursor >= len(self.steps):
            action = ("final", "[script exhausted] no further scripted step")
        else:
            action = self.steps[self.cursor]
        self.cursor += 1
        if callable(action):
            action = action(messages)
        if action[0] == "final":
            message = AIMessage(content=action[1])
        else:
            name, args = action
            if name not in self.bound_names:
                raise RuntimeError(f"scripted tool {name!r} not bound; bound={self.bound_names}")
            message = AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call-{self.cursor:04d}", "type": "tool_call"}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def system_prompt_text(messages) -> str:
    for m in messages:
        mtype = m.get("type") if isinstance(m, dict) else getattr(m, "type", None)
        if mtype == "system":
            content = m.get("content") if isinstance(m, dict) else m.content
            return text_of(content)
    return ""


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def env_record() -> dict:
    import platform
    import sys

    record = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import importlib.metadata as md

        for pkg in ("deepagents", "langchain", "langchain-core", "langgraph"):
            try:
                record[pkg] = md.version(pkg)
            except Exception:
                record[pkg] = "missing"
    except Exception as exc:
        record["packages_error"] = str(exc)
    return record