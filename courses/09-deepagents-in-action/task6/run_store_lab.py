"""Task6 E5 (Advanced): StoreBackend per-user isolation + cross-thread persistence.

Hypotheses (see experiment-plan.json):
  H7a: with ONE shared store and namespace=lambda rt: (rt.context.user_id, "files"),
       alice's /memory/AGENTS.md written in thread A1 is injected into the system
       prompt of alice's FRESH thread A2 (cross-thread persistence inside a user).
  H7b: bob, sharing the same store, does NOT see alice's file: his system prompt
       carries no preference and ls("/") is empty (namespace isolation between users).

The namespace comes from the runtime context (context_schema), which is how a
server would scope per-user storage. Evidence records the real system prompts.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_common import Recorder, ScriptedModel, env_record, now_iso, sha256_text, system_prompt_text, write_json

ALICE_PREF = "ALICE-PREF: 我喜欢中文回答，回答要简洁。"
BOB_PREF = "BOB-PREF: I prefer English answers."
MEMORY_PATH = "/memory/AGENTS.md"


@dataclass
class UserCtx:
    user_id: str


def build_agent(model, store):
    from deepagents import create_deep_agent
    from deepagents.backends import StoreBackend
    from langgraph.checkpoint.memory import MemorySaver

    backend = StoreBackend(namespace=lambda rt: (rt.context.user_id, "files"), store=store)
    return create_deep_agent(
        model=model,
        system_prompt="你是一个助理。",
        memory=[MEMORY_PATH],
        backend=backend,
        context_schema=UserCtx,
        checkpointer=MemorySaver(),
    )


def memory_answer_step(preference, yes_text, no_text):
    def step(messages):
        prompt = system_prompt_text(messages)
        if preference in prompt:
            return ("final", yes_text)
        return ("final", no_text)

    return step


def run_user_turn(store, user_id: str, thread_id: str, steps, tag: str, user_msg: str) -> dict:
    agent = build_agent(ScriptedModel(steps=steps, tag=tag), store)
    rec = Recorder()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": thread_id}, "callbacks": [rec]},
        context={"user_id": user_id},
    )
    prompt = system_prompt_text(rec.requests[0]["messages"]) if rec.requests else ""
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "final_message": result["messages"][-1].content,
        "system_prompt_has_own_preference": None,  # filled by caller
        "system_prompt_excerpt": prompt[:900],
        "requests": rec.requests,
        "events": rec.events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()

    # alice thread A1: write her preference into the shared store (her namespace)
    a1 = run_user_turn(
        store,
        "alice",
        "store-A1",
        steps=[
            ("write_file", {"file_path": MEMORY_PATH, "content": ALICE_PREF}),
            ("final", "alice 已写入偏好。"),
        ],
        tag="alice-writer",
        user_msg="记住：我喜欢中文回答，回答要简洁。",
    )

    # alice thread A2: FRESH thread, same user -> memory must arrive
    a2 = run_user_turn(
        store,
        "alice",
        "store-A2",
        steps=[memory_answer_step(ALICE_PREF, "alice 的记忆：中文回答。", "alice 没有记忆。")],
        tag="alice-reader",
        user_msg="我的偏好是什么？",
    )
    a2["system_prompt_has_own_preference"] = ALICE_PREF in a2["system_prompt_excerpt"] or ALICE_PREF in json.dumps(a2["requests"], ensure_ascii=False)

    # bob thread B1: same store, different user -> must NOT see alice's file
    b1 = run_user_turn(
        store,
        "bob",
        "store-B1",
        steps=[
            ("ls", {"path": "/"}),
            memory_answer_step(ALICE_PREF, "bob 看到了 alice 的偏好（不该发生）", "bob 没有看到 alice 的偏好。"),
        ],
        tag="bob-reader",
        user_msg="我的偏好是什么？",
    )
    b1_prompt = json.dumps(b1["requests"], ensure_ascii=False)
    b1["system_prompt_has_alice_preference"] = ALICE_PREF in b1_prompt
    ls_outputs = [
        e for e in b1["events"]
        if e["event"] == "tool_end" and isinstance(e.get("output"), dict) and e["output"].get("name") == "ls"
    ]
    b1["ls_saw_alice_file"] = "AGENTS.md" in json.dumps(ls_outputs, ensure_ascii=False)

    # bob writes his own preference in his namespace; alice must not see it either
    b2 = run_user_turn(
        store,
        "bob",
        "store-B2",
        steps=[
            ("write_file", {"file_path": MEMORY_PATH, "content": BOB_PREF}),
            ("final", "bob 已写入偏好。"),
        ],
        tag="bob-writer",
        user_msg="记住：I prefer English answers.",
    )
    a3 = run_user_turn(
        store,
        "alice",
        "store-A3",
        steps=[memory_answer_step(BOB_PREF, "alice 看到了 bob 的偏好（不该发生）", "alice 没有看到 bob 的偏好。")],
        tag="alice-reader-2",
        user_msg="有别人的偏好吗？",
    )
    a3_prompt = json.dumps(a3["requests"], ensure_ascii=False)
    a3["system_prompt_has_bob_preference"] = BOB_PREF in a3_prompt

    checks = {
        "H7a_alice_cross_thread_sees_own_memory": a2["system_prompt_has_own_preference"],
        "H7a_alice_answer_uses_memory": "中文" in a2["final_message"],
        "H7b_bob_cannot_see_alice_file": not b1["system_prompt_has_alice_preference"] and not b1["ls_saw_alice_file"],
        "H7b_alice_cannot_see_bob_file": not a3["system_prompt_has_bob_preference"],
    }

    summary = {
        "experiment": "task6-E5-store-isolation",
        "at": now_iso(),
        "env": env_record(),
        "alice_pref_sha256": sha256_text(ALICE_PREF),
        "bob_pref_sha256": sha256_text(BOB_PREF),
        "checks": checks,
        "all_passed": all(checks.values()),
        "threads": {"A1_write": a1, "A2_read": a2, "B1_read": b1, "B2_write": b2, "A3_read": a3},
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({"checks": checks, "all_passed": summary["all_passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
