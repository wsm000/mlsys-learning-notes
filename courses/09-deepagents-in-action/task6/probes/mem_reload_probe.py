"""Probe: MemoryMiddleware loads once per thread (before_agent skips if
memory_contents already in state). Change the file on disk, re-invoke the SAME
thread: is the new content injected? And a FRESH thread: is it?
"""
import shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab_common import Recorder, ScriptedModel, system_prompt_text
from deepagents.backends import FilesystemBackend
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

import os, tempfile
root = Path(os.environ.get("TASK6_PROBE_ROOT", os.path.join(tempfile.gettempdir(), "task6probe", "memreload")))
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
backend = FilesystemBackend(root_dir=str(root))
cp = MemorySaver()

def agent(model):
    return create_deep_agent(model=model, system_prompt="助理", memory=["/memory/AGENTS.md"],
                             backend=backend, checkpointer=cp)

# thread T: first invoke loads empty memory
a = agent(ScriptedModel(steps=[("final", "ok")], tag="p1"))
rec = Recorder()
a.invoke({"messages": [{"role": "user", "content": "hi"}]},
         config={"configurable": {"thread_id": "T"}, "callbacks": [rec]})
print("invoke1 prompt has V1:", "V1" in system_prompt_text(rec.requests[0]["messages"]))

# change the file on disk
(root / "memory").mkdir(exist_ok=True)
(root / "memory" / "AGENTS.md").write_text("V1 content", encoding="utf-8")

# SAME thread, second invoke
a2 = agent(ScriptedModel(steps=[("final", "ok")], tag="p2"))
rec2 = Recorder()
a2.invoke({"messages": [{"role": "user", "content": "hi again"}]},
          config={"configurable": {"thread_id": "T"}, "callbacks": [rec2]})
print("same-thread invoke2 prompt has V1:", "V1" in system_prompt_text(rec2.requests[0]["messages"]))

# FRESH thread, third invoke
a3 = agent(ScriptedModel(steps=[("final", "ok")], tag="p3"))
rec3 = Recorder()
a3.invoke({"messages": [{"role": "user", "content": "hi fresh"}]},
          config={"configurable": {"thread_id": "T2"}, "callbacks": [rec3]})
print("fresh-thread invoke3 prompt has V1:", "V1" in system_prompt_text(rec3.requests[0]["messages"]))