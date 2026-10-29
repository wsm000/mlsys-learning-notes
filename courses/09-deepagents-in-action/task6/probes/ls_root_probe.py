"""Probe: does FilesystemBackend create root_dir? (run on vm-60, deepagents 0.7.8)

Finding: it does NOT. ls on a non-existent root (or any not-yet-written path)
returns path_not_found, while write_file creates parent dirs on demand. This
trapped the first live L1 attempt: the real model explored with ls first, every
ls failed, and it never reached write_file.
"""
import shutil
from pathlib import Path
from deepagents.backends import FilesystemBackend

root = Path("/tmp/task6probe/fs2")
if root.exists():
    shutil.rmtree(root)
backend = FilesystemBackend(root_dir=str(root))  # root NOT created
print("root exists before:", root.exists())
for p in ("/", "/memory"):
    res = backend.ls(p)
    print("ls", repr(p), "->", getattr(res, "error", None))
print("root exists after ls:", root.exists())
w = backend.write("/memory/AGENTS.md", "hello")
print("write ->", getattr(w, "error", None), getattr(w, "path", None))
res = backend.ls("/")
print("after write ls('/') ->", getattr(res, "error", None))
