import inspect
from deepagents import create_deep_agent
sig = inspect.signature(create_deep_agent)
for name, p in sig.parameters.items():
    print(name, ":", (p.annotation if p.annotation is not inspect._empty else "?"))
