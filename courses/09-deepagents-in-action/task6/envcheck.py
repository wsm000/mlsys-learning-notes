import sys
try:
    import deepagents, langchain, langgraph
    print("python", sys.version.split()[0])
    import importlib.metadata as md
    for pkg in ("deepagents", "langchain", "langchain-core", "langgraph", "langgraph-sdk", "openai"):
        try:
            print(pkg, md.version(pkg))
        except Exception as e:
            print(pkg, "MISSING", e)
    from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
    print("InterruptOnConfig fields:", list(InterruptOnConfig.__annotations__))
    from deepagents import MemoryMiddleware, create_deep_agent
    from deepagents.backends import StoreBackend, StateBackend, FilesystemBackend, CompositeBackend
    from deepagents.middleware.filesystem import FilesystemPermission
    from deepagents.graph import _build_interrupt_on_from_permissions
    print("ALL_IMPORTS_OK")
except Exception as e:
    print("FAIL:", type(e).__name__, e)
