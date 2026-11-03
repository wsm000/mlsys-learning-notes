"""Record installed source locations and stack assembly, without credentials."""
import inspect
import json
from pathlib import Path
import importlib.metadata
from deepagents import graph
from deepagents.middleware.filesystem import FilesystemMiddleware

records = {}
for name, obj in (("create_deep_agent", graph.create_deep_agent), ("FilesystemMiddleware.__init__", FilesystemMiddleware.__init__)):
    lines, start = inspect.getsourcelines(obj)
    selected = [{"line": start+i, "source": line.rstrip()} for i,line in enumerate(lines)
                if any(s in line for s in ('StateBackend', 'backend =', 'Middleware(', 'middleware.append', 'middleware.extend'))]
    records[name] = {"file": inspect.getsourcefile(obj), "start_line": start, "selected_lines": selected}
records['version'] = importlib.metadata.version('deepagents')
Path('outputs').mkdir(exist_ok=True)
Path('outputs/components.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
