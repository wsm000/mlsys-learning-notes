"""Two separate processes: disk-backed write, then read without old messages."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
                  BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
                  BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from deepagents.backends import FilesystemBackend
from my_deepagent.demo_binding import build_agent

phase, run = sys.argv[1:]
root = Path('/home/simin/course09-quanty/persistent-reports') / run
root.mkdir(parents=True, exist_ok=True)
target = '/report.md'
disk = root / 'report.md'
if phase == 'write':
    question = ('Use analyze_price_volume for DEMO window=3, then write_file to /report.md. '
                'Include return, both mean volumes, volume ratio and synthetic-data limitation. '
                'Report the saved path. Do not use execute or other filesystem paths.')
elif phase == 'read':
    question = ('Use read_file to read /report.md and report its contents. '
                'Do not query price-volume data, write files or invent missing contents.')
else:
    raise SystemExit('phase must be write/read')
result = build_agent(backend=FilesystemBackend(root_dir=root, virtual_mode=True)).invoke(
    {'messages': [{'role': 'user', 'content': question}]})
messages = [m.model_dump(mode='json') for m in result['messages']]
calls = [c for m in messages for c in m.get('tool_calls', [])]
returns = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
expected = 'write_file' if phase == 'write' else 'read_file'
assert any(c['name'] == expected and returns.get(c['id'], {}).get('status') == 'success' for c in calls)
if phase == 'read':
    assert all(c['name'] == 'read_file' for c in calls), calls
assert disk.is_file()
raw = disk.read_bytes()
content = raw.decode('utf-8')
record = {'phase': phase, 'pid': os.getpid(), 'root_dir': str(root), 'virtual_path': target,
          'disk_path': str(disk), 'disk_sha256': hashlib.sha256(raw).hexdigest(),
          'input_messages_count': 1, 'input': question, 'messages': messages,
          'disk_content': content}
out = Path('/home/simin/course09-quanty/outputs') / f'persistence-{run}-{phase}.json'
out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
print('phase:', phase, 'pid:', os.getpid(), 'calls:', [(c['name'], c['args']) for c in calls], flush=True)
print('disk:', disk, 'sha256:', record['disk_sha256'], flush=True)
print('evidence:', out, flush=True)
