"""Real-model file-tool experiment; run each phase in a separate process."""
import hashlib
import importlib.metadata
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
assert phase in ('write', 'edit', 'read')
assert run.isalnum() or all(c.isalnum() or c == '-' for c in run)
root = Path('/home/simin/course09-quanty/task2_workspace') / run
root.mkdir(parents=True, exist_ok=True)
disk = root / 'reports/result.md'
target = '/reports/result.md'
before = 'Agent 任务执行完成\n结果已保存至报告文件\n'
after = 'Agent 任务已核验完成\n结果已保存至报告文件\n'
output = Path('/home/simin/course09-quanty/outputs') / ('task2-' + run)
output.mkdir(parents=True, exist_ok=True)
previous = disk.read_text() if disk.exists() else None
if phase == 'write':
    if disk.exists():
        raise SystemExit('Refusing write phase: file already exists; inspect it first.')
    prompt = f'Use write_file to create {target} with exactly this content: {before!r}. Interpret the escaped newline characters as actual newlines, including the final newline. Then use read_file to verify both lines. Do not use execute or other tools to write. Report any discrepancy without further modification.'
elif phase == 'edit':
    prompt = f'First read_file {target}. If its content is not exactly {before!r} (escaped newlines mean real newlines), stop and report differences. Otherwise use edit_file to replace only Agent 任务执行完成 with Agent 任务已核验完成, leaving all other bytes unchanged. Then read_file again and verify the two lines. If the result differs, report it and do not modify again. Do not use execute.'
else:
    prompt = f'Only use read_file to read {target} and report its two lines. Do not write, edit, or execute anything.'
record = {'phase': phase, 'pid': os.getpid(), 'python': sys.version,
    'deepagents': importlib.metadata.version('deepagents'), 'root_dir': str(root),
    'virtual_path': target, 'disk_path': str(disk), 'input': prompt,
    'input_messages_count': 1, 'before': previous}
try:
    result = build_agent(backend=FilesystemBackend(root_dir=root, virtual_mode=True)).invoke(
        {'messages': [{'role': 'user', 'content': prompt}]})
    messages = [m.model_dump(mode='json') for m in result['messages']]
    record['messages'] = messages
    calls = [c for m in messages for c in m.get('tool_calls', [])]
    returns = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
    names = [c['name'] for c in calls]
    content = disk.read_text()
    record.update(calls=names, disk_content=content,
        sha256=hashlib.sha256(disk.read_bytes()).hexdigest())
    expected_names = {'write': ['write_file', 'read_file'],
        'edit': ['read_file', 'edit_file', 'read_file'], 'read': ['read_file']}[phase]
    file_names = [n for n in names if n in ('write_file', 'read_file', 'edit_file')]
    assert file_names == expected_names, names
    assert 'execute' not in names, names
    assert all(c['id'] in returns and returns[c['id']].get('status') == 'success' for c in calls)
    assert all(c['args']['file_path'] == target for c in calls if c['name'] in expected_names)
    assert content == (before if phase == 'write' else after), repr(content)
    if phase == 'edit':
        assert previous == before
        assert content == previous.replace('Agent 任务执行完成', 'Agent 任务已核验完成', 1)
    if phase == 'read':
        assert content == previous
    for c in calls:
        if c['name'] == 'read_file':
            returned = str(returns[c['id']]['content'])
            assert '结果已保存至报告文件' in returned
            expected_first = 'Agent 任务执行完成' if phase == 'write' or (phase == 'edit' and c is calls[0]) else 'Agent 任务已核验完成'
            assert expected_first in returned
    record['verified'] = True
except Exception as exc:
    record['verified'] = False
    record['error'] = type(exc).__name__ + ': ' + str(exc)
    raise
finally:
    (output / (phase + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({k: record.get(k) for k in ('phase', 'pid', 'calls', 'disk_path', 'disk_content', 'verified', 'error')}, ensure_ascii=False), flush=True)
