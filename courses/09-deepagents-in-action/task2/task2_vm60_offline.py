"""Scripted model, real Deep Agents tools. No model API or credentials."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, '/home/simin/course09-quanty/course-source/notebooks/src')
from course_notebooks.testing import ScriptedChatModel
from langchain_core.messages import AIMessage
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

phase = sys.argv[1]
root = Path('/home/simin/course09-quanty/task2_workspace/offline-20261015-01')
disk = root / 'reports/result.md'
out = Path('/home/simin/course09-quanty/outputs/task2-offline-20261015-01')
out.mkdir(parents=True, exist_ok=True)
root.mkdir(parents=True, exist_ok=True)
target = '/reports/result.md'
before = 'Agent 任务执行完成\n结果已保存至报告文件\n'
after = 'Agent 任务已核验完成\n结果已保存至报告文件\n'
previous = disk.read_text() if disk.exists() else None
if phase == 'write':
    assert previous is None, 'Existing file: refusing overwrite'
    steps = [('write_file', {'file_path': target, 'content': before}), ('read_file', {'file_path': target})]
elif phase == 'edit':
    assert previous == before
    steps = [('read_file', {'file_path': target}), ('edit_file', {'file_path': target, 'old_string': 'Agent 任务执行完成', 'new_string': 'Agent 任务已核验完成'}), ('read_file', {'file_path': target})]
elif phase == 'read':
    steps = [('read_file', {'file_path': target})]
else:
    raise ValueError(phase)
responses = [AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': f'{phase}-{i}', 'type': 'tool_call'}]) for i, (name, args) in enumerate(steps)]
responses.append(AIMessage(content='Scripted sequence finished; external assertions verify results.'))
agent = create_deep_agent(model=ScriptedChatModel(responses=responses),
    backend=FilesystemBackend(root_dir=root, virtual_mode=True))
result = agent.invoke({'messages': [{'role': 'user', 'content': f'Offline mechanism experiment: {phase}.'}]})
messages = [m.model_dump(mode='json') for m in result['messages']]
returns = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
content = disk.read_text()
record = {'mode': 'scripted-model-real-tools', 'phase': phase, 'pid': os.getpid(),
    'deepagents': importlib.metadata.version('deepagents'), 'python': sys.version,
    'root_dir': str(root), 'virtual_path': target, 'disk_path': str(disk),
    'before': previous, 'disk_content': content, 'messages': messages,
    'sha256': hashlib.sha256(disk.read_bytes()).hexdigest()}
(out / (phase + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2))
for i, (name, args) in enumerate(steps):
    reply = returns[f'{phase}-{i}']
    assert reply.get('status') == 'success', reply
    if name == 'read_file':
        expected = before if phase == 'write' or (phase == 'edit' and i == 0) else after
        for line in expected.splitlines():
            assert line in str(reply['content'])
assert content == (before if phase == 'write' else after)
if phase == 'edit':
    assert content == previous.replace('Agent 任务执行完成', 'Agent 任务已核验完成', 1)
if phase == 'read':
    assert content == previous
record['verified'] = True
(out / (phase + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps({'phase': phase, 'pid': os.getpid(), 'calls': [s[0] for s in steps], 'content': content, 'verified': True}, ensure_ascii=False))
