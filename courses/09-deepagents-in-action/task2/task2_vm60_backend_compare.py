"""Independent-process Backend lifecycle comparison; scripted model, real tools."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0, '/home/simin/course09-quanty/course-source/notebooks/src')
from course_notebooks.testing import ScriptedChatModel
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.utils import file_data_to_string
from langchain_core.messages import AIMessage

TARGET = '/reports/raw.md'
CONTENT = 'Agent 任务已核验完成\n结果已保存至报告文件\n'

def child(phase, directory):
    out = Path(directory)
    root = out / 'disk'
    writing = phase.endswith('write')
    is_state = phase.startswith('state')
    tool = 'write_file' if writing else 'read_file'
    args = {'file_path': TARGET}
    if writing:
        args['content'] = CONTENT
    model = ScriptedChatModel(responses=[AIMessage(content='', tool_calls=[
        {'name': tool, 'args': args, 'id': phase, 'type': 'tool_call'}]), AIMessage(content='Finished')])
    backend = StateBackend() if is_state else FilesystemBackend(root_dir=root, virtual_mode=True)
    agent = create_deep_agent(model=model, backend=backend)
    inputs = {'messages': [{'role': 'user', 'content': f'{tool} {TARGET}'}]}
    if phase == 'state-restored-read':
        inputs['files'] = json.loads((out / 'state-files.json').read_text())
    result = agent.invoke(inputs)
    messages = [m.model_dump(mode='json') for m in result['messages']]
    reply = next(m for m in messages if m.get('tool_call_id') == phase)
    rec = {'phase': phase, 'pid': os.getpid(), 'backend': type(backend).__name__,
        'input_keys': list(inputs), 'old_chat_messages': False, 'target': TARGET,
        'messages': messages, 'python': sys.version, 'deepagents': importlib.metadata.version('deepagents')}
    if phase == 'state-write':
        files = result['files']
        content = file_data_to_string(files[TARGET])
        assert content == CONTENT, repr(content)
        # Explicit experiment export, not StateBackend automatic persistence.
        (out / 'state-files.json').write_text(json.dumps(files, ensure_ascii=False, indent=2))
        rec.update(content=content, sha256=hashlib.sha256(content.encode()).hexdigest(),
            export_writer='experiment program')
    if phase == 'filesystem-write':
        data = (root / TARGET.lstrip('/')).read_bytes()
        assert data == CONTENT.encode()
        rec.update(content=data.decode(), sha256=hashlib.sha256(data).hexdigest())
    if phase == 'state-empty-read':
        text = str(reply['content'])
        assert 'not found' in text.lower() or 'does not exist' in text.lower(), text
        assert CONTENT.splitlines()[0] not in text
        rec['expected_missing'] = True
    else:
        assert reply.get('status') == 'success', reply
        if not writing:
            for line in CONTENT.splitlines():
                assert line in str(reply['content'])
            if is_state:
                actual = file_data_to_string(result['files'][TARGET]).encode()
            else:
                actual = (root / TARGET.lstrip('/')).read_bytes()
            assert actual == CONTENT.encode()
            rec['sha256'] = hashlib.sha256(actual).hexdigest()
    rec['passed'] = True
    (out / (phase + '.json')).write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps({'phase': phase, 'pid': os.getpid(), 'passed': True,
        'tool_response': reply['content']}, ensure_ascii=False), flush=True)

if len(sys.argv) > 1:
    child(sys.argv[1], sys.argv[2])
else:
    name = 'task2-backend-compare-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path('/home/simin/course09-quanty/outputs') / name
    (out / 'disk').mkdir(parents=True, exist_ok=False)
    phases = ['state-write', 'filesystem-write', 'state-empty-read', 'state-restored-read', 'filesystem-read']
    for phase in phases:
        subprocess.run([sys.executable, __file__, phase, str(out)], check=True)
    records = [json.loads((out / (p + '.json')).read_text()) for p in phases]
    assert len({r['pid'] for r in records}) == len(records)
    hashes = [r['sha256'] for r in records if 'sha256' in r]
    assert len(set(hashes)) == 1
    summary = {'passed': True, 'mode': 'scripted model; real tools; fresh process per phase',
        'checkpointer': None, 'root_dir': str(out / 'disk'), 'results': [
            {'phase': r['phase'], 'pid': r['pid'], 'passed': r['passed']} for r in records],
        'matching_sha256': hashes[0]}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('EVIDENCE ' + str(out), flush=True)
