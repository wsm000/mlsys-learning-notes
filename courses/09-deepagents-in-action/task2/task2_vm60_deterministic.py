"""Program writes exact bytes; real Agent reads; external gates check files."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
os.environ.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
    BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
    BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from deepagents.backends import FilesystemBackend
from my_deepagent.demo_binding import build_agent

run = 'deterministic-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
root = Path('/home/simin/course09-quanty/task2_workspace') / run
out = Path('/home/simin/course09-quanty/outputs') / ('task2-' + run)
(root / 'reports').mkdir(parents=True, exist_ok=False)
out.mkdir(parents=True, exist_ok=False)
raw = 'Agent 任务已核验完成\n结果已保存至报告文件\n'.encode('utf-8')
report = '原始数据核验通过\n'.encode('utf-8') + raw
record = {'root_dir': str(root), 'writer': 'Python Path.write_bytes, not Agent', 'steps': []}

def check(name, expected):
    data = (root / 'reports' / name).read_bytes()
    if data != expected:
        raise ValueError('Byte mismatch: ' + name)
    return {'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def read(name, expected):
    path = '/reports/' + name
    step = {'operation': 'Agent read_file', 'path': path}
    record['steps'].append(step)
    result = build_agent(backend=FilesystemBackend(root_dir=root, virtual_mode=True)).invoke(
        {'messages': [{'role': 'user', 'content': f'只调用一次 read_file 读取 {path}，然后复述文件内容。不要调用其他工具，不得修改任何文件。'}]})
    messages = [m.model_dump(mode='json') for m in result['messages']]
    step['messages'] = messages
    calls = [c for m in messages for c in m.get('tool_calls', [])]
    replies = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
    assert len(calls) == 1 and calls[0]['name'] == 'read_file'
    c = calls[0]
    assert c['args']['file_path'] == path
    assert replies[c['id']].get('status') == 'success'
    for line in expected.decode().splitlines():
        assert line in str(replies[c['id']]['content'])
    step['verified'] = check(name, expected)
    step['raw_unchanged'] = check('raw.md', raw)
    if (root / 'reports/result.md').exists():
        step['report_unchanged'] = check('result.md', report)
    print('PASS Agent read: ' + path, flush=True)

try:
    (root / 'reports/raw.md').write_bytes(raw)
    record['steps'].append({'operation': 'program write raw', 'verified': check('raw.md', raw)})
    read('raw.md', raw)
    snapshot = (root / 'reports/raw.md').read_bytes()
    (root / 'reports/result.md').write_bytes('原始数据核验通过\n'.encode('utf-8') + snapshot)
    record['steps'].append({'operation': 'program assemble report from verified raw bytes',
        'verified': check('result.md', report), 'raw_unchanged': check('raw.md', snapshot)})
    read('result.md', report)
    read('raw.md', raw)
    record['passed'] = True
except Exception as exc:
    record.update(passed=False, error=type(exc).__name__ + ': ' + str(exc))
    raise
finally:
    (out / 'trace.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({'root_dir': str(root), 'evidence': str(out), 'passed': record.get('passed'), 'error': record.get('error')}, ensure_ascii=False), flush=True)
