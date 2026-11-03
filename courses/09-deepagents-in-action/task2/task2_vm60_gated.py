"""Real-model operations with program-controlled byte verification gates."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
    BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
    BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from deepagents.backends import FilesystemBackend
from my_deepagent.demo_binding import build_agent

run = 'gated-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
root = Path('/home/simin/course09-quanty/task2_workspace') / run
root.mkdir(parents=True, exist_ok=False)
out = Path('/home/simin/course09-quanty/outputs') / ('task2-' + run)
out.mkdir(parents=True, exist_ok=False)
raw = 'Agent 任务已核验完成\n结果已保存至报告文件\n'.encode('utf-8')
report = '原始数据核验通过\n'.encode('utf-8') + raw
record = {'run': run, 'root_dir': str(root), 'output_dir': str(out),
    'python': sys.version, 'deepagents': importlib.metadata.version('deepagents'),
    'criteria': 'UTF-8; LF after each line including final line; exact bytes', 'steps': []}

def save():
    (out / 'trace.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))

def verify(path, expected):
    actual = path.read_bytes() if path.exists() else None
    if actual != expected:
        index = None if actual is None else next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
        raise ValueError(json.dumps({'path': str(path), 'expected': expected.decode(),
            'actual': None if actual is None else actual.decode('utf-8', errors='replace'),
            'first_differing_byte_zero_based': index}, ensure_ascii=False))
    return {'path': str(path), 'byte_count': len(actual), 'sha256': hashlib.sha256(actual).hexdigest(), 'exact_bytes': True}

def operation(number, name, path, content=None):
    step = {'number': number, 'tool': name, 'path': path}
    record['steps'].append(step)
    prompt = f'本次只允许调用一次 {name}，路径为 {path}。不要调用其他工具，不要规划。'
    if content is not None:
        prompt += '写入内容由以下 JSON 字符串精确定义，请解析转义后写入，UTF-8，每个 \\n 表示一个真实 LF，包括最后的 LF：' + json.dumps(content.decode(), ensure_ascii=False)
    else:
        prompt += '读回并报告文件内容，不得修改。'
    step['input'] = prompt
    save()
    # A fresh invocation prevents the model from executing future stages early.
    result = build_agent(backend=FilesystemBackend(root_dir=root, virtual_mode=True)).invoke(
        {'messages': [{'role': 'user', 'content': prompt}]})
    messages = [m.model_dump(mode='json') for m in result['messages']]
    step['messages'] = messages
    save()
    calls = [c for m in messages for c in m.get('tool_calls', [])]
    replies = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
    assert len(calls) == 1 and calls[0]['name'] == name, 'Unexpected tool calls'
    c = calls[0]
    assert c['args']['file_path'] == path
    assert replies[c['id']].get('status') == 'success'
    if content is not None:
        assert c['args']['content'].encode('utf-8') == content, 'Write arguments differ'
    else:
        expected = raw if path.endswith('raw.md') else report
        for line in expected.decode().splitlines():
            assert line in str(replies[c['id']]['content']), 'Readback missing line'
    step['checks'] = [verify(root / 'reports/raw.md', raw)]
    if (root / 'reports/result.md').exists():
        step['checks'].append(verify(root / 'reports/result.md', report))
    elif number >= 3:
        raise ValueError('Report missing')
    step['passed'] = True
    save()
    print(f'Step {number} passed: {name} {path}; exact bytes verified', flush=True)

try:
    operation(1, 'write_file', '/reports/raw.md', raw)
    operation(2, 'read_file', '/reports/raw.md')
    snapshot = (root / 'reports/raw.md').read_bytes()
    operation(3, 'write_file', '/reports/result.md', report)
    assert (root / 'reports/raw.md').read_bytes() == snapshot
    operation(4, 'read_file', '/reports/result.md')
    operation(5, 'read_file', '/reports/raw.md')
    record['passed'] = True
except Exception as exc:
    record.update(passed=False, error=type(exc).__name__ + ': ' + str(exc))
    print('STOP: ' + record['error'], flush=True)
    raise
finally:
    save()
    print(json.dumps({'root_dir': str(root), 'evidence': str(out), 'passed': record.get('passed')}, ensure_ascii=False), flush=True)
