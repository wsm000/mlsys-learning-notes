"""Run the user's four-step task with the main-template real model."""
import hashlib
import json
import os
from pathlib import Path
import sys
import importlib.metadata

os.environ.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
    BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
    BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from deepagents.backends import FilesystemBackend
from my_deepagent.demo_binding import build_agent

root = Path('/home/simin/course09-quanty/task2_workspace/real-four-steps-20261015-01')
root.mkdir(parents=True, exist_ok=True)
assert not any(root.iterdir()), 'Refusing to overwrite an existing experiment'
out = Path('/home/simin/course09-quanty/outputs/task2-real-four-steps-20261015-01')
out.mkdir(parents=True, exist_ok=True)
raw = 'Agent 任务已核验完成\n结果已保存至报告文件\n'
report = '原始数据核验通过\n' + raw
prompt = '''请严格按顺序执行以下任务，不要跳过或合并步骤；任何一步发现不一致，立即停止，并报告“步骤编号、期望内容、实际内容、差异位置”，不要自动修正。
文件操作仅使用内置文件工具，不要使用 execute。所有文件均为 UTF-8，每行末尾使用一个 LF 换行，包括文件最后一行。FilesystemBackend 会在写入时创建父目录。
1. 保存原始数据：确保 /reports 目录存在；将以下两行原样写入 /reports/raw.md：
Agent 任务已核验完成
结果已保存至报告文件
2. 读回原始数据：读取 /reports/raw.md，逐行核对是否与第 1 步完全一致，包括顺序、字符、空格、标点和换行。若不一致，停止并报告差异。
3. 生成报告：若第 2 步通过，将以下三行原样写入 /reports/result.md：
原始数据核验通过
Agent 任务已核验完成
结果已保存至报告文件
4. 核验：读回 /reports/result.md，确认其内容恰好为第 3 步规定的三行；再次读取 /reports/raw.md，确认仍与第 1 步的两行完全一致。若任一不一致，停止并报告差异。
全部一致后，仅回复：任务完成：原始数据已保存并读回，报告已生成并核验通过。'''
record = {'input': prompt, 'root_dir': str(root), 'pid': os.getpid(),
    'python': sys.version, 'deepagents': importlib.metadata.version('deepagents')}
try:
    result = build_agent(backend=FilesystemBackend(root_dir=root, virtual_mode=True)).invoke(
        {'messages': [{'role': 'user', 'content': prompt}]})
    messages = [m.model_dump(mode='json') for m in result['messages']]
    record['messages'] = messages
    calls = [c for m in messages for c in m.get('tool_calls', [])]
    replies = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
    file_calls = [c for c in calls if c['name'] in ('write_file', 'read_file', 'edit_file')]
    expected = [('write_file', '/reports/raw.md'), ('read_file', '/reports/raw.md'),
        ('write_file', '/reports/result.md'), ('read_file', '/reports/result.md'), ('read_file', '/reports/raw.md')]
    assert [(c['name'], c['args']['file_path']) for c in file_calls] == expected
    assert all(c['name'] in ('ls', 'write_file', 'read_file') for c in calls)
    assert all(replies[c['id']].get('status') == 'success' for c in calls)
    # Require each preceding file tool to finish before the next is requested.
    for prev, following in zip(file_calls, file_calls[1:]):
        return_index = next(i for i, m in enumerate(messages) if m.get('tool_call_id') == prev['id'])
        request_index = next(i for i, m in enumerate(messages) if any(c['id'] == following['id'] for c in m.get('tool_calls', [])))
        assert return_index < request_index, 'Steps were merged or reordered'
    record['files'] = {}
    for name, content in [('raw.md', raw), ('result.md', report)]:
        data = (root / 'reports' / name).read_bytes()
        record['files'][name] = {'content': data.decode('utf-8'), 'sha256': hashlib.sha256(data).hexdigest()}
        assert data == content.encode('utf-8'), f'Unexpected bytes: {name}'
    assert file_calls[0]['args']['content'] == raw
    assert file_calls[2]['args']['content'] == report
    for c in file_calls:
        if c['name'] == 'read_file':
            for line in (report if c['args']['file_path'].endswith('result.md') else raw).splitlines():
                assert line in str(replies[c['id']]['content'])
    assert messages[-1]['content'].strip() == '任务完成：原始数据已保存并读回，报告已生成并核验通过。'
    record['verified'] = True
except Exception as exc:
    record.update(verified=False, error=type(exc).__name__ + ': ' + str(exc))
    raise
finally:
    (out / 'trace.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({k: record.get(k) for k in ('root_dir', 'verified', 'error', 'files')}, ensure_ascii=False), flush=True)
