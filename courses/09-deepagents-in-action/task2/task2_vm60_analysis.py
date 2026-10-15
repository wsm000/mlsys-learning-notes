"""Real Agent reads source and writes an analysis; external program verifies."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
os.environ.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
    BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
    BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from deepagents.backends import FilesystemBackend
from my_deepagent.demo_binding import build_agent

run = 'analysis-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
root = Path('/home/simin/course09-quanty/task2_workspace') / run
out = Path('/home/simin/course09-quanty/outputs') / ('task2-' + run)
root.mkdir(parents=True, exist_ok=False)
out.mkdir(parents=True, exist_ok=False)
source = '第1天：100\n第2天：110\n第3天：105\n'.encode('utf-8')
(root / 'source.md').write_bytes(source)
record = {'root_dir': str(root), 'source_writer': 'experiment program', 'source': source.decode()}
agent = build_agent(backend=FilesystemBackend(root_dir=root, virtual_mode=True))
prompt = '''先仅调用 read_file 读取 /source.md；取得工具返回后，按文件实际数据分析第3天相较第1天的整体涨幅。随后用 write_file 保存 /report.json，内容为一个合法 JSON 对象，字段必须为 source_path、start_day、end_day、start_value、end_value、return_pct、formula、explanation。day 和 value 以及 return_pct 均用数值，return_pct 使用百分数数值（如5表示5%）。explanation 用中文说明整体区间与最后一天相较前一天的变化不矛盾。不要修改 source.md，不要使用 execute，不要读写其他路径。写入后 read_file 读回 /report.json 再回复。'''
try:
    result = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
    messages = [m.model_dump(mode='json') for m in result['messages']]
    record['messages'] = messages
    calls = [c for m in messages for c in m.get('tool_calls', [])]
    replies = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
    assert [(c['name'], c['args'].get('file_path')) for c in calls] == [
        ('read_file', '/source.md'), ('write_file', '/report.json'), ('read_file', '/report.json')]
    assert all(replies[c['id']].get('status') == 'success' for c in calls)
    for previous, following in zip(calls, calls[1:]):
        completed = next(i for i, m in enumerate(messages) if m.get('tool_call_id') == previous['id'])
        requested = next(i for i, m in enumerate(messages) if any(c['id'] == following['id'] for c in m.get('tool_calls', [])))
        assert completed < requested, 'Dependency order violated'
    assert (root / 'source.md').read_bytes() == source, 'Source changed'
    report = json.loads((root / 'report.json').read_text())
    record['report'] = report
    for key, expected in {'source_path':'/source.md', 'start_day':1, 'end_day':3,
        'start_value':100, 'end_value':105, 'return_pct':5}.items():
        assert report[key] == expected, (key, report[key], expected)
    assert isinstance(report['formula'], str) and report['formula']
    assert isinstance(report['explanation'], str) and report['explanation']
    record.update(passed=True, checks={'source_unchanged':True, 'tool_order':True,
        'source_values_and_interval':True, 'return_pct':True},
        limitation='Formula and explanation require human review; nonempty strings alone do not establish correctness.')
except Exception as exc:
    record.update(passed=False, error=type(exc).__name__ + ': ' + str(exc))
    raise
finally:
    (out / 'trace.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({k: record.get(k) for k in ('root_dir', 'passed', 'error', 'report')}, ensure_ascii=False), flush=True)
    print('EVIDENCE ' + str(out), flush=True)
