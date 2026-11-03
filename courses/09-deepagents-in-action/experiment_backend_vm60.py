"""Run the course main application's agent and inspect its returned file state."""
import json
import os
from pathlib import Path
import uuid

os.environ.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
                  BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
                  BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from my_deepagent.demo_binding import build_agent

name = 'backend-' + uuid.uuid4().hex + '.md'
target = '/reports/' + name
disk_paths = [Path(target), Path.cwd() / 'reports' / name]
before = {str(p): p.exists() for p in disk_paths}
question = (f'Analyze DEMO with window=3 using analyze_price_volume. '
            f'Then use write_file to save a short Markdown report to {target}. '
            'Include return percentage, baseline/recent mean volumes, volume ratio and synthetic-data limitation. '
            f'After write_file succeeds, use read_file to read {target} and verify those values. '
            'Do not claim this is saved on the OS disk. Finish with the verification result.')
result = build_agent().invoke({'messages': [{'role': 'user', 'content': question}]})
messages = [m.model_dump(mode='json') for m in result['messages']]
requests = [c for m in messages for c in m.get('tool_calls', [])]
returns = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
files = result.get('files', {})
file_data = files.get(target)
raw = file_data.get('content', '') if isinstance(file_data, dict) else ''
content = raw if isinstance(raw, str) else '\n'.join(raw)
checks = {
    'query_write_read_success': all(any(c['name'] == name and returns.get(c['id'], {}).get('status') == 'success'
                                    for c in requests) for name in ('analyze_price_volume', 'write_file', 'read_file')),
    'file_in_returned_state': file_data is not None,
    'report_contains_expected_numbers': all(x in content for x in ('2.91', '1300', '1700', '1.31')),
    'disk_before': before,
    'disk_after': {str(p): p.exists() for p in disk_paths},
}
out = Path('/home/simin/course09-quanty/outputs/backend-experiment.json')
out.write_text(json.dumps({'target': target, 'checks': checks, 'files': files, 'messages': messages}, ensure_ascii=False, indent=2, default=str))
print('Requests:', json.dumps(requests, ensure_ascii=False))
print('Checks:', json.dumps(checks))
print('Report:', content)
print('Evidence:', out)
