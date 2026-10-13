import json
from pathlib import Path

root = Path('evidence')
write = json.loads((root / 'persistence-write.json').read_text(encoding='utf-8'))
read = json.loads((root / 'persistence-read.json').read_text(encoding='utf-8'))
assert write['pid'] != read['pid']
assert write['disk_sha256'] == read['disk_sha256']
assert write['disk_content'] == read['disk_content']
assert read['input_messages_count'] == 1
calls = [c for m in read['messages'] for c in m.get('tool_calls', [])]
assert calls and all(c['name'] == 'read_file' for c in calls)
returns = {m['tool_call_id']: m for m in read['messages'] if m.get('type') == 'tool'}
for c in calls:
    assert c['args']['file_path'].lstrip('/') == 'report.md'
    message = returns[c['id']]
    assert message['status'] == 'success'
    for line in read['disk_content'].splitlines():
        if line.strip():
            assert line in message['content']
result = {'different_processes': True, 'disk_hash_unchanged': True,
          'read_started_without_old_messages': True, 'only_read_tool_called': True,
          'readback_matches_disk_lines': True,
          'scope': 'Two Python processes on the same vm; not an OS reboot test.'}
(root / 'persistence-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
