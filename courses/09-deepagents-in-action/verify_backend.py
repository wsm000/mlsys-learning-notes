"""Verify captured evidence without another model call."""
import json
from pathlib import Path

path = Path('evidence/backend-experiment.json')
d = json.loads(path.read_text(encoding='utf-8'))
raw = d['files'][d['target']]['content']
content = raw if isinstance(raw, str) else '\n'.join(raw)
calls = [c for m in d['messages'] for c in m.get('tool_calls', [])]
returns = {m['tool_call_id']: m for m in d['messages'] if m.get('type') == 'tool'}
names = [c['name'] for c in calls]
assert names == ['analyze_price_volume', 'write_file', 'read_file'], names
assert all(returns[c['id']]['status'] == 'success' for c in calls)
write, read = calls[1:]
assert write['args']['file_path'] == read['args']['file_path'] == d['target']
assert write['args']['content'] == content
read_content = returns[read['id']]['content']
for value in ('2.91', '1300', '1700', '1.31', 'synthetic'):
    assert value in content and value in read_content
assert not any(d['checks']['disk_before'].values())
assert not any(d['checks']['disk_after'].values())
verification = {'query_write_read_id_matched': True, 'written_content_matches_state': True,
                'readback_contains_expected_numbers': True, 'checked_disk_paths_absent': True,
                'note': 'Original report check falsely failed due to string/list handling; corrected offline.'}
Path('evidence/backend-verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
Path('evidence/backend-report.md').write_text(content, encoding='utf-8')
print(json.dumps(verification))
