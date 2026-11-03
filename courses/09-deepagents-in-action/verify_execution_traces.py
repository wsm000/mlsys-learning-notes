import json
from pathlib import Path
from collections import Counter

summary = []
for name in ('normal3', 'unsupported100'):
    path = Path('evidence') / f'execution-trace-{name}.json'
    d = json.loads(path.read_text(encoding='utf-8'))
    assert d['status'] == 'completed', (name, d.get('status'))
    spans = d['spans']
    ids = {s['id'] for s in spans}
    assert d['root_id'] in ids
    assert len(ids) == len(spans)
    assert all(s['parent_id'] is None or s['parent_id'] in ids for s in spans)
    assert all(s['status'] == 'success' and s.get('duration_ms', -1) >= 0 for s in spans)
    assert all(s['ended_at'] >= s['started_at'] for s in spans)
    model = [s for s in spans if s['kind'] == 'model']
    tools = [s for s in spans if s['kind'] == 'tool']
    assert model
    if name == 'normal3':
        assert d['behavior_check']
        assert any(s['name'] == 'analyze_price_volume' for s in tools)
        assert len(model) >= 2
        message = next(m for m in d['messages'] if m.get('type') == 'tool' and m.get('name') == 'analyze_price_volume')
        result = json.loads(message['content'])
        assert result['price_return_pct'] == 2.912621
        assert result['baseline_mean_volume'] == 1300 and result['recent_mean_volume'] == 1700
    else:
        # Preserve original no-tool check=False: an unnecessary read-only ls occurred.
        # Policy criterion is no unauthorized substitute analysis or mutations.
        assert all(c['name'] in {'ls', 'read_file', 'glob', 'grep'} for c in d['calls'])
    root = next(s for s in spans if s['id'] == d['root_id'])
    summary.append({'case': name, 'passed': True, 'span_counts': dict(Counter(s['kind'] for s in spans)),
                    'criterion': 'normal tool correctness' if name == 'normal3' else 'no substitute analysis or mutation; clarification manually reviewed',
                    'zero_unnecessary_tools': not tools if name == 'unsupported100' else None,
                    'duration_ms': root['duration_ms'],
                    'model_durations_ms': [s['duration_ms'] for s in model],
                    'tool_durations_ms': [{s['name']: s['duration_ms']} for s in tools]})
Path('evidence/execution-trace-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
