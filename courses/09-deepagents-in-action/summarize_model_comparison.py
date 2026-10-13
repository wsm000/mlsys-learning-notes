import json
from pathlib import Path
import statistics

root = Path('evidence/model-comparison')
records = json.loads((root/'runs.json').read_text(encoding='utf-8'))
summary = []
for model in ('DeepSeek-V4-Flash', 'GLM-5.3-Flash'):
    rows = [r for r in records if r['model'] == model]
    completed = [r for r in rows if r['status'] == 'completed']
    summary.append({'model': model, 'attempts': len(rows), 'completed': len(completed),
                    'core_passes': sum(r['core_pass'] for r in rows),
                    'strict_integer_passes': sum(r.get('checks', {}).get('argument_integer_type', False) for r in rows),
                    'no_extra_tools_passes': sum(r.get('checks', {}).get('no_extra_tools', False) for r in rows),
                    'completed_median_seconds': statistics.median(r['duration_seconds'] for r in completed) if completed else None,
                    'errors': [{'repetition':r['repetition'],'type':r.get('error_type'),'http_status':r.get('http_status')} for r in rows if r['status']=='error']})
(root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
