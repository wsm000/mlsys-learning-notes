"""Summarize an existing paired trace without making model calls."""
import json,sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]);summary={}
for arm in ('off','on'):
    data=json.loads((root/arm/'result.json').read_text(encoding='utf-8'))
    msgs=data['messages'];calls=[c for m in msgs for c in m.get('tool_calls',[])]
    counts=Counter((c['name'],json.dumps(c['args'],sort_keys=True,ensure_ascii=False)) for c in calls if c['name']!='write_todos')
    usage=Counter()
    for m in msgs:
        for k in ('input_tokens','output_tokens','total_tokens'):
            usage[k]+=(m.get('usage_metadata') or {}).get(k,0) or 0
    entries=[]
    for i,m in enumerate(msgs):
        for c in m.get('tool_calls',[]):entries.append({'message_index':i,'name':c['name'],'args':c['args'],'id':c['id']})
    writes=[e for e in entries if e['name']=='write_file' and e['args'].get('file_path')=='/reports/comparison.json']
    readbacks=[e for e in entries if e['name']=='read_file' and e['args'].get('file_path')=='/reports/comparison.json']
    read_after_write=bool(writes and readbacks) and any(m.get('tool_call_id')==writes[-1]['id'] for m in msgs[:readbacks[-1]['message_index']])
    summary[arm]={k:data.get(k) for k in ['mechanical_pass','mechanical_checks','call_counts','tool_call_count','model_call_count','elapsed_seconds','error']}
    summary[arm].update(usage=dict(usage),duplicate_non_todo_calls=sum(n-1 for n in counts.values()),report_read_after_write_response=read_after_write,todo_snapshot_count=len(data['todo_snapshots']),final_todos=data['todo_snapshots'][-1] if data['todo_snapshots'] else [],tool_response_errors=[m for m in msgs if m.get('type')=='tool' and m.get('status')=='error'],report=data.get('report'))
    (root/arm/'call_sequence.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
