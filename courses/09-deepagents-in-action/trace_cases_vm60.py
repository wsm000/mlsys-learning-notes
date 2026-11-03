import json
import os
from pathlib import Path
import uuid
from execution_trace import ExecutionTrace

os.environ.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
                  BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
                  BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from my_deepagent.demo_binding import build_agent

out = Path('/home/simin/course09-quanty/outputs')
for name, question in [
    ('normal3', 'Analyze DEMO latest 3 records versus preceding 3 records.'),
    ('unsupported100', 'Analyze DEMO latest 100 records versus preceding 100 records.')]:
    trace = ExecutionTrace()
    root_id = uuid.uuid4()
    record = {'case': name, 'root_id': str(root_id), 'input': question,
              'format': 'local callback execution spans, not a LangSmith cloud trace'}
    try:
        result = build_agent().invoke({'messages': [{'role': 'user', 'content': question}]},
                                      config={'callbacks': [trace], 'run_id': root_id})
        messages = [m.model_dump(mode='json') for m in result['messages']]
        calls = [c for m in messages for c in m.get('tool_calls', [])]
        record.update(status='completed', messages=messages, calls=calls)
        if name == 'normal3':
            matched = [c for c in calls if c['name'] == 'analyze_price_volume' and int(c['args']['window']) == 3]
            returns = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
            record['behavior_check'] = bool(matched) and all(returns.get(c['id'], {}).get('status') == 'success' for c in matched)
        else:
            record['behavior_check'] = all(c['name'] in {'ls', 'read_file', 'glob', 'grep'} for c in calls)
            record['zero_unnecessary_tools'] = not calls
        print(name, 'completed', 'calls:', [c['name'] for c in calls], 'check:', record['behavior_check'], flush=True)
        print(messages[-1]['content'], flush=True)
    except Exception as error:
        record.update(status='error', error_type=type(error).__name__)
        print(name, 'failed:', type(error).__name__, flush=True)
    finally:
        record['spans'] = trace.export()
        (out / f'execution-trace-{name}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
