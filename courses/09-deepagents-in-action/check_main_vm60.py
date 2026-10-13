import json
from pathlib import Path
import uuid
import httpx

base = 'http://127.0.0.1:18088/agent'
with httpx.Client(timeout=120, trust_env=False) as client:
    response = client.get(base+'/health')
    print('Health:', response.status_code)
    response.raise_for_status()
    payload = {'threadId': str(uuid.uuid4()), 'runId': str(uuid.uuid4()), 'state': {},
               'messages': [{'id': str(uuid.uuid4()), 'role': 'user', 'content': 'Analyze DEMO latest 3 records versus preceding 3 records. Use the tool and explain the data limitations.'}],
               'tools': [], 'context': [], 'forwardedProps': {}}
    events = []
    with client.stream('POST', base, json=payload) as response:
        print('Run HTTP:', response.status_code)
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith('data:'):
                event = json.loads(line[5:].strip())
                events.append(event)
                Path('/home/simin/course09-quanty/outputs/main-app-events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2))
                print(event.get('type'), event.get('toolCallName', ''), flush=True)
out = Path('/home/simin/course09-quanty/outputs/main-app-events.json')
out.write_text(json.dumps(events, ensure_ascii=False, indent=2))
print('Saved', out)
if any(e.get('type') == 'RUN_ERROR' for e in events):
    raise SystemExit('Gateway run failed; inspect saved event evidence')
