import json
from pathlib import Path
import time
from openai import OpenAI

client = OpenAI(api_key=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
    base_url='https://developer.amd.com.cn/radeon/api/v1', timeout=20, max_retries=0)
record = {}
for task in ('models', 'minimal_chat'):
    started = time.monotonic()
    try:
        if task == 'models':
            result = [m.id for m in client.models.list()]
        else:
            response = client.chat.completions.create(model='DeepSeek-V4-Flash',
                messages=[{'role': 'user', 'content': 'Reply OK only.'}], max_tokens=32)
            result = response.choices[0].message.content
        record[task] = {'success': True, 'result': result}
    except Exception as exc:
        record[task] = {'success': False, 'error_type': type(exc).__name__,
            'http_status': getattr(exc, 'status_code', None)}
    record[task]['seconds'] = round(time.monotonic()-started, 2)
    print(json.dumps({task: record[task]}, ensure_ascii=False), flush=True)
Path('/home/simin/course09-quanty/outputs/task2-connectivity-probe.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
