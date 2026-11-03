import json
from pathlib import Path
from openai import OpenAI

key = Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip()
client = OpenAI(api_key=key, base_url='https://developer.amd.com.cn/radeon/api/v1', timeout=30, max_retries=0)
try:
    response = client.chat.completions.create(model='MiMo-V2.6-Flash', messages=[{'role': 'user', 'content': 'Reply OK only.'}], max_tokens=32)
    record = {'status': 'success', 'answer': response.choices[0].message.content}
except Exception as exc:
    record = {'status': 'failed', 'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
Path('outputs').mkdir(exist_ok=True)
Path('outputs/connectivity.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
