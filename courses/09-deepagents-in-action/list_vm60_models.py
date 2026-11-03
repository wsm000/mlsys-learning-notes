from pathlib import Path
from openai import OpenAI
client = OpenAI(api_key=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(), base_url='https://developer.amd.com.cn/radeon/api/v1', timeout=30, max_retries=0)
for model in client.models.list():
    print(model.id)
