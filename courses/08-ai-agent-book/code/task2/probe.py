"""AMD connectivity probe. Key is entered locally and never saved."""
import getpass
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = 'https://developer.amd.com.cn/radeon/api/v1'

def main():
    key = os.environ.get('RADEON_API_KEY') or getpass.getpass('AMD API Key (hidden): ')
    if not key.strip():
        raise SystemExit('No key entered.')
    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        request = Request(BASE + path, data=data, headers={
            'Authorization': 'Bearer ' + key.strip(), 'Content-Type': 'application/json'})
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    evidence = {'base_url': BASE, 'status': 'partial'}
    try:
        models = [item['id'] for item in call('/models')['data']]
        evidence['models'] = models
        print('Available models:', ', '.join(models))
        model = input('Enter a model ID from this list: ').strip()
        if model not in models:
            raise ValueError('Model must be selected from the live list.')
        payload = {'model': model, 'messages': [{'role': 'user', 'content': 'Reply OK only.'}], 'max_tokens': 64}
        result = call('/chat/completions', payload)
        evidence.update(status='passed', request=payload, response=result)
        print('Response:', result['choices'][0]['message']['content'])
    except (HTTPError, URLError, ValueError, KeyError) as exc:
        evidence['error'] = {'type': type(exc).__name__, 'http_status': getattr(exc, 'code', None)}
        print('Probe failed:', evidence['error'])
    target = Path(__file__).parent / 'outputs' / 'connectivity.json'
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Evidence saved:', target)

if __name__ == '__main__':
    main()
