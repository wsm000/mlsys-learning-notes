"""Small live Memory teaching experiment; not the book's full 60-case benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE = 'https://developer.amd.com.cn/radeon/api/v1'
MODEL = os.environ.get('RADEON_MODEL', 'DeepSeek-V4.1-Flash')

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def chat(out, label, messages):
    key = os.environ.get('RADEON_API_KEY')
    if not key:
        key = Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip()
    payload = {'model': MODEL, 'messages': messages, 'max_tokens': 1000, 'temperature': 0, 'reasoning_effort': 'none'}
    request = Request(BASE + '/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    marker = out / 'last-call.txt'
    for attempt in range(3):
        if marker.exists():
            time.sleep(max(0, 35 - (time.time() - float(marker.read_text()))))
        marker.write_text(str(time.time()))
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            break
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace').replace(key, '[REDACTED]')
            retry_after = exc.headers.get('Retry-After', '35')
            save(out / (label + '-attempt-' + str(attempt + 1) + '.json'),
                 {'request': payload, 'http_status': exc.code, 'detail': detail, 'retry_after': retry_after})
            print('API error: ' + detail[:800], flush=True)
            if 'daily' in detail.lower() or 'quota' in detail.lower():
                raise RuntimeError('Provider quota exhausted; see saved receipt') from None
            if exc.code != 429 or attempt == 2:
                raise RuntimeError('HTTP status ' + str(exc.code)) from None
            if float(retry_after) > 120:
                raise RuntimeError('Provider requires a long pause; see receipt') from None
            time.sleep(min(60, max(0, float(retry_after))))
            print(label + ': HTTP 429, waiting before retry', flush=True)
        except (TimeoutError, OSError) as exc:
            save(out / (label + '-attempt-' + str(attempt + 1) + '.json'),
                 {'request': payload, 'error_type': type(exc).__name__})
            if attempt == 2:
                raise
            print(label + ': transport timeout/error, retrying', flush=True)
    save(out / (label + '.json'), {'pid': os.getpid(), 'request': payload,
        'response': result, 'seconds': round(time.monotonic() - started, 3)})
    content = result['choices'][0]['message']['content']
    if not content:
        raise RuntimeError('Empty content; inspect saved response.')
    return content

def extract(out, label, statement):
    text = chat(out, label, [
        {'role': 'system', 'content': '从用户原话提取明确、长期有效的教学比喻偏好。忽略今天时间限制等临时安排。不要推测。只返回 JSON 对象，格式为 {"preference":"偏好原文", "evidence":"原话中的连续引用"}。'},
        {'role': 'user', 'content': statement}])
    start, end = text.find('{'), text.rfind('}')
    data = json.loads(text[start:end + 1])
    if not all(isinstance(data.get(k), str) and data[k].strip() for k in ('preference', 'evidence')):
        raise ValueError('Invalid extracted memory schema')
    if data['evidence'] not in statement:
        raise ValueError('Evidence is not a quote from the user statement')
    if any(x in data['preference'] for x in ('十分钟', '10分钟')):
        raise ValueError('Temporary time limit leaked into long-term preference')
    return data

def answer(out, label, read_memory):
    messages = [{'role': 'system', 'content': '你是技术学习助手。请用中文简短解释，最多200字。'}]
    selected = []
    if read_memory:
        records = json.loads((out / 'memory.json').read_text(encoding='utf-8'))
        selected = [r for r in records if r['user_id'] == 'demo_user' and r['status'] == 'active']
        if len(selected) != 1:
            raise ValueError('Expected exactly one active memory')
        messages[0]['content'] += '\n当前用户的有效教学偏好：' + selected[0]['preference']
    save(out / (label + '-read.json'), {'pid': os.getpid(), 'read_enabled': read_memory,
        'selected_memories': selected, 'old_conversation_included': False})
    messages.append({'role': 'user', 'content': '什么是 Embedding？'})
    content = chat(out, label, messages)
    print(label + ': ' + content, flush=True)

def run(out):
    save(out / 'environment.json', {'hostname': socket.gethostname(), 'python': platform.python_version(),
        'model': MODEL, 'base_url': BASE, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': '教学最小用户记忆实验；不是教材3-1/3-2完整评测', 'parent_pid': os.getpid()})
    statements = ['解释技术概念时，请先用养蜂作比喻。今天我只有十分钟。',
                  '以后解释技术概念时，改用做饭作比喻，不再用养蜂。']
    save(out / 'user-statements.json', statements)
    first = extract(out, '01-extract', statements[0])
    records = [dict(first, id='m1', user_id='demo_user', status='active', revision=1)]
    save(out / 'memory.json', records)
    save(out / 'memory-before-update.json', records)
    print('01: preference extracted and persisted', flush=True)
    for label, enabled in [('02-without-memory', False), ('03-with-memory', True)]:
        args = [sys.executable, str(Path(__file__).resolve()), '--out', str(out), '--answer', label]
        if enabled:
            args.append('--read-memory')
        subprocess.run(args, check=True)
    new = extract(out, '04-extract-update', statements[1])
    records[0].update(status='superseded', superseded_by='m2')
    records.append(dict(new, id='m2', user_id='demo_user', status='active', revision=2))
    save(out / 'memory.json', records)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(out),
                    '--answer', '05-after-update', '--read-memory'], check=True)
    reads = {name: json.loads((out / (name + '-read.json')).read_text()) for name in
             ['02-without-memory', '03-with-memory', '05-after-update']}
    checks = {'no_memory_condition_empty': not reads['02-without-memory']['selected_memories'],
              'initial_memory_read': reads['03-with-memory']['selected_memories'][0]['id'] == 'm1',
              'updated_memory_read': reads['05-after-update']['selected_memories'][0]['id'] == 'm2',
              'fresh_processes': len({r['pid'] for r in reads.values()} | {os.getpid()}) == 4,
              'old_memory_retained_inactive': records[0]['status'] == 'superseded',
              'no_raw_old_conversation': all(not r['old_conversation_included'] for r in reads.values())}
    save(out / 'checks.json', {'checks': checks, 'mechanism_passed': all(checks.values()),
        'limitation': '单用户教学机制检查；回答需人工核对；不是独立LLM评分，也不是教材完整评测。'})
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.json'))}
    save(out / 'manifest.json', manifest)
    print('MECHANISM_CHECKS ' + json.dumps(checks), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--answer')
    parser.add_argument('--read-memory', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.answer:
        answer(args.out, args.answer, args.read_memory)
    else:
        run(args.out.resolve())
