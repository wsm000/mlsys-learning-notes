import json
import hashlib
import os
from pathlib import Path
os.environ.setdefault('RADEON_MODEL', 'DeepSeek-V4-Flash')
from memory_lab import chat, save

root = Path('/home/simin/projects/task2-memory-rag/outputs')
source = root / 'rag-20261015-01'
out = root / 'rag-followup-20261015-01'
out.mkdir(exist_ok=True)
retrieval = json.loads((source / 'retrieval.json').read_text())
questions = json.loads((source / 'questions.json').read_text())
question = questions[0]['question']
hits = retrieval['60-dense'][0]['hits']
answer = chat(out, 'small-chunk-answer', [
    {'role': 'system', 'content': '只根据提供证据回答虚构产品问题。缺少的细节必须说资料不足，不要推测。指出实际引用的chunk_id。用中文，最多150字。'},
    {'role': 'user', 'content': json.dumps({'question': question, 'evidence': hits}, ensure_ascii=False)}])
print('SMALL_CHUNKS: ' + answer, flush=True)
records = json.loads((root / 'memory-20261015-05' / 'memory.json').read_text())
active = [r for r in records if r['user_id'] == 'demo_user' and r['status'] == 'active']
save(out / 'read-user-memory.json', active)
assert len(active) == 1
hits = retrieval['120-dense-filtered'][0]['hits']
answer2 = chat(out, 'memory-plus-rag', [
    {'role': 'system', 'content': '只根据证据回答虚构产品问题；缺失事实明确说资料不足，引用chunk_id。教学比喻不能添加产品事实。用户有效偏好：' + active[0]['preference'] + '。用中文，最多200字。'},
    {'role': 'user', 'content': json.dumps({'question': question, 'evidence': hits}, ensure_ascii=False)}])
print('MEMORY_PLUS_RAG: ' + answer2, flush=True)
save(out / 'summary.json', {'small_chunk_answer': answer, 'memory_plus_rag': answer2,
    'limitation': '第二轮同时改变分块和偏好，不能用两轮差异单独估计记忆效应；记忆独立对照见上一实验。'})
save(out / 'manifest.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json') if p.name != 'manifest.json'})
