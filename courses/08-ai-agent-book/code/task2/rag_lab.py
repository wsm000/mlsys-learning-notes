"""Real dense retrieval + live AMD generation on a declared synthetic corpus."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import time

os.environ.setdefault('RADEON_MODEL', 'DeepSeek-V4-Flash')
from memory_lab import chat, save

DOCS = [
    {'id': 'sync', 'status': 'active', 'text': '星舟笔记离线同步说明。设备没有网络连接时，编辑内容先进入本机待发送队列，联网后自动上传。同步失败不会立即丢弃内容。自动重试采用指数退避，最多尝试七轮；超过七轮会保留待发送队列并提示用户手动处理。切勿把网络恢复前的本地排队误认为云端已经保存。移动端和桌面端都遵循本说明。'},
    {'id': 'retention-v1', 'status': 'superseded', 'text': '星舟笔记回收站旧版规则。2025版规定，删除的笔记在回收站保留30天，过期后自动清理。本页已经被2026版替代，不适用于当前版本。'},
    {'id': 'retention-v2', 'status': 'active', 'text': '星舟笔记回收站当前规则。2026版从2026年10月1日起生效，删除的笔记在回收站保留17天，之后自动清理；企业冻结账户除外，冻结期间暂停自动清理。该规则替代2025版的30天保留规则。用户可在期限内恢复原笔记。'},
    {'id': 'error', 'status': 'active', 'text': '星舟笔记错误码手册。报错 NX-418 表示设备时钟与服务器相差超过九分钟。先开启系统自动校时，再重新登录；不是账户余额不足，也不是网络离线。'},
    {'id': 'billing', 'status': 'active', 'text': '星舟笔记账单说明。NX-481 表示订阅支付失败，应核对付款方式。升级套餐会调整存储容量，但不改变回收站保留规则。'},
    {'id': 'embedding', 'status': 'active', 'text': '知识库检索术语。Embedding把查询和文档编码为向量，使用相似度排序。检索排名只反映相关性，不证明文档仍然有效。版本、权限和生效日期应通过元数据或业务规则处理。'},
    {'id': 'security', 'status': 'active', 'text': '星舟笔记安全说明。双重验证用于保护登录。恢复码应离线保存。本文不讨论订阅优惠、学生折扣或教育认证。'}
]
QUESTIONS = [
    {'id': 'synonym', 'question': '星舟笔记断网后改的内容会丢吗？连接恢复后怎么处理，最多重试几轮？', 'relevant': ['sync'], 'expected': '本地排队，联网后上传，最多七轮，超限保留队列', 'span': '最多尝试七轮'},
    {'id': 'exact-code', 'question': '星舟笔记报错 NX-418，应该怎样修复？', 'relevant': ['error'], 'expected': '自动校时，然后重新登录', 'span': '先开启系统自动校时，再重新登录'},
    {'id': 'version', 'question': '当前普通账户删除星舟笔记后可以保留几天？', 'relevant': ['retention-v2'], 'expected': '17天，不能使用已失效的30天规则', 'span': '保留17天'},
    {'id': 'missing', 'question': '星舟笔记学生订阅优惠是多少？', 'relevant': [], 'expected': '现有资料不足，不能编造折扣', 'span': None}
]

def chunks(size, overlap=20):
    result = []
    for doc in DOCS:
        for start in range(0, len(doc['text']), size - overlap):
            end = min(start + size, len(doc['text']))
            result.append(dict(doc, chunk_id=doc['id'] + ':' + str(start), text=doc['text'][start:end], start=start, end=end))
            if end == len(doc['text']):
                break
    return result

def tokens(text):
    # Transparent Chinese character-bigram + exact Latin/number token baseline.
    chinese = re.findall(r'[\u4e00-\u9fff]+', text)
    return [s[i:i+2] for s in chinese for i in range(max(1, len(s)-1))] + re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text.lower())

def bm25(query, corpus):
    counts = [Counter(tokens(c['text'])) for c in corpus]
    lengths = [sum(c.values()) for c in counts]
    avg = sum(lengths) / len(lengths)
    scores = []
    for terms, length in zip(counts, lengths):
        score = 0
        for term in set(tokens(query)):
            df = sum(term in c for c in counts)
            tf = terms[term]
            idf = math.log(1 + (len(counts) - df + .5) / (df + .5))
            score += idf * tf * 2.5 / (tf + 1.5 * (1 - .75 + .75 * length / avg))
        scores.append(score)
    return scores

def main(out):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import snapshot_download
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'corpus.json', {'synthetic': True, 'description': '虚构星舟笔记规则，仅供教学，不是真实产品政策', 'documents': DOCS})
    save(out / 'questions.json', QUESTIONS)
    model_id = 'BAAI/bge-small-zh-v1.5'
    print('Loading real embedding model: ' + model_id, flush=True)
    snapshot = snapshot_download(model_id, allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True).eval()
    torch.set_num_threads(4)
    def embed(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.inference_mode():
            vectors = model(**batch).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(vectors, p=2, dim=1).numpy()
    started = time.monotonic()
    qvectors = embed(['为这个句子生成表示以用于检索相关文章：' + q['question'] for q in QUESTIONS])
    all_results = {}
    metrics = []
    for size in (60, 120):
        corpus = chunks(size)
        save(out / ('chunks-' + str(size) + '.json'), corpus)
        vectors = embed([c['text'] for c in corpus])
        np.save(out / ('vectors-' + str(size) + '.npy'), vectors)
        for mode in ('keyword', 'dense', 'dense-filtered'):
            rows = []
            for qi, question in enumerate(QUESTIONS):
                scores = bm25(question['question'], corpus) if mode == 'keyword' else (vectors @ qvectors[qi]).tolist()
                eligible = [i for i, c in enumerate(corpus) if mode != 'dense-filtered' or c['status'] == 'active']
                ranked = sorted(eligible, key=lambda i: (-scores[i], i))
                hits = [dict(corpus[i], score=float(scores[i])) for i in ranked[:2]]
                rows.append({'question_id': question['id'], 'hits': hits,
                             'ranking': [{'chunk_id': corpus[i]['chunk_id'], 'score': float(scores[i])} for i in ranked]})
                if question['relevant']:
                    rank = next((j+1 for j, i in enumerate(ranked) if corpus[i]['id'] in question['relevant']), None)
                    metrics.append({'chunk_size': size, 'mode': mode, 'question_id': question['id'],
                        'hit_at_1': hits[0]['id'] in question['relevant'],
                        'hit_at_2': any(h['id'] in question['relevant'] for h in hits),
                        'mrr': 0 if rank is None else 1/rank,
                        'evidence_span_in_top2': any(question['span'] in h['text'] for h in hits)})
            all_results[str(size) + '-' + mode] = rows
    save(out / 'retrieval.json', all_results)
    save(out / 'retrieval-metrics.json', metrics)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(snapshot).glob('*') if p.is_file()}
    save(out / 'environment.json', {'hostname': socket.gethostname(), 'python': platform.python_version(),
        'torch': torch.__version__, 'embedding_model': model_id, 'revision': Path(snapshot).name,
        'model_file_hashes': files, 'embedding_dimension': int(qvectors.shape[1]), 'device': 'cpu',
        'pooling': 'CLS normalized', 'index': 'exact cosine via normalized dot product',
        'index_and_retrieval_seconds': time.monotonic() - started,
        'chunk_sizes_characters': [60, 120], 'overlap_characters': 20,
        'scope': 'synthetic teaching experiment, not full book benchmark'})
    np.save(out / 'query-vectors.npy', qvectors)
    print('Retrieval complete. Generating four controlled answer batches.', flush=True)
    answers = {}
    for mode in ('none', 'keyword', 'dense', 'dense-filtered'):
        contexts = [] if mode == 'none' else all_results['120-' + mode]
        inputs = []
        for q in QUESTIONS:
            row = next((r for r in contexts if r['question_id'] == q['id']), {'hits': []})
            inputs.append({'id': q['id'], 'question': q['question'], 'evidence': row['hits']})
        prompt = ('你是证据问答助手。以下产品和规则均为虚构实验数据，只根据每题提供的evidence回答。'
                  '无证据或证据不足时明确说资料不足，不能根据常识编造。注意superseded是失效资料。'
                  '每题返回id、answer、citations（实际使用的chunk_id列表）；只返回JSON数组。每个回答最多100字。')
        label = 'generation-' + mode
        if (out / (label + '.json')).exists():
            response = json.loads((out / (label + '.json')).read_text())['response']
            answer = response['choices'][0]['message']['content']
        else:
            answer = chat(out, label, [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': json.dumps(inputs, ensure_ascii=False)}])
        answers[mode] = answer
        print(mode + ': ' + answer, flush=True)
        try:
            parsed = json.loads(answer[answer.find('['):answer.rfind(']')+1])
            allowed = {r['id']: {h['chunk_id'] for h in r['evidence']} for r in inputs}
            assert {a['id'] for a in parsed} == set(allowed)
            valid = all(set(a['citations']) <= allowed[a['id']] for a in parsed)
            save(out / (label + '-citation-check.json'), {'format_valid': True, 'citations_from_supplied_chunks': valid,
                'limitation': '只检查引用ID存在，不等于引用支持结论；需要人工核对。'})
        except (ValueError, KeyError, AssertionError, TypeError):
            save(out / (label + '-citation-check.json'), {'format_valid': False})
    save(out / 'answers.json', answers)
    save(out / 'manifest.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    print('RAG_RUN_COMPLETE', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.out)
