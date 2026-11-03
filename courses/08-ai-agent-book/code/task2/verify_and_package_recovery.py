"""Verify actual SSE reconstruction, immutable baseline, canonical hashes, and sources."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/full-60x4-recovery-20261017-01'
BASELINE=ROOT/'outputs/full-60x4-amd-20261015-01'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

assert json.loads((OUT/'recovery-status.json').read_text())['status']=='passed'
assert json.loads((OUT/'audit.json').read_text())['status']=='passed'
baseline=json.loads((OUT/'baseline-snapshot.json').read_text())
assert all(sha(BASELINE/'checkpoints'/name)==digest for name,digest in baseline.items())
for name,digest in baseline.items():
 path=BASELINE/'checkpoints'/name
 if json.loads(path.read_text()).get('status')!='completed':
  target=OUT/'baseline-origin-checkpoints'/name
  target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(path,target)
  assert sha(target)==digest
stream_checks=[]
for receipt_file in OUT.glob('receipt-*.json'):
 call=json.loads(receipt_file.read_text())
 raw=Path(call['stream_file'])
 assert sha(raw)==call['stream_sha256']
 if call.get('error'): continue
 content=[]
 reasoning=[]
 finish=None
 actual_model=None
 usage=None
 for line in raw.read_text().splitlines():
  row=json.loads(line)
  actual_model=row.get('model',actual_model)
  if row.get('usage') is not None: usage=row['usage']
  for choice in row.get('choices') or []:
   if choice.get('index',0)!=0: continue
   delta=choice.get('delta') or {}
   if delta.get('content'): content.append(delta['content'])
   if delta.get('reasoning'): reasoning.append(delta['reasoning'])
   if choice.get('finish_reason'): finish=choice['finish_reason']
 response=call['response']
 assert ''.join(content)==response['choices'][0]['message']['content']
 assert ''.join(reasoning)==response['choices'][0]['message'].get('reasoning','')
 assert finish==response['choices'][0]['finish_reason']=='stop'
 assert actual_model==response['model']
 assert usage==call.get('usage')
 stream_checks.append({'purpose':call['purpose'],'model':actual_model,'stream_sha256':sha(raw),
   'content_characters':sum(map(len,content)),'reasoning_characters':sum(map(len,reasoning)),
   'finish_reason':finish,'usage_available':usage is not None,'response_matches_stream':True})
save(OUT/'stream-audit.json',{'status':'passed','successful_streams':len(stream_checks),'checks':stream_checks,
 'baseline_checkpoints_unchanged':len(baseline)})
latest=json.loads((OUT/'canonical-latest.json').read_text())
run=Path(latest['run_dir'])
for name,digest in latest['artifacts'].items(): assert sha(run/name)==digest,name
sources=[]
for item in latest['inputs']:
 path=Path(item['path'])
 assert sha(path)==item['sha256'],str(path)
 sources.append(path)
sources.extend([ROOT/'amd_full_acceptance.py',ROOT/'verify_and_package_recovery.py',BASELINE/'adapter-v1-original.py'])
source_manifest={}
for path in sources:
 relative=path.relative_to(ROOT)
 target=OUT/'source'/relative
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(path,target)
 source_manifest[str(relative)]=sha(target)
save(OUT/'source-hashes.json',source_manifest)

notes=['# 流式恢复的证据核验','',
 '原配置基线的240份检查点字节均未改变，其中238份完成、2份未完成。新目录中的恢复记录补齐两项，含模型不合格回答及其0分，不替换成更好的答案。','',
 '新请求均记录为stream=true；实际SSE分片分别保存。以下核验不依赖拼接器自身的判断，重新从原始分片构造content与reasoning，检查结束原因、模型身份和usage字段。','',
 '| 调用 | 模型 | 最终内容字符 | 推理字段字符 | usage可用 |','|---|---|---:|---:|---|']
for row in stream_checks:
 notes.append(f"| {row['purpose']} | {row['model']} | {row['content_characters']} | {row['reasoning_characters']} | {row['usage_available']} |")
notes.extend(['','尽管请求reasoning_effort=none，本次Qwen返回流仍包含reasoning字段。这是本次观测，不能据此断言所有模型或接口均如此；预算不能仅按可见JSON长度估计。','',
 '流式接收与超时延长同时发生，因此这里只能说明组合恢复成功，未分别验证两项改动的因果贡献。提高评审上限没有修改原始答案或评分标准，但混合预算的汇总不应称为统一配置对照。','',
 '原始官方实验指针仍保留partial；恢复实验有单独的combined/validation/latest.json。两者不互相冒充。'])
(OUT/'stream-verification-notes.md').write_text('\n'.join(notes),encoding='utf-8')
files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.zip','.tmp') and p.name not in ('artifact-manifest.json','recovery.log')]
save(OUT/'artifact-manifest.json',{str(p.relative_to(OUT)):sha(p) for p in files})
with zipfile.ZipFile(OUT/'recovery-evidence.zip','w',zipfile.ZIP_DEFLATED) as package:
 for path in files+[OUT/'artifact-manifest.json']: package.write(path,path.relative_to(OUT))
with zipfile.ZipFile(OUT/'recovery-evidence.zip') as package: assert package.testzip() is None
print(json.dumps({'status':'passed','stream_count':len(stream_checks),'unchanged_baseline_files':len(baseline),
 'source_files':len(source_manifest),'canonical_hashes_verified':True,'package_bytes':(OUT/'recovery-evidence.zip').stat().st_size}))
