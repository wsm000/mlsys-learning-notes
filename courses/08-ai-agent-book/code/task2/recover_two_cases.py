"""Isolated streaming recovery; never overwrites the 238/240 baseline."""
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from openai import OpenAI
from openai.types.chat import ChatCompletion
import httpx

ROOT=Path('/home/simin/projects/task2-memory-rag')
BASELINE=ROOT/'outputs/full-60x4-amd-20261015-01'
OUT=ROOT/'outputs/full-60x4-recovery-20261017-01'
OUT.mkdir(exist_ok=True)
BOOK=ROOT/'official-book'
KEY=os.getenv('RADEON_API_KEY') or Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip()
ENDPOINT='https://developer.amd.com.cn/radeon/api/v1'
spec=importlib.util.spec_from_file_location('recovery_official',BOOK/'chapter3/user-memory/run_evaluation.py')
official=importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
source=json.loads((BASELINE/'preflight.json').read_text())
TARGETS=[('layer2_07_multiple_medications','advanced_json_cards'),('layer3_13_emergency_medical_cascade','json_cards')]
LOCK=threading.Lock()
NEXT_START=0.0

def save(path,data):
 temp=path.with_suffix('.'+str(threading.get_ident())+'.tmp')
 temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 temp.replace(path)

# Reuse only the standalone validator function, without executing its old campaign.
tree=ast.parse((ROOT/'amd_full_acceptance.py').read_text())
validator=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_response')
namespace={'official':official}
exec(compile(ast.Module(body=[validator],type_ignores=[]),'recovery_validator','exec'),namespace)

class StreamRecorder(official.ChatRecorder):
 def __init__(self,client,provider,endpoint): super().__init__(client,'amd',endpoint)
 def create(self,*,purpose,**request):
  global NEXT_START
  request.pop('seed',None)
  request.update(stream=True,reasoning_effort='none')
  for attempt in range(3):
   with LOCK:
    delay=max(0,NEXT_START-time.monotonic())
    NEXT_START=max(time.monotonic(),NEXT_START)+4
   if delay: time.sleep(delay)
   uid=uuid.uuid4().hex
   rawpath=OUT/('stream-'+uid+'.jsonl')
   record={'purpose':purpose,'provider':'amd','endpoint':ENDPOINT,'request':request,
    'started_at':datetime.now(timezone.utc).isoformat(),'stream_file':str(rawpath),'attempt':attempt+1,
    'response_assembly':'Concatenated delta.content from raw SSE chunks; no content rewriting'}
   started=time.monotonic()
   pieces=[]
   reasoning=[]
   usage=None
   finish=None
   identity={}
   last_notice=started
   try:
    with rawpath.open('w',encoding='utf-8') as raw:
     with self.client.chat.completions.create(**request) as stream:
      for chunk in stream:
       data=chunk.model_dump(mode='json')
       raw.write(json.dumps(data,ensure_ascii=False)+'\n')
       raw.flush()
       identity.update({k:data[k] for k in ('id','created','model','system_fingerprint') if data.get(k) is not None})
       if data.get('usage') is not None: usage=data['usage']
       for choice in data.get('choices') or []:
        if choice.get('index',0)!=0: continue
        delta=choice.get('delta') or {}
        if delta.get('content'): pieces.append(delta['content'])
        if delta.get('reasoning'): reasoning.append(delta['reasoning'])
        if choice.get('finish_reason'): finish=choice['finish_reason']
       if time.monotonic()-last_notice>=20:
        print('STREAM_PROGRESS '+purpose+' chars='+str(sum(map(len,pieces))),flush=True)
        last_notice=time.monotonic()
    if finish is None: raise ValueError('Stream ended without finish_reason')
    assembled={'id':identity.get('id','stream-'+uid),'object':'chat.completion',
      'created':identity.get('created',int(time.time())),'model':identity.get('model',request['model']),
      'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':''.join(pieces)}}],
      'usage':usage}
    if reasoning: assembled['choices'][0]['message']['reasoning']=''.join(reasoning)
    response=ChatCompletion.model_validate(assembled)
    record.update(response=assembled,usage=usage,latency_ms=round((time.monotonic()-started)*1000,3),
      stream_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),
      usage_available=usage is not None)
    namespace['validate_response'](purpose,response)
    self.calls.append(record)
    save(OUT/('receipt-'+uid+'.json'),record)
    print('CALL_COMPLETE '+purpose+' finish='+finish,flush=True)
    return response
   except Exception as exc:
    record['latency_ms']=round((time.monotonic()-started)*1000,3)
    record['error']={'type':type(exc).__name__,'message':str(exc).replace(KEY,'[REDACTED]')}
    if rawpath.exists(): record['stream_sha256']=hashlib.sha256(rawpath.read_bytes()).hexdigest()
    self.calls.append(record)
    save(OUT/('receipt-'+uid+'.json'),record)
    print('RECOVERY_ERROR '+purpose+' '+str(record['error'])[:500],flush=True)
    status=getattr(exc,'status_code',None)
    if attempt==2 or status in (400,401,403,404): raise
    retry=getattr(getattr(exc,'response',None),'headers',{}).get('retry-after','10')
    wait=float(retry or 10)
    if wait>120: raise RuntimeError('Provider requires extended wait') from None
    time.sleep(max(5,wait))

official.ChatRecorder=StreamRecorder
os.environ['ARK_API_KEY']=KEY
os.environ['MOONSHOT_API_KEY']=KEY
args=official.build_parser().parse_args(['--all','--workers','2','--writer-endpoint',ENDPOINT,'--judge-endpoint',ENDPOINT,
 '--writer-model','DeepSeek-V4-Flash','--judge-model','Qwen3.8-Flash-Next','--timeout','600',
 '--memory-max-tokens','6000','--answer-max-tokens','1600','--judge-max-tokens','8192',
 '--checkpoint-dir',str(OUT/'checkpoints')])
cases=official.load_cases(args.test_cases_dir,args)
by_id={case['test_id']:case for case in cases}
campaign=official.Campaign(args)
campaign.writer_client=OpenAI(api_key=KEY,base_url=ENDPOINT,max_retries=0,timeout=httpx.Timeout(600,connect=30,write=60,pool=30))
campaign.judge_client=OpenAI(api_key=KEY,base_url=ENDPOINT,max_retries=0,timeout=httpx.Timeout(600,connect=30,write=60,pool=30))
campaign.checkpoint_signature.update(source_commit=source['commit'],recovery_sha256=script_hash,
 memory_max_tokens=6000,answer_max_tokens=1600,judge_max_tokens=8192,stream=True,read_timeout_seconds=600)
baseline_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASELINE/'checkpoints').glob('*.json')}
if (OUT/'baseline-snapshot.json').exists():
 assert json.loads((OUT/'baseline-snapshot.json').read_text())==baseline_hashes,'Baseline changed since recovery started'
else: save(OUT/'baseline-snapshot.json',baseline_hashes)
shutil.copy2(BASELINE/'preflight.json',OUT/'preflight.json')
for path in (BASELINE/'checkpoints').glob('*.json'):
 target=OUT/'checkpoints'/path.name
 if target.exists(): continue
 cp=json.loads(path.read_text())
 if (cp['test_id'],cp['mode']) in TARGETS:
  cp['recovery_origin']={'path':str(path),'sha256':baseline_hashes[path.name],'signature':cp['signature'],
   'changes':{'transport':'SSE streaming','read_timeout':'180 -> 600 seconds','judge_max_tokens':'2400 -> 8192'},
   'unchanged':['models','memory/answer token budgets','prompts','rubric','existing states and responses']}
  cp['signature']=campaign.checkpoint_signature
  save(target,cp)
 else:
  assert cp.get('status')=='completed'
  shutil.copy2(path,target)
save(OUT/'recovery-policy.json',{'targets':TARGETS,'models_unchanged':True,'prompts_unchanged':True,
 'baseline_status':'partial 238/240; retained unchanged','recovery_judge_max_tokens':8192,
 'baseline_judge_max_tokens':2400,'transport':'streaming for new calls only','read_timeout_seconds':600,
 'memory_max_tokens':6000,'answer_max_tokens':1600,
 'comparison_note':'Combined descriptive results use two recovery configurations; not a uniform-budget rerun.'})
save(OUT/'recovery-status.json',{'status':'running','target_count':2,'baseline_completed':238})
errors=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures={pool.submit(campaign.run_one,by_id[test_id],mode):(test_id,mode) for test_id,mode in TARGETS}
 for future in as_completed(futures):
  identity=futures[future]
  try:
   result=future.result()
   print('RECOVERED '+str(identity)+' '+json.dumps(result['judge']),flush=True)
  except Exception as exc:
   errors.append({'target':identity,'type':type(exc).__name__,'error':str(exc).replace(KEY,'[REDACTED]')})

assert all(hashlib.sha256((BASELINE/'checkpoints'/name).read_bytes()).hexdigest()==digest for name,digest in baseline_hashes.items())
environment={**os.environ,'AUDIT_CAMPAIGN_DIR':str(OUT)}
subprocess.run(['python3',str(ROOT/'audit_full_acceptance.py')],env=environment,check=True)
audit=json.loads((OUT/'audit.json').read_text())
results=[]
receipts=[]
for path in sorted((OUT/'checkpoints').glob('*.json')):
 cp=json.loads(path.read_text())
 if cp.get('status')=='completed':
  results.append(cp['result'])
  receipts.extend(cp.get('writer_calls',[])+cp.get('judge_calls',[]))
evidence={'status':'passed' if audit['status']=='passed' and not errors else 'partial',
 'scope':{'baseline_completed':238,'coverage_completed':len(results),'coverage_expected':240,'recovery_targets':TARGETS},
 'acceptance':{'all_240_completed':len(results)==240,'independent_audit':audit['status']=='passed',
   'baseline_unchanged':True,'uniform_original_configuration':False},
 'configuration':json.loads((OUT/'recovery-policy.json').read_text()),
 'summary':{'aggregate':official.aggregate(results),'recovery_errors':errors},'results':results}
manifest=official.write_campaign_evidence(OUT/'combined','3-1-and-3-2-recovery',evidence,receipts,
 input_paths=[Path(__file__),ROOT/'audit_full_acceptance.py',BOOK/'chapter3/user-memory/run_evaluation.py',
 OUT/'recovery-policy.json',OUT/'baseline-snapshot.json',*[Path(case['_path']) for case in cases]])
save(OUT/'canonical-latest.json',manifest)
save(OUT/'recovery-status.json',{'status':evidence['status'],'coverage_completed':len(results),'baseline_unchanged':True,
 'audit_errors':audit['errors'],'recovery_errors':errors,'uniform_original_configuration':False})
lines=['# 3-1／3-2定向恢复实验','',f"恢复后覆盖：{len(results)}/240；状态：{evidence['status']}。",'',
 '原配置238/240结果完全保留。本次只恢复两项技术失败，不重跑已完成的低分答案。','',
 '多药物第三次记忆更新改用流式接收，记忆预算仍为6000 tokens。紧急医疗既有答案保持不变，评审预算由2400提高为8192。新调用读取超时600秒，模型、官方提示和评分标准不变。','',
 '这是混合配置下的恢复覆盖，不是原配置统一预算的240项重跑。质量分数为独立模型判断，流程通过也不等于每题正确。流式接口若未返回usage，新增调用的token与成本记为不可用，不作零成本解释。','',
 '| 模式 | 完成数 | 质量通过率 | 平均奖励 | 幻觉否决率 |','|---|---:|---:|---:|---:|']
for mode,groups in audit['aggregate'].items():
 row=groups['overall']
 if row['completed']:
  lines.append(f"| {mode} | {row['completed']}/60 | {row['quality_pass_rate']:.1%} | {row['mean_reward']:.3f} | {row['hallucination_rate']:.1%} |")
lines.extend(['','## 恢复结果',''])
for identity in TARGETS:
 cp=json.loads((OUT/'checkpoints'/(identity[0]+'--'+identity[1]+'.json')).read_text())
 lines.append('- '+identity[0]+' / '+identity[1]+'：'+cp['status'])
 if cp.get('result'): lines.append('  '+json.dumps(cp['result']['judge'],ensure_ascii=False))
if errors: lines.extend(['','## 未解决问题','',json.dumps(errors,ensure_ascii=False,indent=2)])
(OUT/'recovery-report.md').write_text('\n'.join(lines),encoding='utf-8')
files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.zip') and p.name not in ('recovery.log','artifact-manifest.json')]
save(OUT/'artifact-manifest.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
with zipfile.ZipFile(OUT/'recovery-evidence.zip','w',zipfile.ZIP_DEFLATED) as package:
 for path in files+[OUT/'artifact-manifest.json']: package.write(path,path.relative_to(OUT))
print('RECOVERY_FINISHED '+json.dumps(json.loads((OUT/'recovery-status.json').read_text())),flush=True)
