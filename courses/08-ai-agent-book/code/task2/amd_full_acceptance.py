"""Run the pinned official 3-1/3-2 campaign through AMD with audit safeguards."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from openai import OpenAI

ROOT=Path('/home/simin/projects/task2-memory-rag')
BOOK=ROOT/'official-book'
HERE=BOOK/'chapter3/user-memory'
OUT=ROOT/'outputs/full-60x4-amd-20261015-01'
OUT.mkdir(exist_ok=True)
KEY=os.getenv('RADEON_API_KEY') or (Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip())
BASE='https://developer.amd.com.cn/radeon/api/v1'
WRITER='DeepSeek-V4-Flash'
JUDGE='Qwen3.8-Flash-Next'
spec=importlib.util.spec_from_file_location('official_memory_campaign',HERE/'run_evaluation.py')
official=importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
SOURCE=json.loads((BOOK/'source-manifest.json').read_text())
SCRIPT_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LOCK=threading.Lock()
JOURNAL_LOCK=threading.Lock()
NEXT_START=0.0
BLOCKED=threading.Event()
COUNTS=Counter()
BLOCK_REASON=None

def write(path,data):
 temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 temp.replace(path)

def throttle():
 global NEXT_START
 with LOCK:
  delay=max(0,NEXT_START-time.monotonic())
  NEXT_START=max(time.monotonic(),NEXT_START)+3.5
 if delay: time.sleep(delay)
 if BLOCKED.is_set(): raise RuntimeError('Provider blocked: '+str(BLOCK_REASON))

def journal(call):
 sanitized=json.loads(json.dumps(call,ensure_ascii=False).replace(KEY,'[REDACTED]'))
 with JOURNAL_LOCK:
  with (OUT/'live-receipts.jsonl').open('a',encoding='utf-8') as f:
   f.write(json.dumps(sanitized,ensure_ascii=False)+'\n')
  COUNTS['attempts']+=1
  COUNTS['errors' if 'error' in call else 'successful_calls']+=1
  usage=call.get('usage') or {}
  COUNTS['total_tokens']+=int(usage.get('total_tokens') or 0)
  COUNTS['reported_cost']+=float(usage.get('cost') or 0)
  write(OUT/'progress.json',{'status':'blocked' if BLOCKED.is_set() else 'running',
    'counts':dict(COUNTS),'last_purpose':call.get('purpose'),'updated_epoch':time.time(),
    'checkpoint_dir':str(OUT/'checkpoints'),'block_reason':BLOCK_REASON})

def validate_response(purpose,response):
 choice=response.choices[0]
 if choice.finish_reason=='length': raise ValueError('Output truncated at max_tokens')
 text=choice.message.content
 if not text or not text.strip(): raise ValueError('Empty response content')
 if 'memory update ' in purpose:
  value=official.parse_json(text)
  if not isinstance(value,dict): raise ValueError('Memory response requires JSON object')
  memory=value.get('memory',value)  # Match the official runner's root-object fallback.
  mode=next(m for m in official.MODES if ' '+m+' session ' in purpose)
  if mode=='json_cards':
   if not isinstance(memory,dict): raise ValueError('json_cards requires hierarchical object')
  else:
   if not isinstance(memory,list): raise ValueError(mode+' requires array')
   if mode=='advanced_json_cards':
    required={'category','card_key','backstory','person','relationship','timestamp','status','facts'}
    if any(not isinstance(c,dict) or not required<=set(c) for c in memory): raise ValueError('Incomplete advanced card')
   elif any(not isinstance(item,str) for item in memory): raise ValueError('Notes must be strings')
 elif 'independent judge ' in purpose:
  value=official.parse_json(text)
  dims=value.get('dimensions') or {}
  if set(dims)!={'precision','recall','reasoning','proactivity'}: raise ValueError('Incomplete judge dimensions')
  for v in dims.values():
   if type(v.get('score')) is not int or not 1<=v['score']<=4: raise ValueError('Invalid judge score')
   if not v.get('reasoning') or not v.get('evidence'): raise ValueError('Missing judge reasoning/evidence')
  if type((value.get('hallucination') or {}).get('detected')) is not bool: raise ValueError('Missing hallucination verdict')

class AMDRecorder(official.ChatRecorder):
 def __init__(self,client,provider,endpoint):
  super().__init__(client,'amd',endpoint)
 def create(self,*,purpose,**request):
  global BLOCK_REASON
  request.pop('seed',None)  # AMD explicitly drops seed; do not imply determinism.
  request['reasoning_effort']='none'
  for attempt in range(4):
   throttle()
   before=len(self.calls)
   try:
    response=super().create(purpose=purpose,**request)
    validate_response(purpose,response)
    journal(self.calls[-1])
    return response
   except Exception as exc:
    if len(self.calls)==before:
     self.calls.append({'purpose':purpose,'provider':'amd','endpoint':BASE,'request':request,
       'error':{'type':type(exc).__name__,'message':str(exc).replace(KEY,'[REDACTED]')}})
    else:
     self.calls[-1]['error']={'type':type(exc).__name__,'message':str(exc).replace(KEY,'[REDACTED]')}
    message=str(exc).lower()
    if getattr(exc,'status_code',None) in (401,403) or 'daily usage' in message or 'quota exhausted' in message:
     BLOCK_REASON=str(exc).replace(KEY,'[REDACTED]')[:1200]
     BLOCKED.set()
    journal(self.calls[-1])
    if BLOCKED.is_set() or attempt==3: raise
    status=getattr(exc,'status_code',None)
    if status is not None and status not in (429,500,502,503,504): raise
    retry=getattr(getattr(exc,'response',None),'headers',{}).get('retry-after','0')
    delay=max(2*(attempt+1),float(retry or 0))
    if delay>120:
     BLOCK_REASON='Provider requires extended Retry-After '+str(delay)
     BLOCKED.set()
     raise
    print('RETRY '+purpose+' '+type(exc).__name__,flush=True)
    time.sleep(delay)

original_campaign=official.Campaign
class AMDCampaign(original_campaign):
 def __init__(self,args):
  os.environ['ARK_API_KEY']=KEY
  os.environ['MOONSHOT_API_KEY']=KEY
  super().__init__(args)
  self.writer_client=OpenAI(api_key=KEY,base_url=BASE,timeout=args.timeout,max_retries=0)
  self.judge_client=OpenAI(api_key=KEY,base_url=BASE,timeout=args.timeout,max_retries=0)
  assert args.writer_model!=args.judge_model
  self.checkpoint_signature.update(source_commit=SOURCE['commit'],adapter_sha256=SCRIPT_HASH)
 def run_one(self,case,mode):
  if BLOCKED.is_set(): raise RuntimeError('Provider blocked; job not called')
  path=self._checkpoint_path(case['test_id'],mode)
  if path.exists():
   checkpoint=json.loads(path.read_text())
   previous=checkpoint.get('signature') or {}
   if previous!=self.checkpoint_signature:
    legacy=OUT/'adapter-v1-original.py'
    legacy_hash=hashlib.sha256(legacy.read_bytes()).hexdigest() if legacy.exists() else None
    compatible={**previous,'adapter_sha256':SCRIPT_HASH}
    if previous.get('adapter_sha256')!=legacy_hash or compatible!=self.checkpoint_signature:
     raise RuntimeError('Incompatible checkpoint signature; refusing reuse')
    if checkpoint.get('test_id')!=case['test_id'] or checkpoint.get('mode')!=mode:
     raise RuntimeError('Checkpoint identity mismatch')
    if checkpoint.get('status')=='completed' and checkpoint.get('result'):
     result=dict(checkpoint['result'])
     result['_receipts']=checkpoint.get('writer_calls',[])+checkpoint.get('judge_calls',[])
     result['_resumed']=True
     return result
    checkpoint.setdefault('adapter_migrations',[]).append({'previous_signature':previous,
      'new_adapter_sha256':SCRIPT_HASH,'epoch':time.time(),
      'reason':'Align validator with official parsed.get(memory, parsed); prompts/models/rubric unchanged'})
    checkpoint['signature']=self.checkpoint_signature
    self._write_checkpoint(path,checkpoint)
  return super().run_one(case,mode)
official.Campaign=AMDCampaign
official.ChatRecorder=AMDRecorder
original_write=official.write_campaign_evidence
def evidence_write(project_dir,experiment,evidence,receipts=(),**kwargs):
 evidence['configuration']['writer_provider']='amd'
 evidence['configuration']['judge_provider']='amd'
 evidence['configuration']['independence']='Different model and fresh judge context; same AMD hosting provider'
 evidence['configuration']['seed_effective']=False
 evidence['configuration']['adapter_sha256']=SCRIPT_HASH
 evidence['configuration']['source_commit']=SOURCE['commit']
 evidence['configuration']['sdk_retries']=0
 evidence['configuration']['minimum_request_spacing_seconds']=3.5
 evidence['configuration']['response_schema_validation']=True
 evidence['configuration']['root_object_fallback']='Matches official parsed.get(memory, parsed)'
 evidence['configuration']['legacy_checkpoint_reuse']='Completed v1 results retained unchanged; pending signatures migrated with provenance'
 kwargs['input_paths']=[*kwargs.get('input_paths',[]),Path(__file__),BOOK/'source-manifest.json']
 if (OUT/'adapter-v1-original.py').exists(): kwargs['input_paths'].append(OUT/'adapter-v1-original.py')
 result=original_write(project_dir,experiment,evidence,receipts,**kwargs)
 write(OUT/'canonical-latest.json',result)
 return result
official.write_campaign_evidence=evidence_write

argv=['--all','--workers','4','--writer-endpoint',BASE,'--judge-endpoint',BASE,
 '--writer-model',WRITER,'--judge-model',JUDGE,'--checkpoint-dir',str(OUT/'checkpoints'),
 '--memory-max-tokens','6000','--answer-max-tokens','1600','--judge-max-tokens','2400','--timeout','180']
args=official.build_parser().parse_args(argv)
cases=official.load_cases(args.test_cases_dir,args)
layers=Counter(c['category'] for c in cases)
assert len(cases)==60 and layers=={'layer1':20,'layer2':20,'layer3':20}
assert len({c['test_id'] for c in cases})==60
source_files={f['path']:f for f in SOURCE['files']}
verified=[]
for path in [HERE/'run_evaluation.py',BOOK/'chapter3/experiment_utils.py',*[Path(c['_path']) for c in cases]]:
 content=path.read_bytes()
 git_hash=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
 assert git_hash==source_files[path.relative_to(BOOK).as_posix()]['sha']
 verified.append({'path':str(path),'sha256':hashlib.sha256(content).hexdigest(),'git_blob':git_hash})
sessions=sum(len(c['conversation_histories']) for c in cases)
preflight={'status':'dataset_verified','commit':SOURCE['commit'],'case_count':60,'layers':dict(layers),
 'modes':list(official.MODES),'jobs':240,'historical_sessions':sessions,
 'minimum_api_calls':4*(sessions+2*60),'sessions_by_layer':{layer:sum(len(c['conversation_histories']) for c in cases if c['category']==layer) for layer in layers},
 'files':verified,'writer_model':WRITER,'judge_model':JUDGE,'same_provider':True,
 'compatibility':'Official campaign under existing Python 3.10; not root locked Python 3.12 environment',
 'author_checkpoints_excluded':True,'checkpoint_dir':str(OUT/'checkpoints')}
write(OUT/'preflight.json',preflight)
print('PREFLIGHT '+json.dumps({k:v for k,v in preflight.items() if k!='files'},ensure_ascii=False),flush=True)

if '--preflight' in sys.argv:
 client=OpenAI(api_key=KEY,base_url=BASE,timeout=90,max_retries=0)
 models=[m.id for m in client.models.list()]
 write(OUT/'models.json',models)
 assert WRITER in models and JUDGE in models
 probe=AMDRecorder(client,'amd',BASE)
 for model in (WRITER,JUDGE):
  response=probe.create(purpose='capability probe '+model,model=model,
    messages=[{'role':'user','content':'Return JSON only: {"ok":true}'}],
    max_tokens=64,response_format={'type':'json_object'},temperature=0)
  assert official.parse_json(response.choices[0].message.content).get('ok') is True
  print('MODEL_READY '+model,flush=True)
 preflight['status']='ready'
 write(OUT/'preflight.json',preflight)
else:
 sys.argv=[str(HERE/'run_evaluation.py'),*argv]
 try:
  code=official.main()
  latest=json.loads((OUT/'canonical-latest.json').read_text())
  write(OUT/'progress.json',{'status':latest['status'],'summary':latest['summary'],
    'acceptance':latest['acceptance'],'counts':dict(COUNTS),'block_reason':BLOCK_REASON})
  raise SystemExit(code)
 except BaseException as exc:
  if not isinstance(exc,SystemExit):
   write(OUT/'fatal.json',{'type':type(exc).__name__,'error':str(exc).replace(KEY,'[REDACTED]')})
  raise
