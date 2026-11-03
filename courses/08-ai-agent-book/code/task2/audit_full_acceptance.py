"""Independently reconstruct expected requests from official cases/checkpoints."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

ROOT=Path('/home/simin/projects/task2-memory-rag')
BOOK=ROOT/'official-book'
OUT=Path(os.getenv('AUDIT_CAMPAIGN_DIR',str(ROOT/'outputs/full-60x4-amd-20261015-01')))
spec=importlib.util.spec_from_file_location('audit_official',BOOK/'chapter3/user-memory/run_evaluation.py')
official=importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
args=official.build_parser().parse_args(['--all'])
cases=official.load_cases(args.test_cases_dir,args)
by_id={c['test_id']:c for c in cases}
errors=[]
rows=[]
counts=Counter()
usage=Counter()

def require(condition,message):
 if not condition: raise ValueError(message)

def successful(calls,purpose):
 rows=[c for c in calls if c.get('purpose')==purpose and 'error' not in c and 'response' in c
       and c['response']['choices'][0].get('finish_reason')!='length']
 require(bool(rows),'Missing successful receipt: '+purpose)
 return rows[-1]

for path in sorted((OUT/'checkpoints').glob('*.json')):
 try:
  cp=json.loads(path.read_text())
  counts['checkpoints']+=1
  counts['completed' if cp.get('status')=='completed' else 'in_progress']+=1
  case=by_id[cp['test_id']]
  mode=cp['mode']
  require(mode in official.MODES,'Unknown mode')
  require(cp['signature']['writer_model']!=cp['signature']['judge_model'],'Judge model equals writer')
  require(cp['signature']['source_commit']==json.loads((OUT/'preflight.json').read_text())['commit'],'Source commit changed')
  previous=official.initial_memory(mode)
  for index,state in enumerate(cp.get('memory_states',[]),start=1):
   history=case['conversation_histories'][index-1]
   purpose=f"3-1/3-2 memory update {case['test_id']} {mode} session {index}"
   call=successful(cp['writer_calls'],purpose)
   require(call['request']['messages']==official.memory_prompt(mode,previous,history,index),'Writer input isolation mismatch')
   require(call['request']['model']==cp['signature']['writer_model'],'Writer model changed')
   parsed=official.parse_json(call['response']['choices'][0]['message']['content'])
   require(parsed.get('memory',parsed)==state['memory'],'Memory is not exact live response')
   require(state['conversation_id']==history['conversation_id'],'Session order mismatch')
   previous=state['memory']
   counts['states_audited']+=1
  if cp.get('status')!='completed': continue
  require(len(cp['memory_states'])==len(case['conversation_histories']),'Missing session state')
  answer_call=successful(cp['writer_calls'],f"3-1/3-2 answer {case['test_id']} {mode}")
  answer=answer_call['response']['choices'][0]['message']['content']
  require(answer_call['request']['messages']==official.answer_prompt(mode,previous,case['user_question']),'Answer input mismatch or raw history leakage')
  judge_call=successful(cp['judge_calls'],f"3-1/3-2 independent judge {case['test_id']} {mode}")
  require(judge_call['request']['messages']==official.judge_prompt(case,answer),'Judge prompt/source mismatch')
  require(judge_call['request']['model']==cp['signature']['judge_model'],'Judge model changed')
  judge_raw=official.parse_json(judge_call['response']['choices'][0]['message']['content'])
  summary=official.judge_summary(judge_raw)
  require(summary==cp['result']['judge'],'Score differs from official rubric')
  require(answer==cp['result']['answer'],'Answer differs from live response')
  calls=cp['writer_calls']+cp['judge_calls']
  tokens=sum(int((c.get('usage') or {}).get('total_tokens') or 0) for c in calls)
  reported_cost=sum(float((c.get('usage') or {}).get('cost') or 0) for c in calls)
  rows.append({'test_id':case['test_id'],'layer':case['category'],'mode':mode,**summary,
   'api_attempts':len(calls),'failed_attempts':sum('error' in c for c in calls),'total_tokens':tokens,'reported_cost':reported_cost,
   'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 except Exception as exc:
  errors.append({'checkpoint':path.name,'type':type(exc).__name__,'error':str(exc)})

expected={(c['test_id'],m) for c in cases for m in official.MODES}
done={(r['test_id'],r['mode']) for r in rows}
aggregate={}
for mode in official.MODES:
 aggregate[mode]={}
 for layer in ['layer1','layer2','layer3','overall']:
  selected=[r for r in rows if r['mode']==mode and (layer=='overall' or r['layer']==layer)]
  aggregate[mode][layer]={'completed':len(selected),'expected':60 if layer=='overall' else 20,
   'quality_pass_rate':sum(r['passed'] for r in selected)/len(selected) if selected else None,
   'mean_reward':statistics.mean(r['reward'] for r in selected) if selected else None,
   'hallucination_rate':sum(r['hallucination_veto'] for r in selected)/len(selected) if selected else None,
   'tokens':sum(r['total_tokens'] for r in selected),'reported_cost':sum(r['reported_cost'] for r in selected)}
audit={'status':'passed' if done==expected and not errors else 'partial','audited_epoch':time.time(),
 'coverage_completed':len(done),'coverage_expected':240,'counts':dict(counts),'errors':errors,
 'missing':[{'test_id':i,'mode':m} for i,m in sorted(expected-done)],'aggregate':aggregate,'rows':rows,
 'meaning':'passed means coverage/isolation/receipts verified, not every answer scored successful',
 'reported_cost_note':'Provider-reported cost field only; not an invoice or confirmed charge.'}
temp=OUT/'audit.tmp'
temp.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
temp.replace(OUT/'audit.json')
print(json.dumps({k:audit[k] for k in ['status','coverage_completed','coverage_expected','counts','errors']},ensure_ascii=False))
