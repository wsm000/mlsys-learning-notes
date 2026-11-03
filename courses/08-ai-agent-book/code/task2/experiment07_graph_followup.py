"""Reuse original model relations; fix container parsing, then test entity naming."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,re,time,hashlib
from pathlib import Path
from collections import defaultdict,deque
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag')
BASE=ROOT/'outputs/experiment07-equivalent-20261018'
OUT=ROOT/'outputs/experiment07-graph-followup-20261018';OUT.mkdir(parents=True,exist_ok=True)
chunks=json.loads((BASE/'input-chunks.json').read_text());byid={c['id']:c for c in chunks}
original=json.loads((BASE/'receipts.json').read_text());edges=[];errors=[]
def parse(text):
 value=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))
 return value if isinstance(value,list) else value['edges']
def validate(items,chunk,allowed=None):
 out=[]
 for e in items:
  if not isinstance(e,dict):continue
  s,o,r=e.get('subject'),e.get('object'),e.get('relation')
  if not all(isinstance(x,str) and x for x in (s,o,r)) or s not in chunk['text'] or o not in chunk['text']:continue
  if allowed and (s not in allowed or o not in allowed):continue
  out.append({'subject':s,'relation':r,'object':o,'source_id':chunk['id'],'source_text':chunk['text']})
 return out
for receipt in original:
 if not receipt['label'].startswith('graph-'):continue
 cid=receipt['label'][6:]
 try:edges.extend(validate(parse(receipt['raw_output']),byid[cid]))
 except Exception as e:errors.append({'chunk':cid,'error':str(e)})
def walk(graph,start='青禾蜂场'):
 adj=defaultdict(list)
 for i,e in enumerate(graph):adj[e['subject']].append(i)
 queue=deque([(start,0,[])]);seen={start};used=set();paths=[]
 while queue:
  entity,depth,path=queue.popleft()
  if depth>=4:continue
  for i in adj.get(entity,[]):
   e=graph[i];used.add(i);newpath=path+[i];paths.append({'end':e['object'],'edge_indices':newpath})
   if e['object'] not in seen:seen.add(e['object']);queue.append((e['object'],depth+1,newpath))
 return {'start':start,'max_hops':4,'visited_entities':sorted(seen),'source_ids':sorted({graph[i]['source_id'] for i in used}),'edge_indices':sorted(used),'paths':paths,'reaches_manual_M':'手册M' in seen}
baseline_walk=walk(edges)
torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True,dtype=torch.float16,device_map='cuda').eval()
receipts=[]
def call(label,system,user,max_tokens=384):
 messages=[{'role':'system','content':system},{'role':'user','content':user}]
 batch=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda');begin=time.perf_counter()
 with torch.inference_mode():output=model.generate(**batch,max_new_tokens=max_tokens,do_sample=False)
 text=tok.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
 receipts.append({'label':label,'messages':messages,'raw_output':text,'latency_ms':(time.perf_counter()-begin)*1000,'finish':'length' if output.shape[1]-batch['input_ids'].shape[1]>=max_tokens else 'eos'})
 (OUT/'receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
 print('CALL '+label,flush=True);return text
allowed=['蜂群A','蜂螨','手册M']
raw=call('C2-controlled-entity-names','只提取资料明确说明的有向关系，返回JSON数组，每项含subject、relation、object。实体名称必须严格选自用户提供的名单，不要拆开手册编号，不要将整个句子当实体。不要增加资料没有的关系。','实体名单：'+json.dumps(allowed,ensure_ascii=False)+'\n资料：'+byid['C2']['text'])
try:replacement=validate(parse(raw),byid['C2'],set(allowed));repair_error=None
except Exception as e:replacement=[];repair_error=str(e)
controlled_edges=[e for e in edges if e['source_id']!='C2']+replacement
controlled_walk=walk(controlled_edges)
question='青禾蜂场管理的蜂群遇到了什么问题，应参考哪本手册，手册要求先做什么？'
answers=[]
for name,graph,traversal in [('unconstrained_entity_names',edges,baseline_walk),('controlled_entity_names',controlled_edges,controlled_walk)]:
 context='\n'.join(f"[{e['source_id']}] {e['subject']} --{e['relation']}--> {e['object']}" for i,e in enumerate(graph) if i in traversal['edge_indices'])
 answer=call(name+'-answer','只依据证据回答，引用[C编号]。缺少哪部分就明确说明，不得根据常识猜测手册。','问题：'+question+'\n证据：\n'+context,256)
 answers.append({'method':name,'question':question,'context':context,'answer':answer})
report={'status':'completed','scope':'targeted mechanism demonstration; response-container compatibility then live entity-name constraint for C2 only','original_receipts_sha256':hashlib.sha256((BASE/'receipts.json').read_bytes()).hexdigest(),'baseline':{'edges':edges,'traversal':baseline_walk,'parse_errors':errors},'controlled':{'edges':controlled_edges,'traversal':controlled_walk,'replacement_C2_edges':replacement,'parse_error':repair_error,'allowed_names':allowed},'answers':answers,'limitations':'Allowed entity names supplied by experiment author; no automatic entity linking or full GraphRAG community pipeline. Endpoint grounding does not validate relation truth. Failed source relations retained.'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETED '+json.dumps({'before':baseline_walk,'after':controlled_walk},ensure_ascii=False),flush=True)
