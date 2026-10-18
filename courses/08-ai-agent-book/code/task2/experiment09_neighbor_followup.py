"""Expand each selected history window by +/-1 within its own conversation."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ast,json,time,hashlib,re
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag');BASE=ROOT/'outputs/experiment09-equivalent-20261018'
OUT=ROOT/'outputs/experiment09-neighbor-followup-20261018';OUT.mkdir(parents=True,exist_ok=True)
baseline=json.loads((BASE/'results.json').read_text())
tree=ast.parse((ROOT/'experiment09_equivalent.py').read_text())
prompt=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='answer_system' for t in n.targets))
torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True,dtype=torch.float16,device_map='cuda').eval()
results=[];receipts=[]
for case in baseline['results']:
 chunks=json.loads((BASE/case['case_id']/'chunks.json').read_text());selected=case['arms']['session_coverage']['evidence'];wanted={c['id'] for c in selected};expansion=[]
 for hit in selected:
  siblings=[c for c in chunks if c['conversation_id']==hit['conversation_id']];pos=next(i for i,c in enumerate(siblings) if c['id']==hit['id'])
  neighbors=siblings[max(0,pos-1):pos+2];wanted.update(c['id'] for c in neighbors);expansion.append({'hit':hit['id'],'same_session_neighbors':[c['id'] for c in neighbors]})
 evidence=sorted([c for c in chunks if c['id'] in wanted],key=lambda c:(-int(c['timestamp'][:10].replace('-','')),c['conversation_id'],c['round_start']))
 context='\n\n'.join(f"[{c['id']}] conversation={c['conversation_id']} timestamp={c['timestamp']} rounds={c['round_start']}-{c['round_end']}\n{c['text']}" for c in evidence)
 messages=[{'role':'system','content':prompt},{'role':'user','content':'Question: '+case['question']+'\nHistorical evidence:\n'+context}]
 inputs=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda');begin=time.perf_counter()
 with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=512,do_sample=False)
 answer=tok.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 receipts.append({'case_id':case['case_id'],'messages':messages,'raw_output':answer,'input_tokens':inputs['input_ids'].shape[1],'output_tokens':output.shape[1]-inputs['input_ids'].shape[1],'latency_ms':(time.perf_counter()-begin)*1000})
 results.append({'case_id':case['case_id'],'question':case['question'],'seeds':[c['id'] for c in selected],'expansion':expansion,'evidence':evidence,'answer':answer,'latest_amount_explicitly_in_context':bool(re.search(r'95,000',context)) if 'financial' in case['case_id'] else None,'both_car_names_in_context':all(x in context for x in ('Honda','Tesla')) if 'vehicles' in case['case_id'] else None})
 print(case['case_id']+' '+answer,flush=True)
(OUT/'receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'results.json').write_text(json.dumps({'status':'completed','baseline_sha256':hashlib.sha256((BASE/'results.json').read_bytes()).hexdigest(),'method':'Author-controlled +/-1 same-conversation expansion of session-coverage hits; chronological presentation within each session; same answer model/prompt/budget. Larger context and presentation both change, not an autonomous agent action.','results':results},ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETED',flush=True)
