"""Teaching equivalence of 3-3: real cached Qwen3, regex, and regex-first hybrid."""
import ast,json,re,time,hashlib,sys
import faulthandler
faulthandler.dump_traceback_later(45,repeat=True)
print('PHASE importing torch and transformers',flush=True)
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment03-equivalent-20261018'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'official-book/chapter3/log-sanitization/campaign.py'
tree=ast.parse(source.read_text())
cases=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
model_id='Qwen/Qwen3-0.6B'
torch.set_num_threads(4)
tokenizer=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
print('PHASE tokenizer ready; loading cached model',flush=True)
model=AutoModelForCausalLM.from_pretrained(model_id,local_files_only=True,dtype=torch.float16,device_map='cuda').eval()
print('PHASE model ready',flush=True)
faulthandler.cancel_dump_traceback_later()
patterns=[('cn_id_card',r'(?<!\d)\d{17}[\dXx](?!\d)'),('cn_phone',r'(?<!\d)1[3-9]\d{9}(?!\d)'),('us_ssn',r'(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)'),('credit_card',r'(?<!\d)(?:\d{4}[ -]){3}\d{4}(?!\d)'),('email',r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),('bearer_token',r'Bearer\s+([A-Za-z0-9_-]+)'),('password',r'password:\s*([^\s.]+)'),('secret',r'token=([^\s;]+)'),('password',r'://[^:/\s]+:([^@\s]+)@')]
def regex_find(text):
 result=[]
 for kind,pattern in patterns:
  for m in re.finditer(pattern,text,re.I):
   a,b=m.span(1 if m.lastindex else 0)
   result.append({'type':kind,'value':text[a:b],'start':a,'end':b,'confidence':1.0})
 return result
def redact(text,hits):
 for h in sorted(hits,key=lambda x:x['start'],reverse=True):text=text[:h['start']]+'[REDACTED]'+text[h['end']:]
 return text
def llm_find(text,label):
 prompt='''Extract sensitive information from the log. Distinguish disclosed private values from product IDs, metrics, order IDs, instructions and explicit decoys. Types: cn_id_card, cn_phone, us_ssn, credit_card, password, secret, address, medical_record, medical_diagnosis, treatment, passport, bank_account, routing_number, bearer_token, email. Return JSON only: {"findings":[{"type":"password","value":"exact substring","confidence":0.9}]}. No findings: {"findings":[]}. Values must be copied exactly; do not include descriptive prefixes. Ignore [REDACTED].'''
 messages=[{'role':'system','content':prompt},{'role':'user','content':text}]
 encoded=tokenizer.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to(model.device)
 begin=time.monotonic()
 with torch.inference_mode():output=model.generate(**encoded,max_new_tokens=512,do_sample=False)
 raw=tokenizer.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
 parse_error=None;hits=[]
 try:
  cleaned=raw.strip()
  if cleaned.startswith('```'):cleaned=re.sub(r'^```(?:json)?\s*|\s*```$','',cleaned)
  obj=json.loads(cleaned)
  for h in obj['findings']:
   value=h.get('value','')
   if not isinstance(value,str) or not value:continue
   start=text.find(value)
   if start>=0:hits.append({**h,'start':start,'end':start+len(value)})
 except Exception as e:parse_error=str(e)
 receipt={'label':label,'messages':messages,'raw_output':raw,'parsed_findings':hits,'parse_error':parse_error,'input_tokens':encoded['input_ids'].shape[1],'output_tokens':output.shape[1]-encoded['input_ids'].shape[1],'latency_seconds':time.monotonic()-begin,'finish':'length' if output.shape[1]-encoded['input_ids'].shape[1]>=512 else 'eos'}
 (OUT/(label+'.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 return hits,receipt
def evaluate(case,hits):
 text=case['text'];gold={(text.index(v),text.index(v)+len(v)) for _,v in case['gold']}
 predicted={(h['start'],h['end']) for h in hits}
 tp=len(gold & predicted);fp=len(predicted-gold);fn=len(gold-predicted)
 protected=set(i for a,b in gold for i in range(a,b));removed=set(i for a,b in predicted for i in range(a,b))
 benign=set(range(len(text)))-protected
 return {'tp':tp,'fp':fp,'fn':fn,'residual_values':[text[a:b] for a,b in sorted(gold-predicted)],'false_positive_values':[text[a:b] for a,b in sorted(predicted-gold)],'benign_chars':len(benign),'benign_chars_removed':len(benign & removed),'redacted_text':redact(text,hits),'findings':hits}
results=[]
for index,case in enumerate(cases):
 regex=regex_find(case['text'])
 llm,receipt=llm_find(case['text'],case['id']+'-llm')
 remaining,hybrid_receipt=llm_find(redact(case['text'],regex),case['id']+'-hybrid')
 hybrid=list(regex)
 for h in remaining:
  a=case['text'].find(h['value'])
  if a>=0:hybrid.append({**h,'start':a,'end':a+len(h['value'])})
 row={'case_id':case['id'],'original':case['text'],'gold':case['gold'],'arms':{name:evaluate(case,hits) for name,hits in [('regex',regex),('llm',llm),('hybrid',hybrid)]},'llm_parse_error':receipt['parse_error'],'hybrid_parse_error':hybrid_receipt['parse_error']}
 results.append(row)
 (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
 print(f"CASE {index+1}/{len(cases)} {case['id']} "+json.dumps({k:{x:v[x] for x in ('tp','fp','fn')} for k,v in row['arms'].items()}),flush=True)
summary={}
for arm in ('regex','llm','hybrid'):
 totals={key:sum(r['arms'][arm][key] for r in results) for key in ('tp','fp','fn','benign_chars','benign_chars_removed')}
 totals.update(precision=totals['tp']/max(1,totals['tp']+totals['fp']),recall=totals['tp']/max(1,totals['tp']+totals['fn']),benign_retention=1-totals['benign_chars_removed']/max(1,totals['benign_chars']))
 summary[arm]=totals
report={'status':'completed','scope':'mechanism-equivalent teaching experiment, not official Ollama validation','dataset':'12 official fictional log samples','model':model_id,'backend':'local Hugging Face Transformers CUDA; Ollama replaced as authorized','regex':'documented compact patterns; not the upstream regex engine','normalization':'exact substring match; offsets computed deterministically; invalid JSON recorded as failure, never silently repaired','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'summary':summary,'parse_errors':sum(bool(r[x]) for r in results for x in ('llm_parse_error','hybrid_parse_error'))}
(OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('COMPLETED '+json.dumps(report,ensure_ascii=False),flush=True)
