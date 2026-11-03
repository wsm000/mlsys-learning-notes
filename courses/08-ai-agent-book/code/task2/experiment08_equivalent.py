"""Fixed one-shot RAG vs live model-controlled bounded retrieval loop."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,re,time,math
from pathlib import Path
from collections import Counter,defaultdict
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment08-equivalent-20261018';OUT.mkdir(parents=True,exist_ok=True)
chunks=json.loads((ROOT/'outputs/experiment07-equivalent-20261018/input-chunks.json').read_text())
vocab=['青禾蜂场','白石蜂场','河谷镇','山岭镇','蜂群A','蜂群B','蜂螨','手册M','手册N','监测','数量','防治','饲料','保温','检查','要求','冬季','参考','管理','位于','先','补充','缺少','处理','卫生','饮水','防风']
def tokenize(text):return [term for term in vocab for _ in range(text.count(term))]
index=defaultdict(dict);lengths={}
for c in chunks:
 c['tokens']=tokenize(c['text']);lengths[c['id']]=len(c['tokens'])
 for t,tf in Counter(c['tokens']).items():index[t][c['id']]=tf
avgdl=sum(lengths.values())/len(chunks)
def search(query):
 scores=defaultdict(float)
 for t in tokenize(query):
  hits=index[t];idf=math.log(1+(len(chunks)-len(hits)+.5)/(len(hits)+.5))
  for cid,tf in hits.items():scores[cid]+=idf*tf*2.5/(tf+1.5*(.25+.75*lengths[cid]/avgdl))
 ranked=sorted(scores,key=lambda c:(-scores[c],c))[:2]
 return {'query':query,'tokens':tokenize(query),'hits':[{'id':cid,'score':scores[cid],'text':next(c['text'] for c in chunks if c['id']==cid)} for cid in ranked]}
def context(rows):return '\n'.join('['+c['id']+'] '+c['text'] for c in rows)
torch.set_num_threads(4);model_id='Qwen/Qwen3-0.6B'
tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(model_id,local_files_only=True,dtype=torch.float16,device_map='cuda').eval()
receipts=[]
def call(label,system,user,max_tokens):
 messages=[{'role':'system','content':system},{'role':'user','content':user}]
 encoded=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda');begin=time.perf_counter()
 with torch.inference_mode():output=model.generate(**encoded,max_new_tokens=max_tokens,do_sample=False)
 raw=tok.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
 receipt={'label':label,'messages':messages,'raw_output':raw,'latency_ms':(time.perf_counter()-begin)*1000,'input_tokens':encoded['input_ids'].shape[1],'output_tokens':output.shape[1]-encoded['input_ids'].shape[1],'finish':'length' if output.shape[1]-encoded['input_ids'].shape[1]>=max_tokens else 'eos'}
 receipts.append(receipt);(OUT/'receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
 print('CALL '+label+' '+raw.replace('\n',' ')[:200],flush=True);return raw
answer_prompt='只根据提供的证据回答问题，不得补充养蜂常识。每项结论引用对应的[C编号]。缺少哪部分就明确说证据未说明。简短回答。'
planner_prompt='''你是检索控制器。根据问题和已取得的证据，决定是否需要继续调用search。只输出一个JSON对象，不要解释、不要写答案。
字段action只能是search或finish。需要补证据时action=search，同时字段query写一个简短检索词，优先用证据中出现的一两个实体名称。
若全部子问题已有直接证据，action=finish。若已重复查找且仍缺信息，也可finish，最终回答应说明缺失。不要重复以前的query。不要假设某种蜂场问题必然采用某本手册，必须找到连接关系。'''
questions=[{'id':'simple','question':'手册M要求先做什么？','gold_sources':['C3'],'kind':'single-hop'},
 {'id':'multihop','question':'青禾蜂场管理的蜂群遇到了什么问题，应参考哪本手册，第一步要做什么？','gold_sources':['C1','C2','C3'],'kind':'multi-hop'},
 {'id':'missing','question':'手册M规定蜂螨数量达到多少必须处理？','gold_sources':[],'kind':'unanswerable numeric threshold'}]
results=[]
for q in questions:
 initial=search(q['question']);baseline_rows=initial['hits'];begin=time.perf_counter()
 baseline_answer=call(q['id']+'-one-shot-answer',answer_prompt,'问题：'+q['question']+'\n证据：\n'+context(baseline_rows),256)
 baseline_ms=(time.perf_counter()-begin)*1000
 seen={c['id']:c for c in baseline_rows};searches=[initial];actions=[];stop='additional-search budget exhausted';begin=time.perf_counter()
 for turn in range(4):
  prompt='问题：'+q['question']+'\n已取得的证据：\n'+context(list(seen.values()))+'\n已执行查询：'+json.dumps([s['query'] for s in searches],ensure_ascii=False)+'\n还可追加检索次数：'+str(3-(len(searches)-1))
  raw=call(q['id']+'-controller-'+str(turn+1),planner_prompt,prompt,160)
  try:
   decision=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip()))
   if not isinstance(decision,dict) or decision.get('action') not in ('search','finish'):raise ValueError('invalid action schema')
  except Exception as e:actions.append({'raw':raw,'error':str(e)});stop='controller format error';break
  actions.append({'raw':raw,'decision':decision,'evidence_before':[c for c in seen]})
  if decision['action']=='finish':stop='model decided finish';break
  query=decision.get('query')
  if not isinstance(query,str) or not query.strip():stop='empty query';break
  if query in [s['query'] for s in searches]:stop='repeated query rejected';break
  if len(searches)>=4:stop='additional-search budget exhausted';break
  found=search(query);searches.append(found)
  for c in found['hits']:seen[c['id']]=c
 agent_answer=call(q['id']+'-agent-answer',answer_prompt,'问题：'+q['question']+'\n证据：\n'+context(list(seen.values())),256)
 agent_ms=(time.perf_counter()-begin)*1000
 def coverage(rows):return len({c['id'] for c in rows}&set(q['gold_sources']))/len(q['gold_sources']) if q['gold_sources'] else None
 results.append({**q,'one_shot':{'searches':[initial],'evidence':baseline_rows,'answer':baseline_answer,'evidence_coverage':coverage(baseline_rows),'latency_ms':baseline_ms},'agentic':{'searches':searches,'actions':actions,'stop_reason':stop,'evidence':list(seen.values()),'answer':agent_answer,'evidence_coverage':coverage(list(seen.values())),'latency_ms':agent_ms},'citation_syntax':{name:re.findall(r'\[(C\d+)\]',answer) for name,answer in [('one_shot',baseline_answer),('agentic',agent_answer)]}})
 (OUT/'results.partial.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 print('CASE '+q['id']+' searches='+str(len(searches))+' stop='+stop,flush=True)
report={'status':'completed','scope':'mechanism-equivalent authored bee corpus, not official legal QA benchmark','models':{'controller':model_id,'answerer':model_id},'controlled_factors':'Same corpus, deterministic BM25, top-2 per search, identical first search, same answer model/prompt/budget. Agent accumulates up to 4 searches; extra evidence and compute are part of the treatment.','agent_mechanism':'Live model emits bounded search/finish actions after seeing evidence; no scripted next query or reference answer is supplied. Only short action JSON is requested, no private reasoning trace.','limitations':'Explicit vocabulary tokenizer; no external judge; evidence coverage does not prove answer correctness; only 3 teaching questions.','chunks':chunks,'lexical_vocabulary':vocab,'results':results}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETED '+json.dumps([{'id':r['id'],'one_shot_sources':[x['id'] for x in r['one_shot']['evidence']],'agent_sources':[x['id'] for x in r['agentic']['evidence']],'agent_searches':len(r['agentic']['searches']),'stop':r['agentic']['stop_reason']} for r in results],ensure_ascii=False),flush=True)
