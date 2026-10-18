"""Dual-layer memory: retained live cards trigger verification in raw history."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ast,json,re,time,math,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import yaml,torch
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment11-equivalent-20261018';OUT.mkdir(parents=True,exist_ok=True)
casepath=ROOT/'official-book/chapter3/user-memory-evaluation/test_cases/layer3/01_travel_coordination.yaml'
checkpoint=ROOT/'outputs/full-60x4-recovery-20261017-01/checkpoints/layer3_01_travel_coordination--advanced_json_cards.json'
case=yaml.safe_load(casepath.read_text());cp=json.loads(checkpoint.read_text())
tree=ast.parse((ROOT/'experiment09_equivalent.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('make_chunks','words')]
helpers={'re':re};exec(compile(ast.Module(body=functions,type_ignores=[]),'reused_chunker','exec'),helpers)
chunks=helpers['make_chunks'](case);words=helpers['words']
for c in chunks:c['session_text']='\n'.join(m['role'].upper()+': '+m['content'] for h in case['conversation_histories'] if h['conversation_id']==c['conversation_id'] for m in h['messages'])
(OUT/'chunks-original.json').write_text(json.dumps([{k:v for k,v in c.items() if k!='session_text'} for c in chunks],ensure_ascii=False,indent=2),encoding='utf-8')
# All cards were generated in the earlier live 3-1 run, not authored for this question.
retained={};origins={}
for state in cp['memory_states']:
 for card in state['memory']:
  key=card['card_key'];previous=retained.get(key)
  if previous is None or str(card.get('timestamp',''))>=str(previous.get('timestamp','')):
   retained[key]=card;origins[key]={'session_index':state['session_index'],'conversation_id':state['conversation_id']}
all_cards=list(retained.values());final_cards=cp['memory_states'][-1]['memory']
def compact(cards):
 out=[]
 for i,c in enumerate(cards):
  facts={k:v for k,v in c['facts'].items() if re.search(r'date|expir|confirm|renewal_decision|paid|payment|coverage|insurance|priority_pass',k)}
  out.append({'id':'K'+str(i+1),'category':c['category'],'card_key':c['card_key'],'person':c['person'],'timestamp':c['timestamp'],'status':c['status'],'backstory':c['backstory'],'facts':facts})
 return out
cards_retained=compact(all_cards);cards_final=compact(final_cards)
(OUT/'cards-provenance.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'source_case_sha256':hashlib.sha256(casepath.read_bytes()).hexdigest(),'writer_call_count':len(cp.get('writer_calls',[])),'state_card_keys':[[c['card_key'] for c in s['memory']] for s in cp['memory_states']],'retention_policy':'Controlled accumulation of earlier live states by card_key, keeping greatest card timestamp; not the original final snapshot. No deletion/tombstone semantics implemented.','retained_card_origins':origins,'retained_cards_full':all_cards,'compact_retained':cards_retained,'compact_original_final':cards_final},ensure_ascii=False,indent=2),encoding='utf-8')
torch.set_num_threads(4);model_id='Qwen/Qwen3-0.6B'
tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
llm=AutoModelForCausalLM.from_pretrained(model_id,local_files_only=True,dtype=torch.float16,device_map='cuda').eval();receipts=[]
def call(label,system,user,max_tokens=384):
 messages=[{'role':'system','content':system},{'role':'user','content':user}]
 encoded=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda');begin=time.perf_counter()
 with torch.inference_mode():output=llm.generate(**encoded,max_new_tokens=max_tokens,do_sample=False)
 raw=tok.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
 receipts.append({'label':label,'messages':messages,'raw_output':raw,'input_tokens':encoded['input_ids'].shape[1],'output_tokens':output.shape[1]-encoded['input_ids'].shape[1],'latency_ms':(time.perf_counter()-begin)*1000,'finish':'length' if output.shape[1]-encoded['input_ids'].shape[1]>=max_tokens else 'eos'})
 (OUT/'receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8');print('CALL '+label,flush=True);return raw
for c in chunks:
 c['prefix']=call('prefix-'+c['id'],'Write a short source-context prefix for the target history excerpt. Identify the person, service, date and topic from the full session. Do not invent facts, transfer later decisions into a tentative excerpt, or answer a question. One sentence, no more than 35 words.','Session ID: '+c['conversation_id']+'\nTimestamp: '+c['timestamp']+'\nFull session:\n'+c['session_text']+'\nTarget excerpt:\n'+c['text'],96)
(OUT/'chunks-contextual.json').write_text(json.dumps([{k:v for k,v in c.items() if k!='session_text'} for c in chunks],ensure_ascii=False,indent=2),encoding='utf-8')
etok=AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5',local_files_only=True)
encoder=AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5',local_files_only=True).to('cuda').eval()
def encode(texts,query=False):
 vectors=[]
 for start in range(0,len(texts),16):
  values=texts[start:start+16]
  if query:values=['为这个句子生成表示以用于检索相关文章：'+t for t in values]
  b=etok(values,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
  with torch.inference_mode():v=encoder(**b).last_hidden_state[:,0]
  vectors.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy())
 return np.concatenate(vectors).astype('float32')
indexes={}
for kind in ('plain','contextual'):
 texts=[c['text'] if kind=='plain' else c['prefix']+'\n'+c['text'] for c in chunks]
 vectors=encode(texts);np.save(OUT/(kind+'-vectors.npy'),vectors);postings=defaultdict(dict);lengths=[]
 for i,text in enumerate(texts):
  tokens=words(text);lengths.append(len(tokens))
  for term,tf in Counter(tokens).items():postings[term][i]=tf
 indexes[kind]={'vectors':vectors,'postings':postings,'lengths':lengths,'avgdl':sum(lengths)/len(lengths)}
def search(kind,query,k=3):
 index=indexes[kind];N=len(chunks);ss=np.zeros(N)
 for term in words(query):
  hits=index['postings'].get(term,{});idf=math.log(1+(N-len(hits)+.5)/(len(hits)+.5))
  for i,tf in hits.items():ss[i]+=idf*tf*2.5/(tf+1.5*(.25+.75*index['lengths'][i]/index['avgdl']))
 ds=index['vectors']@encode([query],True)[0]
 sr=sorted([i for i,v in enumerate(ss) if v>0],key=lambda i:(-ss[i],i))[:10];dr=np.argsort(-ds)[:10].tolist()
 fusion={i:(1/(60+sr.index(i)+1) if i in sr else 0)+(1/(60+dr.index(i)+1) if i in dr else 0) for i in set(sr+dr)}
 ids=sorted(fusion,key=lambda i:(-fusion[i],i))[:k]
 return {'index':kind,'query':query,'hits':[{'id':chunks[i]['id'],'text':chunks[i]['text'],'prefix':chunks[i]['prefix'] if kind=='contextual' else None,'conversation_id':chunks[i]['conversation_id'],'timestamp':chunks[i]['timestamp'],'score':fusion[i]} for i in ids]}
def source_context(hits):return '\n\n'.join(f"[{h['id']}] timestamp={h['timestamp']} session={h['conversation_id']}\n{h['text']}" for h in hits)
def seed_queries(cards):
 out=[]
 for c in cards:
  # Fixed schema-to-query policy, not claimed as autonomous model planning.
  fields=[k.replace('_',' ') for k in c['facts'] if re.search(r'expir|return_date|confirm|priority_pass|coverage|insurance',k)][:4]
  out.append({'card_id':c['id'],'card_key':c['card_key'],'query':c['category'].replace('_',' ')+' '+c['person']+' '+' '.join(fields)})
 return out
question=case['user_question'];plain=search('plain',question);contextual=search('contextual',question)
arms={'plain_rag':{'cards':[],'searches':[plain],'evidence':plain['hits']},'contextual_rag':{'cards':[],'searches':[contextual],'evidence':contextual['hits']}}
for name,cards in [('dual_original_final_cards',cards_final),('dual_retained_cards',cards_retained)]:
 seeds=seed_queries(cards);searches=[contextual]+[search('contextual',s['query'],2) for s in seeds];evidence={h['id']:h for s in searches for h in s['hits']}
 arms[name]={'cards':cards,'seed_queries':seeds,'searches':searches,'evidence':list(evidence.values())}
answer_prompt='''Answer the travel-preparation question using only supplied historical cards and conversation excerpts. Cards are summaries to verify, not authority. Cite [Hnumber] for original evidence; if a card fact has no matching retrieved source, say it needs confirmation. Connect trip dates with identity-document dates and payment-dependent coverage when supported. Distinguish discussion, decision, and completed action. Do not claim a universal six-month passport rule or certain denied boarding; destination and carrier requirements need current verification. No outside policy facts. This is a fictional historical benchmark, not real travel advice. Concise English answer.'''
for name,arm in arms.items():
 cardtext=json.dumps(arm['cards'],ensure_ascii=False,indent=2) if arm['cards'] else 'None'
 arm['answer']=call(name+'-answer',answer_prompt,'Question: '+question+'\nOverview cards (verify):\n'+cardtext+'\nOriginal evidence:\n'+source_context(arm['evidence']),512)
 arm['sessions_retrieved']=sorted({e['conversation_id'] for e in arm['evidence']});arm['source_ids']=[e['id'] for e in arm['evidence']]
 arm['citation_ids']=re.findall(r'\[(H\d+)\]',arm['answer'])
 arm['unknown_citation_ids']=sorted(set(arm['citation_ids'])-set(arm['source_ids']))
 arm['presence_checks_only']={label:bool(re.search(pattern,arm['answer'],re.I)) for label,pattern in {'passport':'passport','february_18':'february\s+18|feb\.?\s+18|2025-02-18','priority_pass':'priority pass','payment_coverage':'amex|american express|paid|insurance'}.items()}
 print('ANSWER '+name+' '+json.dumps({'sources':arm['source_ids'],'checks':arm['presence_checks_only']}),flush=True)
report={'status':'completed','scope':'mechanism-equivalent one official layer3 case, not full 60-case dual-layer campaign','source_commit':'dbc046eb896ac4e39aa19c7774c8bf49583b89a6','case_id':case['test_id'],'question':question,'sessions':len(case['conversation_histories']),'chunks':len(chunks),'cards_provenance':'Previously live-generated Advanced JSON Cards from 3-1 checkpoints; original final snapshot compared with controlled retention of latest card_key versions from earlier states. No hand-authored facts added.','models':{'prefix_and_answer':model_id,'embedding':'BAAI/bge-small-zh-v1.5','card_writer':'Prior 3-1 DeepSeek-V4-Flash, requests retained in source checkpoint'},'controls':'Same question, answer model/prompt/token budget. Contextual prefixes from full source session plus target. Dual arms add cards and fixed schema-triggered searches, hence increase evidence/context budget; not an isolated architecture effect.','known_limits':'No automatic deletion/tombstone handling; controlled retained cards are an illustrative maintenance policy. Raw-history source is required to verify card facts. No external judge; presence checks do not prove correctness. Rubric and source history contain inconsistent passport-policy/processing statements, so no real entry-rule conclusion is adopted.','reference_isolation':'Evaluation criteria and expected_behavior never provided to prefix, search or answer models.','arms':arms}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETED '+json.dumps({'chunks':len(chunks),'original_final_cards':len(cards_final),'retained_cards':len(cards_retained)},ensure_ascii=False),flush=True)
