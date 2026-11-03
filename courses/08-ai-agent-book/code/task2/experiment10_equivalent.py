"""Live contextual prefixes vs plain chunks under identical retrieval queries."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,re,time,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment10-equivalent-20261018';OUT.mkdir(parents=True,exist_ok=True)
documents=[
 {'id':'A','title':'青禾蜂场｜2025年春季生产报告','chunks':['本季蜂蜜产量增长30%，主要来自油菜蜜。','本季增加了20箱蜂，新增蜂群已完成检查。','产量统计仅包含本蜂场，不包含合作蜂场。']},
 {'id':'B','title':'白石蜂场｜2025年春季生产报告','chunks':['本季蜂蜜产量下降10%，主要因为连续降雨。','本季减少了5箱蜂，弱群已完成合并。','本季采蜜时间比计划推迟一周。']},
 {'id':'C','title':'青禾蜂场｜2025年秋季生产报告','chunks':['本季蜂蜜产量增长8%，主要来自荞麦蜜。','本季增加了3箱蜂，蜂箱已编号入册。','入冬前已完成饲料储备检查。']},
 {'id':'D','title':'青禾蜂场｜2024年春季生产报告','chunks':['本季蜂蜜产量增长30%，主要来自油菜蜜。','本季增加了10箱蜂，新增蜂群已完成检查。','产量统计仅包含本蜂场，不包含合作蜂场。']},
]
chunks=[{'id':doc['id']+str(i+1),'document_id':doc['id'],'title':doc['title'],'text':text} for doc in documents for i,text in enumerate(doc['chunks'])]
queries=[
 {'id':'Q1','question':'青禾蜂场2025年春季蜂蜜产量变化多少，主要来自哪种蜂蜜？','gold':['A1']},
 {'id':'Q2','question':'白石蜂场2025年春季蜂蜜产量为什么下降？','gold':['B1']},
 {'id':'Q3','question':'青禾蜂场2025年秋季蜂蜜产量变化多少，主要来自哪种蜂蜜？','gold':['C1']},
 {'id':'Q4','question':'青禾蜂场2024年春季蜂蜜产量变化多少，主要来自哪种蜂蜜？','gold':['D1']},
 {'id':'Q5','question':'青禾蜂场2025年春季增加了多少箱蜂？','gold':['A2']},
 {'id':'Q6','question':'青禾蜂场2024年春季增加了多少箱蜂？','gold':['D2']},
]
torch.set_num_threads(4)
ltok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
llm=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True,dtype=torch.float16,device_map='cuda').eval()
receipts=[];total_begin=time.perf_counter()
for chunk in chunks:
 doc=next(d for d in documents if d['id']==chunk['document_id'])
 source=doc['title']+'\n'+'\n'.join(doc['chunks'])
 messages=[{'role':'system','content':'根据完整文档，为目标文本块写一条简短背景前缀，说明蜂场名称、年份、季节和该块主题。只写前缀，不超过60字。不得编造事实，不要写其他块的具体数值，不要回答任何问题。'},
 {'role':'user','content':'完整文档：\n'+source+'\n\n目标文本块：\n'+chunk['text']}]
 encoded=ltok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda');begin=time.perf_counter()
 with torch.inference_mode():output=llm.generate(**encoded,max_new_tokens=128,do_sample=False)
 prefix=ltok.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True).strip()
 chunk['generated_prefix']=prefix;chunk['contextual_text']=prefix+'\n'+chunk['text']
 chunk['introduced_number_flags']=sorted(set(re.findall(r'\d+(?:\.\d+)?',prefix))-set(re.findall(r'\d+(?:\.\d+)?',source)))
 receipts.append({'chunk_id':chunk['id'],'messages':messages,'raw_output':prefix,'input_tokens':encoded['input_ids'].shape[1],'output_tokens':output.shape[1]-encoded['input_ids'].shape[1],'latency_ms':(time.perf_counter()-begin)*1000,'finish':'length' if output.shape[1]-encoded['input_ids'].shape[1]>=128 else 'eos'})
 (OUT/'prefix-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
 print('PREFIX '+chunk['id']+' '+prefix.replace('\n',' '),flush=True)
prefix_ms=(time.perf_counter()-total_begin)*1000
del llm;torch.cuda.empty_cache()
tok=AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5',local_files_only=True)
encoder=AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5',local_files_only=True).to('cuda').eval()
def encode(texts,query=False):
 if query:texts=['为这个句子生成表示以用于检索相关文章：'+t for t in texts]
 batch=tok(texts,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
 with torch.inference_mode():v=encoder(**batch).last_hidden_state[:,0]
 return torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy()
def words(text):
 out=[]
 for seg in re.findall(r'[\u4e00-\u9fff]+|[A-Za-z]+|\d+',text.lower()):
  if re.fullmatch(r'[\u4e00-\u9fff]+',seg):out.extend([seg[i:i+2] for i in range(len(seg)-1)] if len(seg)>1 else [seg])
  else:out.append(seg)
 return out
def sparse_scores(texts,query):
 postings=defaultdict(dict);lengths=[]
 for i,text in enumerate(texts):
  tokens=words(text);lengths.append(len(tokens))
  for word,tf in Counter(tokens).items():postings[word][i]=tf
 N=len(texts);avgdl=sum(lengths)/N;scores=np.zeros(N)
 for term in words(query):
  hits=postings.get(term,{})
  idf=math.log(1+(N-len(hits)+.5)/(len(hits)+.5))
  for i,tf in hits.items():scores[i]+=idf*tf*2.5/(tf+1.5*(.25+.75*lengths[i]/avgdl))
 return scores
qv=encode([q['question'] for q in queries],True);np.save(OUT/'query-vectors.npy',qv)
rows=[]
for kind,field in [('plain','text'),('contextual','contextual_text')]:
 texts=[c[field] for c in chunks];vectors=encode(texts);np.save(OUT/(kind+'-vectors.npy'),vectors)
 for qi,q in enumerate(queries):
  ss=sparse_scores(texts,q['question']);ds=vectors@qv[qi]
  sr=sorted([i for i,v in enumerate(ss) if v>0],key=lambda i:(-ss[i],chunks[i]['id']))
  dr=sorted(range(len(chunks)),key=lambda i:(-ds[i],chunks[i]['id']))
  rrf={i:(1/(60+sr[:5].index(i)+1) if i in sr[:5] else 0)+(1/(60+dr[:5].index(i)+1) if i in dr[:5] else 0) for i in set(sr[:5]+dr[:5])}
  hr=sorted(rrf,key=lambda i:(-rrf[i],chunks[i]['id']))
  rankings={'bm25':[chunks[i]['id'] for i in sr],'dense':[chunks[i]['id'] for i in dr],'hybrid':[chunks[i]['id'] for i in hr]}
  metrics={}
  for method,ids in rankings.items():
   g=set(q['gold']);metrics[method]={'hit_at_1':int(bool(ids) and ids[0] in g),'recall_at_3':len(set(ids[:3])&g)/len(g),'mrr':next((1/(i+1) for i,cid in enumerate(ids) if cid in g),0)}
  rows.append({'index':kind,**q,'rankings':rankings,'metrics':metrics,'bm25_scores':{c['id']:float(ss[i]) for i,c in enumerate(chunks)},'dense_scores':{c['id']:float(ds[i]) for i,c in enumerate(chunks)}})
summary={}
for kind in ('plain','contextual'):
 group=[r for r in rows if r['index']==kind]
 for method in ('bm25','dense','hybrid'):summary[kind+'_'+method]={m:float(np.mean([r['metrics'][method][m] for r in group])) for m in ('hit_at_1','recall_at_3','mrr')}
report={'status':'completed','scope':'mechanism-equivalent authored bee corpus: 4 complete documents, 12 chunks, 6 fixed labelled queries; not official full benchmark','mechanism':'LLM reads full source document plus target chunk at indexing time, generates prefix, then rebuilds sparse/dense indexes from prefix+original text. Original chunk unchanged; same queries and retrieval methods.','models':{'prefix':'Qwen/Qwen3-0.6B','embedding':'BAAI/bge-small-zh-v1.5'},'controls':'Plain branch indexes chunk body only. Contextual branch adds model-generated source context. Identical source IDs, labels, tokenizer, encoder, fusion, queries. Source titles retained as metadata in both but not used for filtering.','metadata_alternative':'This designed corpus has sufficient title metadata for an engineered title prefix or metadata filter; LLM is not necessary here. That cheaper alternative was not measured.','prefix_usage':{'input_tokens':sum(r['input_tokens'] for r in receipts),'output_tokens':sum(r['output_tokens'] for r in receipts),'generation_ms':prefix_ms,'note':'Local compute observed; no API billed-price claim.'},'documents':documents,'chunks':chunks,'summary':summary,'results':rows,'quality_limits':'Number checks are only a warning heuristic, not prefix entailment verification. Query labels authored for this teaching contrast. Retrieval gain is not answer correctness.'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETED '+json.dumps(summary),flush=True)
