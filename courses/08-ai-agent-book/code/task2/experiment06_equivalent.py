"""Hybrid retrieval and live local joint query-document reranking."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json,math,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment06-equivalent-20261018';OUT.mkdir(parents=True,exist_ok=True)
docs=[
 ('D1','蜂群出现王台、工蜂拥挤，是需要进一步检查的分蜂征兆。','蜂群 王台 工蜂 拥挤 分蜂 检查'),
 ('D2','蜂王正常产卵时，巢房内可以观察到卵和幼虫。','蜂王 产卵 巢房 卵 幼虫'),
 ('D3','蜂箱控制器 AC-17 出现 E42，表示温度传感器断开，应检查连接线。','蜂箱 控制器 AC-17 E42 温度 传感器 断开 检查 连接线'),
 ('D4','蜂箱控制器 AC-71 出现 E24，表示风扇停转，应检查风扇电源。','蜂箱 控制器 AC-71 E24 风扇 停转 检查 电源'),
 ('D5','蜂群越冬应准备充足饲料，适当保温，同时保持通风。','蜂群 越冬 饲料 保温 通风'),
 ('D6','缺蜜时可以适量补充糖浆；饲喂器需要定期清洗。','缺蜜 糖浆 饲喂 饲喂器 清洗'),
 ('D7','花粉为蜜蜂幼虫生长提供蛋白质，是育虫的重要营养来源。','花粉 蜜蜂 幼虫 蛋白质 育虫 营养'),
 ('D8','代码接口 inspect_hive(hive_id) 返回蜂箱检查记录列表，不修改蜂群状态。','inspect_hive hive_id 返回 检查 记录 列表 状态'),
 ('D9','代码接口 feed_hive(hive_id, syrup_ml) 写入饲喂记录，返回操作是否成功。','feed_hive hive_id syrup_ml 饲喂 记录 返回 成功'),
 ('D10','E42 促销活动：AC-17 控制器外壳享折扣。本条为销售广告，不介绍故障排查。','E42 AC-17 控制器 外壳 折扣 促销 广告'),
 ('D11','王台商品目录包含分蜂用品、蜂王笼和检查工具。本条为商品清单。','王台 分蜂 蜂王 检查 商品 工具'),
 ('D12','仓库冬季保温材料采购清单，与活蜂群的越冬管理无关。','仓库 冬季 保温 材料 采购 清单 越冬'),
]
queries=[
 {'id':'Q1','question':'蜂群准备分家时会有哪些表现？','tokens':['分家','表现'],'gold':['D1'],'category':'synonym'},
 {'id':'Q2','question':'AC-17 的 E42 故障该怎样处理？','tokens':['AC-17','E42'],'gold':['D3'],'category':'exact identifier with ad distractor'},
 {'id':'Q3','question':'Where do bee larvae get protein?','tokens':['where','do','bee','larvae','get','protein'],'gold':['D7'],'category':'cross-language'},
 {'id':'Q4','question':'inspect_hive(hive_id) 返回什么？','tokens':['inspect_hive','hive_id','返回'],'gold':['D8'],'category':'code identifier'},
 {'id':'Q5','question':'冬天蜂群怎么保温？','tokens':['冬季','蜂群','保温'],'gold':['D5'],'category':'topic with warehouse distractor'},
 {'id':'Q6','question':'蜂王产卵时，在巢房里可以看到什么？','tokens':['蜂王','产卵','巢房'],'gold':['D2'],'category':'exact terms'},
]
torch.set_num_threads(4)
postings=defaultdict(dict);lengths={}
for doc_id,text,seg in docs:
 tokens=seg.lower().split();lengths[doc_id]=len(tokens)
 for token,tf in Counter(tokens).items():postings[token][doc_id]=tf
N=len(docs);avgdl=sum(lengths.values())/N
def sparse_scores(tokens):
 scores={d[0]:0.0 for d in docs};traces={d[0]:[] for d in docs}
 for token in tokens:
  token=token.lower();hits=postings.get(token,{})
  idf=math.log(1+(N-len(hits)+.5)/(len(hits)+.5))
  for doc_id,tf in hits.items():
   value=idf*tf*2.5/(tf+1.5*(.25+.75*lengths[doc_id]/avgdl))
   scores[doc_id]+=value;traces[doc_id].append({'token':token,'tf':tf,'idf':idf,'contribution':value})
 return scores,traces
embed_id='BAAI/bge-small-zh-v1.5'
tokenizer=AutoTokenizer.from_pretrained(embed_id,local_files_only=True)
encoder=AutoModel.from_pretrained(embed_id,local_files_only=True).to('cuda').eval()
def encode(texts,query=False):
 if query:texts=['为这个句子生成表示以用于检索相关文章：'+t for t in texts]
 batch=tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
 with torch.inference_mode():v=encoder(**batch).last_hidden_state[:,0]
 return torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy()
begin=time.perf_counter();dv=encode([d[1] for d in docs]);qv=encode([q['question'] for q in queries],True);embedding_ms=(time.perf_counter()-begin)*1000
np.save(OUT/'document-vectors.npy',dv);np.save(OUT/'query-vectors.npy',qv)
del encoder;torch.cuda.empty_cache()
reranker_id='Qwen/Qwen3-0.6B'
rtok=AutoTokenizer.from_pretrained(reranker_id,local_files_only=True)
reranker=AutoModelForCausalLM.from_pretrained(reranker_id,local_files_only=True,dtype=torch.float16,device_map='cuda').eval()
yes_ids=rtok.encode('yes',add_special_tokens=False);no_ids=rtok.encode('no',add_special_tokens=False)
assert len(yes_ids)==len(no_ids)==1
docmap={d[0]:d[1] for d in docs};rows=[];receipts=[]
def rank(scores):return sorted(scores,key=lambda x:(-scores[x],x))
def normalize(scores):
 lo=min(scores.values());hi=max(scores.values())
 return {d:(v-lo)/(hi-lo) if hi>lo else 0.0 for d,v in scores.items()}
def metrics(ids,gold,k=3):
 selected=ids[:k];g=set(gold)
 hits=[int(x in g) for x in selected]
 return {'recall_at_3':len(set(selected)&g)/len(g),'mrr':next((1/(i+1) for i,x in enumerate(ids) if x in g),0.0),'ndcg_at_3':sum(h/math.log2(i+2) for i,h in enumerate(hits))/sum(1/math.log2(i+2) for i in range(min(k,len(g))))}
for qi,query in enumerate(queries):
 begin=time.perf_counter();sparse,traces=sparse_scores(query['tokens']);sparse_rank=[d for d in rank(sparse) if sparse[d]>0];sparse_ms=(time.perf_counter()-begin)*1000
 begin=time.perf_counter();dense={d[0]:float(qv[qi]@dv[i]) for i,d in enumerate(docs)};dense_rank=rank(dense);dense_search_ms=(time.perf_counter()-begin)*1000
 # Retrieve five from each arm; rerank only the first six fused candidates.
 a=sparse_rank[:5];b=dense_rank[:5];candidate_set=set(a+b)
 contributions={d:{'sparse':1/(60+a.index(d)+1) if d in a else 0.0,'dense':1/(60+b.index(d)+1) if d in b else 0.0} for d in candidate_set}
 rrf={d:sum(contributions[d].values()) for d in candidate_set};rrf_rank=rank(rrf)
 ns=normalize(sparse);nd=normalize(dense)
 weighted={d:.5*ns[d]+.5*nd[d] for d in candidate_set};weighted_rank=rank(weighted)
 selected=rrf_rank[:6];scores={};begin=time.perf_counter()
 for doc_id in selected:
  messages=[{'role':'system','content':'Decide whether the document directly helps answer the question. Judge meaning, not just shared words. Advertisements or unrelated records are not useful. Answer only yes or no.'},{'role':'user','content':f"Question: {query['question']}\nDocument: {docmap[doc_id]}"}]
  encoded=rtok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda')
  with torch.inference_mode():logits=reranker(**encoded).logits[0,-1].float()
  yes=float(logits[yes_ids[0]]);no=float(logits[no_ids[0]]);score=float(torch.softmax(torch.tensor([yes,no]),dim=0)[0]);scores[doc_id]=score
  receipts.append({'query_id':query['id'],'document_id':doc_id,'messages':messages,'yes_token':yes_ids[0],'no_token':no_ids[0],'yes_logit':yes,'no_logit':no,'relative_yes_score':score})
 rerank_ms=(time.perf_counter()-begin)*1000
 rerank_ids=sorted(selected,key=lambda d:(-scores[d],rrf_rank.index(d)))
 methods={'sparse':sparse_rank,'dense':dense_rank,'rrf':rrf_rank,'weighted':weighted_rank,'joint_reranked_rrf':rerank_ids}
 rows.append({**query,'sparse_scores':sparse,'sparse_term_contributions':traces,'dense_scores':dense,'rrf_contributions':contributions,'rrf_scores':rrf,'weighted_scores':weighted,'rerank_candidate_ids':selected,'rerank_scores':scores,'rankings':methods,'metrics':{name:metrics(ids,query['gold']) for name,ids in methods.items()},'timing_ms':{'sparse':sparse_ms,'dense_vector_search':dense_search_ms,'reranking':rerank_ms}})
 print(query['id']+' '+json.dumps({name:ids[:3] for name,ids in methods.items()}),flush=True)
summary={name:{metric:float(np.mean([r['metrics'][name][metric] for r in rows])) for metric in ('recall_at_3','mrr','ndcg_at_3')} for name in rows[0]['rankings']}
report={'status':'completed','scope':'mechanism-equivalent teaching experiment; 12 authored documents and 6 labelled queries, not a broad benchmark','embedding':embed_id,'reranker':{'model':reranker_id,'method':'local decoder jointly reads question and document; relative next-token yes/no probability','equivalence':'illustrates neural joint relevance scoring; not a pretrained cross-encoder reranker and no improvement is assumed','score_caution':'Relative yes/no score is not a calibrated probability of relevance.'},'segmentation':'Explicit authored lexical tokens isolate sparse scoring; no automatic Chinese tokenizer evaluation','fusion':{'rrf_constant':60,'weighted':'0.5 per-query minmax sparse + 0.5 per-query minmax dense','retrieval_per_arm':5,'rerank_candidates':6},'embedding_time_ms':embedding_ms,'documents':[{'id':d[0],'text':d[1],'tokens':d[2]} for d in docs],'summary':summary,'results':rows}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'rerank-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETED '+json.dumps(summary),flush=True)
