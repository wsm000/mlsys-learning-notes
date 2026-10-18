"""3-4 teaching: identical real vectors, exact search, ANNOY and HNSW."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from annoy import AnnoyIndex
import hnswlib
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment04-equivalent-20261018';OUT.mkdir(parents=True,exist_ok=True)
topics=[('分蜂','蜂群出现王台、工蜂拥挤，需要检查分蜂征兆。','怎样判断蜂群准备分家？'),('蜂王','检查蜂王是否存在，观察产卵和卵虫分布。','怎样确定蜂箱里还有蜂王？'),('饲喂','缺蜜期补充糖浆，注意饲喂器卫生。','没有蜜源的时候怎样补充食物？'),('蜂螨','监测寄生蜂螨的数量，选择合适的综合防治措施。','蜜蜂身上的寄生虫怎么监测？'),('越冬','越冬需要足够饲料、健康蜂群和适当保温通风。','冬季如何保护蜂群？'),('采蜜','成熟蜂蜜通常有封盖，采收时避免混入未成熟蜜。','什么时候可以收获成熟蜂蜜？'),('盗蜂','缺蜜时缩小巢门，及时清理洒出的糖浆以减少盗蜂。','别的蜂来抢食怎么办？'),('通风','高温时改善遮阴和空气流通，保持饮水供应。','天气很热时怎样帮助蜂群散热？'),('蜂箱','定期检查箱体破损、漏雨和底板积水。','木箱进水该检查哪里？'),('花粉','花粉为幼虫生长提供蛋白质，缺粉会影响育虫。','幼虫的蛋白质来自哪里？'),('农药','附近施药前协调时间，减少蜂群暴露和污染。','周围农田喷药要注意什么？'),('运输','转运蜂群时固定巢脾、保证通风并减少颠簸。','搬蜂箱时怎样避免闷死蜜蜂？'),('疾病','异常幼虫和腐臭气味需要及时检查并隔离可疑蜂群。','发现幼虫异常应该怎么办？'),('记录','巡蜂记录应包含日期、蜂群编号、蜂王状态和处理措施。','每次检查后需要记下哪些信息？'),('饮水','为蜂群提供稳定清洁的饮水点并防止溺水。','蜜蜂喝水的地方怎样设置？')]
docs=[{'id':i,'topic':topics[i%15][0],'text':f"蜂场{i//15+1}号观察记录，主题：{topics[i%15][0]}。{topics[i%15][1]} 检查批次{i}。"} for i in range(300)]
queries=[x[2] for x in topics]
torch.set_num_threads(4)
model_id='BAAI/bge-small-zh-v1.5'
tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
model=AutoModel.from_pretrained(model_id,local_files_only=True).to('cuda').eval()
def encode(texts,query=False):
 vectors=[]
 for start in range(0,len(texts),32):
  values=texts[start:start+32]
  if query:values=['为这个句子生成表示以用于检索相关文章：'+t for t in values]
  batch=tok(values,padding=True,truncation=True,max_length=192,return_tensors='pt').to('cuda')
  with torch.inference_mode():v=model(**batch).last_hidden_state[:,0]
  vectors.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy())
 return np.concatenate(vectors).astype('float32')
begin=time.perf_counter();vectors=encode([d['text'] for d in docs]);qv=encode(queries,True);embedding_ms=(time.perf_counter()-begin)*1000
np.save(OUT/'document-vectors.npy',vectors);np.save(OUT/'query-vectors.npy',qv)
k=5;truth=np.argsort(-(qv@vectors.T),axis=1)[:,:k];dim=vectors.shape[1]
results={}
for name in ('annoy','hnsw'):
 begin=time.perf_counter()
 if name=='annoy':
  index=AnnoyIndex(dim,'angular')
  for i,v in enumerate(vectors):index.add_item(i,v)
  index.set_seed(37);index.build(10);index.save(str(OUT/'annoy.index'))
  search=lambda q:index.get_nns_by_vector(q,k,search_k=1000)
 else:
  index=hnswlib.Index(space='cosine',dim=dim);index.init_index(max_elements=400,ef_construction=100,M=16,random_seed=37)
  index.add_items(vectors,np.arange(len(vectors)));index.set_ef(100);index.save_index(str(OUT/'hnsw.index'))
  search=lambda q:index.knn_query(q,k=k)[0][0].tolist()
 build_ms=(time.perf_counter()-begin)*1000
 rankings=[];latencies=[]
 for qi,q in enumerate(qv):
  begin=time.perf_counter();found=search(q);latencies.append((time.perf_counter()-begin)*1000)
  rankings.append({'query':queries[qi],'exact_ids':truth[qi].tolist(),'ann_ids':found,'recall_at_5':len(set(found)&set(truth[qi]))/k,'top_documents':[docs[i] for i in found]})
 results[name]={'build_ms':build_ms,'query_mean_ms':float(np.mean(latencies)),'index_bytes':(OUT/(name+'.index')).stat().st_size,'mean_exact_neighbor_recall_at_5':float(np.mean([r['recall_at_5'] for r in rankings])),'rankings':rankings}
 # Same 200 -> 300 document update, including the ANNOY rebuild cost.
 if name=='hnsw':
  inc=hnswlib.Index(space='cosine',dim=dim);inc.init_index(max_elements=400,ef_construction=100,M=16,random_seed=37);inc.add_items(vectors[:200],np.arange(200));inc.set_ef(100)
  begin=time.perf_counter();inc.add_items(vectors[200:],np.arange(200,300));update_ms=(time.perf_counter()-begin)*1000
  updated=inc.knn_query(qv,k=k)[0]
 else:
  initial=AnnoyIndex(dim,'angular')
  for i,v in enumerate(vectors[:200]):initial.add_item(i,v)
  initial.build(10)
  begin=time.perf_counter();initial.unbuild()
  for i in range(200,300):initial.add_item(i,vectors[i])
  initial.build(10);update_ms=(time.perf_counter()-begin)*1000
  updated=[initial.get_nns_by_vector(q,k,search_k=1000) for q in qv]
 results[name]['update']={'initial_documents':200,'added':100,'requires_rebuild':name=='annoy','latency_ms':update_ms,'mean_exact_neighbor_recall_at_5':float(np.mean([len(set(a)&set(b))/k for a,b in zip(updated,truth)]))}
examples=[]
for qi,q in enumerate(queries):
 ids=truth[qi].tolist();examples.append({'question':q,'question_vector_first8':qv[qi,:8].tolist(),'exact_top5':[{'document':docs[i],'cosine_similarity':float(qv[qi]@vectors[i])} for i in ids]})
report={'status':'completed','scope':'mechanism equivalent; real BGE embeddings and real ANNOY/HNSW, teaching corpus','model':model_id,'documents':300,'queries':15,'dimension':dim,'embedding_ms':embedding_ms,'quality_note':'ANN recall compares exact vector neighbors; it does not establish factual correctness or answer quality. Small synthetic corpus timings are illustrative only.','results':results,'examples':examples}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({n:{key:value for key,value in r.items() if key!='rankings'} for n,r in results.items()},ensure_ascii=False),flush=True)
