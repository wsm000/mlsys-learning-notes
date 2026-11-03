"""Teaching equivalent: live summaries, live entity relations, source tracing."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json,re,time
from pathlib import Path
from collections import defaultdict,deque
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment07-equivalent-20261018';OUT.mkdir(parents=True,exist_ok=True)
chunks=[
 {'id':'C1','group':'青禾蜂场','text':'青禾蜂场位于河谷镇。青禾蜂场管理蜂群A。'},
 {'id':'C2','group':'青禾蜂场','text':'蜂群A发现蜂螨。蜂螨防治参考手册M。'},
 {'id':'C3','group':'青禾蜂场','text':'手册M要求先监测蜂螨数量，再决定防治措施。'},
 {'id':'C4','group':'青禾蜂场','text':'河谷镇冬季寒冷。青禾蜂场的冬季管理包括保温和饲料检查。'},
 {'id':'C5','group':'白石蜂场','text':'白石蜂场位于山岭镇。白石蜂场管理蜂群B。'},
 {'id':'C6','group':'白石蜂场','text':'蜂群B缺少饲料。缺料处理参考手册N。'},
 {'id':'C7','group':'白石蜂场','text':'手册N要求补充饲料并检查饲喂器卫生。'},
 {'id':'C8','group':'白石蜂场','text':'山岭镇冬季多风。白石蜂场的冬季管理包括防风和饮水检查。'},
]
(OUT/'input-chunks.json').write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding='utf-8')
torch.set_num_threads(4)
model_id='Qwen/Qwen3-0.6B'
tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(model_id,local_files_only=True,dtype=torch.float16,device_map='cuda').eval()
receipts=[]
def call(label,system,user,max_tokens=384):
 messages=[{'role':'system','content':system},{'role':'user','content':user}]
 batch=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda')
 begin=time.perf_counter()
 with torch.inference_mode():outputs=model.generate(**batch,max_new_tokens=max_tokens,do_sample=False)
 text=tok.decode(outputs[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
 receipts.append({'label':label,'messages':messages,'raw_output':text,'input_tokens':batch['input_ids'].shape[1],'output_tokens':outputs.shape[1]-batch['input_ids'].shape[1],'latency_ms':(time.perf_counter()-begin)*1000,'finish':'length' if outputs.shape[1]-batch['input_ids'].shape[1]>=max_tokens else 'eos'})
 (OUT/'receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
 print('CALL '+label,flush=True)
 return text
def sources(ids):return '\n'.join('['+c['id']+'] '+c['text'] for c in chunks if c['id'] in ids)
nodes=[]
summary_system='你是资料整理员。只根据输入写简短中文摘要，保留蜂场名、蜂群编号、问题、手册编号和管理措施。不要增加常识或建议。'
for group in ('青禾蜂场','白石蜂场'):
 ids=[c['id'] for c in chunks if c['group']==group]
 text=call('summary-'+group,summary_system,sources(ids),256)
 nodes.append({'id':'S'+str(len(nodes)+1),'level':1,'group':group,'children':ids,'source_ids':ids,'text':text})
root_text=call('summary-root',summary_system,'\n'.join('['+n['id']+'] '+n['text'] for n in nodes),320)
tree={'root':{'id':'ROOT','level':2,'children':[n['id'] for n in nodes],'source_ids':[c['id'] for c in chunks],'text':root_text},'summaries':nodes,'leaves':chunks,'grouping':'explicit farm metadata, not automatic RAPTOR clustering'}
(OUT/'summary-tree.json').write_text(json.dumps(tree,ensure_ascii=False,indent=2),encoding='utf-8')
edges=[];rejected=[];parse_errors=[]
graph_prompt='''只从资料提取明确的有向关系，输出JSON对象，字段edges为数组。每条包含subject、relation、object。subject和object必须原样出现在资料里，不要发明实体。拆成简短实体，例如蜂场管理蜂群、蜂群存在问题、问题参考手册、手册要求措施；每条只连接两个节点。不要把完整句子当主体。没有关系就返回空数组。不要解释。'''
for chunk in chunks:
 raw=call('graph-'+chunk['id'],graph_prompt,chunk['text'],384)
 try:
  clean=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip())
  value=json.loads(clean)
  for e in value.get('edges',[]):
   if not isinstance(e,dict):rejected.append({'chunk_id':chunk['id'],'edge':e,'reason':'not object'});continue
   s=e.get('subject');o=e.get('object');r=e.get('relation')
   if not all(isinstance(x,str) and x for x in (s,o,r)) or s not in chunk['text'] or o not in chunk['text']:
    rejected.append({'chunk_id':chunk['id'],'edge':e,'reason':'missing fields or endpoint absent from source'});continue
   edges.append({'subject':s,'relation':r,'object':o,'source_id':chunk['id'],'source_text':chunk['text']})
 except Exception as e:parse_errors.append({'chunk_id':chunk['id'],'error':str(e)})
adj=defaultdict(list)
for i,e in enumerate(edges):adj[e['subject']].append(i)
def traverse(start,max_hops=4):
 queue=deque([(start,0,[])]);seen={start};used=set();paths=[]
 while queue:
  entity,depth,path=queue.popleft()
  if depth>=max_hops:continue
  for i in adj.get(entity,[]):
   edge=edges[i];used.add(i);newpath=path+[i];paths.append({'end_entity':edge['object'],'edge_indices':newpath})
   if edge['object'] not in seen:seen.add(edge['object']);queue.append((edge['object'],depth+1,newpath))
 return {'start':start,'max_hops':max_hops,'visited_entities':sorted(seen),'edge_indices':sorted(used),'paths':paths,'source_ids':sorted({edges[i]['source_id'] for i in used})}
walk=traverse('青禾蜂场')
graph={'edges':edges,'rejected_edges':rejected,'parse_errors':parse_errors,'traversal':walk,'validation_limit':'Endpoint substring checks only; relation meaning/direction still require source review. No alias normalization or automatic query planning.'}
(OUT/'relation-graph.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
questions=[{'id':'overview','question':'青禾蜂场主要需要关注哪些管理事项？','gold_source_ids':['C2','C4']},
 {'id':'multihop','question':'青禾蜂场管理的蜂群遇到了什么问题，应参考哪本手册，手册要求先做什么？','gold_source_ids':['C1','C2','C3']}]
answers=[]
answer_system='只依据提供的证据回答，引用[C编号]或[S编号]。证据不足就明确说不知道，不得使用养蜂常识补全。回答不超过150字。'
for q in questions:
 # Teaching navigation uses known farm metadata; compare abstract node vs underlying leaves.
 selected=nodes[0]
 contexts={'tree_summary':'[S1] '+selected['text'],
           'tree_leaf_expansion':sources(selected['source_ids']),
           'graph_walk': '\n'.join(f"[{e['source_id']}] {e['subject']} --{e['relation']}--> {e['object']}" for i,e in enumerate(edges) if i in walk['edge_indices'])}
 for method,context in contexts.items():
  text=call(q['id']+'-'+method,answer_system,'问题：'+q['question']+'\n证据：\n'+context,256)
  supplied=selected['source_ids'] if method.startswith('tree') else walk['source_ids']
  answers.append({'question':q,'method':method,'navigation':{'selected_summary':'S1','children':selected['children']} if method.startswith('tree') else walk,'context':context,'answer':text,'source_coverage':len(set(supplied)&set(q['gold_source_ids']))/len(q['gold_source_ids']),'coverage_note':'For summary-only arm this counts provenance, not whether summary retained each required fact.'})
report={'status':'completed','scope':'mechanism-equivalent authored bee corpus, not official Intel RAPTOR/GraphRAG benchmark','model':model_id,'tree_mechanism':'live bottom-up summaries, explicit metadata grouping, original leaves retained and expanded; no learned clustering or semantic navigation','graph_mechanism':'live relation extraction with endpoint grounding, directed four-hop traversal from manually specified start entity; no community summaries or alias normalization','summary_tree':tree,'graph':graph,'answers':answers,'quality_note':'Completed execution is not proof all summaries, triples or answers are correct. Original sources and raw calls retained for inspection.'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETED '+json.dumps({'edges':len(edges),'rejected':len(rejected),'parse_errors':parse_errors,'visited':walk['visited_entities'],'sources':walk['source_ids']},ensure_ascii=False),flush=True)
