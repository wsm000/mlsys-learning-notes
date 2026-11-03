"""Discover factors, extract cases, build prototypes, and match held-out cases."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,re,time,hashlib,warnings
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/experiment12-equivalent-20261018';OUT.mkdir(parents=True,exist_ok=True)
def observation(group,i):
 if group==0:return f'巡检时气温{7+i%3}摄氏度，蜂群活动偏少。记录明确写明饲料储备不足，通风正常，未发现蜂螨迹象。'
 if group==1:return f'巡检时气温{34+i%3}摄氏度，蜂群聚集在箱口。记录明确写明饲料充足，通风不足，饮水点缺水，未发现蜂螨迹象。'
 return f'巡检时气温{20+i%3}摄氏度，蜂群活动异常。记录明确写明饲料充足，通风正常，复查发现蜂螨迹象。'
actions=['记录补给与保温处理','记录改善环境与供水处理','记录转交专业人员复查']
training=[]
for group in range(3):
 for i in range(10):training.append({'id':f'T{len(training)+1:03d}','observation':observation(group,i),'recorded_action':actions[group],'recorded_outcome':'后续活动恢复' if (i+group)%4 else '后续仍需观察'})
heldout=[]
for group in range(3):
 for i in range(2):heldout.append({'id':f'V{len(heldout)+1:03d}','observation':'另一次独立编号的巡检：'+observation(group,i+4),'recorded_action_for_evaluation_only':actions[group]})
heldout.extend([
 {'id':'V007','observation':'气温8摄氏度，饲料充足，通风正常，复查发现蜂螨迹象。','recorded_action_for_evaluation_only':None},
 {'id':'V008','observation':'气温35摄氏度，饲料不足，通风不足，饮水点缺水，复查发现蜂螨迹象。','recorded_action_for_evaluation_only':None},
 {'id':'V009','observation':'仅记录蜂群活动偏少，气温、饲料、通风及蜂螨检查结果均未记录。','recorded_action_for_evaluation_only':None},
])
(OUT/'dataset.json').write_text(json.dumps({'training':training,'heldout':heldout,'provenance':'Authored fictional teaching records. Outcomes and actions designed with patterned associations; no real management effectiveness or causal claim.'},ensure_ascii=False,indent=2),encoding='utf-8')
assert not set(c['id'] for c in training)&set(c['id'] for c in heldout)
torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True,dtype=torch.float16,device_map='cuda').eval();receipts=[]
def call(label,system,user,max_tokens=384):
 messages=[{'role':'system','content':system},{'role':'user','content':user}]
 encoded=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda');begin=time.perf_counter()
 with torch.inference_mode():output=model.generate(**encoded,max_new_tokens=max_tokens,do_sample=False)
 raw=tok.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
 receipts.append({'label':label,'messages':messages,'raw_output':raw,'input_tokens':encoded['input_ids'].shape[1],'output_tokens':output.shape[1]-encoded['input_ids'].shape[1],'latency_ms':(time.perf_counter()-begin)*1000,'finish':'length' if output.shape[1]-encoded['input_ids'].shape[1]>=max_tokens else 'eos'})
 (OUT/'receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8');print('CALL '+label,flush=True);return raw
def parse(raw):return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip()))
sample=[training[i]['observation'] for i in (0,3,6,9,10,13,16,19,20,23,26,29)]
raw=call('training-factor-discovery','你是案例资料整理员。仅根据训练观察发现3到6个有区分度、可从原文判断有无的二元因素。每个因素只表达一个条件。不要将处理措施、结果或案例编号当因素。输出JSON对象，factors字段为数组，每项含key（唯一英文下划线名）和description（中文条件）。不要解释。','训练观察：\n'+'\n'.join(sample),512)
try:
 obj=parse(raw);schema=obj if isinstance(obj,list) else obj['factors']
 assert isinstance(schema,list) and 2<=len(schema)<=8
 assert all(isinstance(f,dict) and isinstance(f.get('key'),str) and isinstance(f.get('description'),str) for f in schema)
 assert len({f['key'] for f in schema})==len(schema)
except Exception as e:
 (OUT/'results.json').write_text(json.dumps({'status':'partial','stage':'factor discovery','error':str(e),'raw_output':raw},ensure_ascii=False,indent=2),encoding='utf-8');raise
(OUT/'discovered-schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
extracted=[]
for split,records in [('training',training),('heldout',heldout)]:
 for record in records:
  raw=call('extract-'+record['id'],'按给定因素顺序从观察中提取值。条件存在=1，明确不存在=0，无法判断=-1。只输出JSON数字数组，每个因素一个值。注意否定词。不要解释，不要推测未记录的信息。','因素顺序：'+json.dumps(schema,ensure_ascii=False)+'\n观察：'+record['observation'],128)
  error=None
  try:
   values=parse(raw)
   if isinstance(values,dict):values=[values.get(f['key'],-1) for f in schema]
   if not isinstance(values,list) or len(values)!=len(schema):raise ValueError('wrong vector length')
   if any(not isinstance(v,(int,float,bool)) or v not in (-1,0,1) for v in values):raise ValueError('non-binary feature')
   values=[int(v) for v in values]
  except Exception as e:error=str(e);values=[-1]*len(schema)
  extracted.append({'id':record['id'],'split':split,'values':values,'parse_error':error})
  (OUT/'extracted-features.json').write_text(json.dumps(extracted,ensure_ascii=False,indent=2),encoding='utf-8')
train_values=np.array([e['values'] for e in extracted if e['split']=='training'],dtype=float)
test_values=np.array([e['values'] for e in extracted if e['split']=='heldout'],dtype=float)
train_values[train_values==-1]=np.nan;test_values[test_values==-1]=np.nan
with warnings.catch_warnings():
 warnings.simplefilter('ignore',RuntimeWarning);medians=np.nanmedian(train_values,axis=0)
medians=np.where(np.isnan(medians),.5,medians)
X=np.where(np.isnan(train_values),medians,train_values);V=np.where(np.isnan(test_values),medians,test_values)
scaler=StandardScaler().fit(X);Z=scaler.transform(X);W=scaler.transform(V)
unique=len(np.unique(Z,axis=0));k=min(3,unique)
clusterer=KMeans(n_clusters=k,n_init=10,random_state=37).fit(Z);assignments=clusterer.labels_
distances=np.linalg.norm(Z-clusterer.cluster_centers_[assignments],axis=1)
threshold=max(.5,float(np.percentile(distances,95)))
prototypes=[]
for cluster in range(k):
 indices=np.where(assignments==cluster)[0];action_counts=Counter(training[i]['recorded_action'] for i in indices);outcomes=Counter(training[i]['recorded_outcome'] for i in indices)
 prototypes.append({'cluster':cluster,'training_case_ids':[training[i]['id'] for i in indices],'count':len(indices),'factor_means':{f['key']:float(np.mean(X[indices,j])) for j,f in enumerate(schema)},'recorded_actions':dict(action_counts),'recorded_outcomes':dict(outcomes),'most_common_recorded_action':action_counts.most_common(1)[0][0]})
classifier=DecisionTreeClassifier(max_depth=3,random_state=37).fit(Z,[r['recorded_action'] for r in training])
importance={f['key']:float(classifier.feature_importances_[i]) for i,f in enumerate(schema)}
predictions=[]
for i,record in enumerate(heldout):
 cluster=int(clusterer.predict(W[i:i+1])[0]);distance=float(np.linalg.norm(W[i]-clusterer.cluster_centers_[cluster]));proto=prototypes[cluster]
 predictions.append({'id':record['id'],'observation':record['observation'],'cluster':cluster,'distance':distance,'outside_training_distance_threshold':distance>threshold,'unknown_factor_count':int(np.isnan(test_values[i]).sum()),'prototype_common_action':proto['most_common_recorded_action'],'decision_tree_action':str(classifier.predict(W[i:i+1])[0]),'heldout_recorded_action':record['recorded_action_for_evaluation_only'],'prototype_action_matches_record':proto['most_common_recorded_action']==record['recorded_action_for_evaluation_only'] if record['recorded_action_for_evaluation_only'] else None})
known=[p for p in predictions if p['heldout_recorded_action'] is not None]
advices=[]
for i in (0,2,4,8):
 pred=predictions[i];proto=prototypes[pred['cluster']]
 text=call('prototype-report-'+pred['id'],'你在解释虚构教学案例的匹配结果。仅描述最接近的训练簇、该簇记录数量与常见处置记录，不把相关性说成因果或疗效，不提供实际养蜂处方。缺信息或偏离训练分布时明确不能可靠类推。不超过150字。','新观察：'+pred['observation']+'\n匹配诊断：'+json.dumps({k:v for k,v in pred.items() if k not in ('heldout_recorded_action','prototype_action_matches_record')},ensure_ascii=False)+'\n训练簇统计：'+json.dumps(proto,ensure_ascii=False),256)
 advices.append({'heldout_id':pred['id'],'model_report':text})
report={'status':'completed','scope':'mechanism-equivalent fictional bee cases, not official CAIL2018 discovery/legal benchmark','mechanism':'Training observations -> live factor discovery -> fixed schema -> live per-case extraction -> training-only imputation/scaling/KMeans -> prototype action/outcome statistics -> heldout nearest prototype and limited narrative.','model':'Qwen/Qwen3-0.6B','split':{'training':len(training),'heldout':len(heldout),'labelled_heldout':len(known),'unlabelled_mixed_or_missing':3},'leakage_controls':'Heldout observations never used for factor discovery, fitting imputation/scaler/clusters/tree, or prototype statistics. Training actions/outcomes excluded from discovery/extraction prompts, used only after features exist. Heldout action labels excluded from model reports and feature extraction.','schema':schema,'extraction_parse_errors':sum(e['parse_error'] is not None for e in extracted),'train_unknown_values':int(np.isnan(train_values).sum()),'test_unknown_values':int(np.isnan(test_values).sum()),'training_imputation_values':medians.tolist(),'clusters':k,'unique_training_vectors':unique,'silhouette':float(silhouette_score(Z,assignments)) if 1<k<len(Z) else None,'prototypes':prototypes,'predictive_feature_importance':importance,'importance_target':'Training recorded-action labels; predictive association in authored data, not causal factor weights.','distance_threshold':threshold,'threshold_note':'Illustrative max(0.5, training within-cluster distance 95th percentile), not calibrated OOD detection. Missingness separately retained.','heldout_predictions':predictions,'prototype_action_match_count':sum(p['prototype_action_matches_record'] for p in known),'labelled_heldout_count':len(known),'narratives':advices,'limitations':'Small templated records; heldout phrasing resembles training templates; action/outcome associations designed by author. High match rate would not show real-world effectiveness, generalization or causality. No independent judge. Narratives may still overstate prototype evidence and must be checked.'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
np.save(OUT/'training-features.npy',X);np.save(OUT/'heldout-features.npy',V)
print('COMPLETED '+json.dumps({'schema':schema,'clusters':k,'parse_errors':report['extraction_parse_errors'],'heldout_matches':report['prototype_action_match_count'],'labelled_heldout':len(known)},ensure_ascii=False),flush=True)
