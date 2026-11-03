"""Controlled schema refinement and transparent extraction for explicit records."""
import json,re,math
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/experiment12-equivalent-20261018'
data=json.loads((OUT/'dataset.json').read_text());train=data['training'];heldout=data['heldout']
schema=[{'key':'temperature_c','type':'number','meaning':'原文明示的摄氏温度，不自行定义低温阈值'},
 {'key':'food_shortage','type':'binary','meaning':'原文明示饲料不足=1、充足=0，否则缺失'},
 {'key':'poor_ventilation','type':'binary','meaning':'原文明示通风不足=1、正常=0，否则缺失'},
 {'key':'mite_present','type':'binary','meaning':'原文明示发现蜂螨=1、未发现=0，否则缺失'}]
def extract(text):
 match=re.search(r'气温(\d+)摄氏度',text)
 temperature=float(match.group(1)) if match else np.nan
 food=1 if re.search(r'饲料(?:储备)?不足',text) else (0 if '饲料充足' in text else np.nan)
 ventilation=1 if '通风不足' in text else (0 if '通风正常' in text else np.nan)
 mites=0 if '未发现蜂螨迹象' in text else (1 if '发现蜂螨迹象' in text else np.nan)
 return [temperature,food,ventilation,mites]
X=np.array([extract(r['observation']) for r in train]);V=np.array([extract(r['observation']) for r in heldout])
assert not np.isnan(X).any()
scaler=StandardScaler().fit(X);Z=scaler.transform(X)
model=KMeans(n_clusters=3,n_init=10,random_state=37).fit(Z)
training_dist=np.linalg.norm(Z-model.cluster_centers_[model.labels_],axis=1)
threshold=max(.5,float(np.percentile(training_dist,95)))
prototypes=[]
for cluster in range(3):
 ids=np.where(model.labels_==cluster)[0];counts=Counter(train[i]['recorded_action'] for i in ids)
 prototypes.append({'cluster':cluster,'training_ids':[train[i]['id'] for i in ids],'count':len(ids),'mean_features':{f['key']:float(np.mean(X[ids,j])) for j,f in enumerate(schema)},'action_counts':dict(counts),'common_action':counts.most_common(1)[0][0]})
predictions=[]
for i,r in enumerate(heldout):
 row={'id':r['id'],'features':[None if np.isnan(v) else float(v) for v in V[i]],'observed_action_for_evaluation_only':r['recorded_action_for_evaluation_only']}
 if np.isnan(V[i]).any():row.update(status='abstain_missing_features',cluster=None,reason='不以训练均值填补缺失后假装精确匹配')
 else:
  vector=scaler.transform(V[i:i+1])[0];cluster=int(model.predict(vector.reshape(1,-1))[0]);distance=float(np.linalg.norm(vector-model.cluster_centers_[cluster]));proto=prototypes[cluster]
  row.update(cluster=cluster,distance=distance,status='outside_training_distance' if distance>threshold else 'matched_training_pattern',prototype_action=proto['common_action'],prototype_action_matches_record=proto['common_action']==r['recorded_action_for_evaluation_only'] if r['recorded_action_for_evaluation_only'] else None)
 predictions.append(row)
known=[p for p in predictions if p['observed_action_for_evaluation_only'] is not None]
report={'status':'completed','scope':'Controlled refinement: operator defined types/negation handling from source observations, deterministic rules replace LLM extractor; schema and extractor both changed. Not automatically discovered by model and not a clean extractor-only ablation.','schema':schema,'extracted_training':[{'id':r['id'],'features':X[i].tolist()} for i,r in enumerate(train)],'training_only_fitting':True,'prototypes':prototypes,'distance_threshold':threshold,'heldout':predictions,'match_count':sum(p.get('prototype_action_matches_record') is True for p in known),'labelled_count':len(known),'limits':'Templated fictional data with explicit wording. Rules and patterns do not establish real-world management advice, efficacy, causality or broad extraction superiority.'}
(OUT/'schema-control.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'matches':report['match_count'],'labelled':len(known),'heldout_statuses':[{p['id']:p['status']} for p in predictions]},ensure_ascii=False),flush=True)
