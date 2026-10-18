"""From-scratch BM25 with transparent tokenization and per-term receipts."""
import json,math,re
from collections import Counter,defaultdict
from pathlib import Path

OUT=Path(__file__).resolve().parent/'outputs/experiment05-equivalent-20261018'
OUT.mkdir(parents=True,exist_ok=True)
# Explicit word boundaries isolate scoring from Chinese tokenizer quality.
documents=[
 {'id':'D1','text':'蜂群出现王台，需要检查分蜂征兆。','segmented':'蜂群 王台 分蜂 检查'},
 {'id':'D2','text':'分蜂管理记录：蜂群有分蜂征兆，检查王台。','segmented':'分蜂 蜂群 分蜂 检查 王台'},
 {'id':'D3','text':'缺蜜期检查蜂群，使用糖浆饲喂。','segmented':'缺蜜 检查 蜂群 糖浆 饲喂'},
 {'id':'D4','text':'检查蜂群越冬饲料，做好保温。','segmented':'检查 蜂群 越冬 饲料 保温'},
 {'id':'D5','text':'检查蜂群中的王台与蜂王产卵。','segmented':'检查 蜂群 王台 蜂王 产卵'},
 {'id':'D6','text':'检查蜂群，监测和防治蜂螨。','segmented':'检查 蜂群 监测 防治 蜂螨'},
]
queries=[{'text':'分蜂 王台','relevant':['D1','D2'],'purpose':'exact keywords'},
 {'text':'分家','relevant':['D1','D2'],'purpose':'synonym without shared tokens'},
 {'text':'分蜂','relevant':['D1','D2'],'purpose':'query expansion controlled by hand, not automatic'},
 {'text':'检查','relevant':[d['id'] for d in documents],'purpose':'ubiquitous low-information term'}]
def tokenize(text):return re.findall(r'\S+',text)
index=defaultdict(dict);lengths={}
for doc in documents:
 doc['tokens']=tokenize(doc['segmented']);lengths[doc['id']]=len(doc['tokens'])
 for word,tf in Counter(doc['tokens']).items():index[word][doc['id']]=tf
N=len(documents);avgdl=sum(lengths.values())/N;k1=1.5;b=.75
def term_score(word,doc_id):
 tf=index.get(word,{}).get(doc_id,0);df=len(index.get(word,{}))
 idf=math.log(1+(N-df+.5)/(df+.5))
 normalizer=k1*(1-b+b*lengths[doc_id]/avgdl)
 tf_factor=tf*(k1+1)/(tf+normalizer)
 return {'term':word,'tf':tf,'df':df,'idf':idf,'document_length':lengths[doc_id],'length_normalizer':normalizer,'tf_factor':tf_factor,'contribution':idf*tf_factor}
results=[]
for query in queries:
 tokens=tokenize(query['text']);candidates=sorted(set(d for t in tokens for d in index.get(t,{})))
 rows=[]
 for doc_id in candidates:
  terms=[term_score(t,doc_id) for t in tokens]
  rows.append({'document_id':doc_id,'score':sum(t['contribution'] for t in terms),'terms':terms})
 rows.sort(key=lambda r:(-r['score'],r['document_id']))
 results.append({**query,'tokens':tokens,'candidate_ids':candidates,'ranking':rows,'recall_at_3':len(set(r['document_id'] for r in rows[:3])&set(query['relevant']))/len(query['relevant'])})
# Independent arithmetic for the first document and the keyword 分蜂.
expected=math.log(1+(6-2+.5)/(2+.5))*(1*2.5)/(1+1.5*(.25+.75*4/(29/6)))
observed=term_score('分蜂','D1')['contribution']
assert abs(expected-observed)<1e-12
tf_saturation=[{'tf':tf,'factor_at_average_length':tf*(k1+1)/(tf+k1)} for tf in (1,2,4,8)]
length_effect=[{'length':dl,'factor_with_tf_1':2.5/(1+k1*(1-b+b*dl/avgdl))} for dl in (4,8,16)]
report={'status':'completed','scope':'mechanism-equivalent teaching experiment; hand-authored bee corpus, explicit segmentation, no third-party retrieval engine','parameters':{'N':N,'average_length':avgdl,'k1':k1,'b':b},'formula':'sum ln(1+(N-df+0.5)/(df+0.5)) * tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))','idf_variant_note':'Uses positive log1p IDF; upstream hand-check uses raw RSJ IDF. Values are not numerically interchangeable.','documents':documents,'inverted_index':dict(index),'queries':results,'hand_check':{'expected':expected,'observed':observed,'absolute_error':abs(expected-observed)},'tf_saturation':tf_saturation,'length_effect':length_effect}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'parameters':report['parameters'],'queries':[{'query':r['text'],'ranking':[(x['document_id'],round(x['score'],4)) for x in r['ranking']],'recall_at_3':r['recall_at_3']} for r in results],'hand_check':report['hand_check']},ensure_ascii=False),flush=True)
