"""Replay precise card-field queries against saved original history chunks."""
import json,re,math,calendar
from pathlib import Path
from datetime import date
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/experiment11-equivalent-20261018'
cards=json.loads((OUT/'cards-provenance.json').read_text(encoding='utf-8'))['compact_retained']
chunks=json.loads((OUT/'chunks-original.json').read_text(encoding='utf-8'))
def words(text):return re.findall(r'[a-z]+|\d+',text.lower())
index=defaultdict(dict);lengths={}
for c in chunks:
 ts=words(c['text']);lengths[c['id']]=len(ts)
 for term,tf in Counter(ts).items():index[term][c['id']]=tf
avg=sum(lengths.values())/len(chunks)
fields={k:v for c in cards for k,v in c['facts'].items()}
rows=[]
for key in ('passport_expiration_date','return_date','passport_update_deadline','business_trip_coverage'):
 if key not in fields:continue
 value=fields[key];query=key.replace('_',' ')+' '+str(value)
 if re.fullmatch(r'\d{4}-\d{2}-\d{2}',str(value)):
  d=date.fromisoformat(value);query=key.replace('_',' ')+f' {calendar.month_name[d.month]} {d.day} {d.year}'
 scores=defaultdict(float)
 for term in words(query):
  hits=index.get(term,{});idf=math.log(1+(len(chunks)-len(hits)+.5)/(len(hits)+.5))
  for cid,tf in hits.items():scores[cid]+=idf*tf*2.5/(tf+1.5*(.25+.75*lengths[cid]/avg))
 ranked=sorted(scores,key=lambda cid:(-scores[cid],cid))[:2]
 rows.append({'field':key,'card_value':value,'query':query,'hits':[next(c for c in chunks if c['id']==cid) for cid in ranked]})
gap=(date.fromisoformat(fields['passport_expiration_date'])-date.fromisoformat(fields['return_date'])).days
report={'status':'completed','method':'Controlled exact-field BM25 queries derived from retained card field names and values; original chunk bodies only. Saved artifacts replayed locally, not autonomous planner output.','date_calculation':{'expiration_date_from_card':fields['passport_expiration_date'],'return_date_from_card':fields['return_date'],'days_between':gap,'meaning':'Calendar interval conditional on card dates. Does not establish any entry/boarding requirement or that renewal is completed.'},'field_queries':rows,'limits':'Retrieved quotes support human verification; no automatic semantic entailment or policy validation is claimed.'}
(OUT/'field-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'gap_days':gap,'hits':[{r['field']:[h['id'] for h in r['hits']]} for r in rows]},ensure_ascii=False))
