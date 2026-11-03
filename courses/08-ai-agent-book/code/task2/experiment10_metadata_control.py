"""Replay saved plain dense scores with explicit source-metadata constraints."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/experiment10-equivalent-20261018'
report=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
byid={c['id']:c for c in report['chunks']};rows=[]
for row in report['results']:
 if row['index']!='plain':continue
 question=row['question']
 farm=re.search(r'(青禾蜂场|白石蜂场)',question).group(0)
 year=re.search(r'20\d{2}',question).group(0)
 season=re.search(r'春季|秋季',question).group(0)
 candidates=[cid for cid,c in byid.items() if all(term in c['title'] for term in (farm,year,season))]
 ranked=sorted(candidates,key=lambda cid:(-row['dense_scores'][cid],cid))
 rows.append({'query_id':row['id'],'question':question,'constraints':{'farm':farm,'year':year,'season':season},'candidate_ids':candidates,'ranking':ranked,'gold':row['gold'],'hit_at_1':int(bool(ranked) and ranked[0] in row['gold'])})
control={'status':'completed','method':'Local deterministic replay of saved plain dense scores; exact title-derived metadata constraints explicitly present in queries, then original dense score order. No additional encoder or model calls.','prefixes_equal_titles':all(c['generated_prefix']==c['title'] for c in report['chunks']),'hits_at_1':sum(r['hit_at_1'] for r in rows),'queries':len(rows),'results':rows,'limits':'All questions explicitly specify these three fields; this is not an automatic temporal-intent parser or a general metadata benchmark.'}
(OUT/'metadata-filter-control.json').write_text(json.dumps(control,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(control,ensure_ascii=False))
