import os
"""Paired, clean-state real-model ablation; no fabricated traces."""
import csv, hashlib, importlib.metadata as md, json, time, traceback
from collections import Counter
from pathlib import Path
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_openai import ChatOpenAI
from quanty_tools import DATA, analyze_price_volume

OUT=Path(os.environ.get('TASK3_OUTPUT_DIR', str(Path(__file__).resolve().parent.parent/'runs')))/('task3-ab-'+time.strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(parents=True)
fixture=DATA.read_bytes()
SYSTEM='You are Quanty. Use tools for numerical claims. Only DEMO synthetic teaching data exists. Do not use execute or delegate. Complete all requested steps and verify the saved artifact.'
PROMPT='''完成多步骤分析：读取 /input/demo.csv，分别调用 analyze_price_volume(window=3) 和 analyze_price_volume(window=5)，比较结果，写入 /reports/comparison.json，最后读回核对。
报告为合法 JSON：{"source":"synthetic","windows":[{"window":3,"price_return_pct":数值,"volume_ratio":数值,"baseline_mean_volume":数值,"recent_mean_volume":数值},{"window":5,"price_return_pct":数值,"volume_ratio":数值,"baseline_mean_volume":数值,"recent_mean_volume":数值}],"comparison":"中文比较收益率、量比及均量差异并引用数值","limitations":"中文说明模拟资料、非因果、不能预测未来"}。
保持工具数值精度；比较大小必须符合数值，不加入资料不支持的价格路径解释。文件仅使用内置文件工具，不使用 execute 或委派。完成全部操作再结束。'''
rows=list(csv.DictReader(fixture.decode().splitlines()))
expected=[]
for w in (3,5):
    b,r=rows[-2*w:-w],rows[-w:]
    bv=sum(float(x['volume']) for x in b)/w;rv=sum(float(x['volume']) for x in r)/w
    expected.append(dict(window=w,price_return_pct=round((float(r[-1]['close'])/float(b[-1]['close'])-1)*100,6),volume_ratio=rv/bv,baseline_mean_volume=bv,recent_mean_volume=rv))
manifest=dict(model='DeepSeek-V4-Flash',temperature=0,max_tokens=4096,timeout=90,max_retries=0,system_prompt=SYSTEM,user_prompt=PROMPT,fixture_sha256=hashlib.sha256(fixture).hexdigest(),versions={k:md.version(k) for k in ['deepagents','langchain','langchain-openai']},order=['off','on'],expected=expected,design='One paired exploratory run. Middleware adds its tool, state and system guidance. No checkpointer; independent filesystem roots. Default Deep Agents harness kept equal. No claim of seed-level determinism.')
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('OUTPUT '+str(OUT),flush=True)
for arm in ('off','on'):
    folder=OUT/arm;root=folder/'workspace';(root/'input').mkdir(parents=True)
    (root/'input/demo.csv').write_bytes(fixture)
    record=dict(arm=arm,messages=[],todo_snapshots=[])
    started=time.monotonic()
    try:
        model=ChatOpenAI(model=manifest['model'],temperature=0,max_tokens=4096,timeout=90,max_retries=0,use_responses_api=False,api_key=os.environ['QUANTY_API_KEY'],base_url=os.environ.get('QUANTY_BASE_URL','https://developer.amd.com.cn/radeon/api/v1'))
        agent=create_deep_agent(model=model,tools=[analyze_price_volume],backend=FilesystemBackend(root_dir=root,virtual_mode=True),middleware=[TodoListMiddleware()] if arm=='on' else [],system_prompt=SYSTEM)
        with (folder/'updates.jsonl').open('w') as stream:
            for update in agent.stream({'messages':[{'role':'user','content':PROMPT}]},config={'recursion_limit':100},stream_mode='updates'):
                stream.write(json.dumps(update,ensure_ascii=False,default=lambda x:x.model_dump(mode='json') if hasattr(x,'model_dump') else str(x))+'\n');stream.flush()
                for node,delta in update.items():
                    if not isinstance(delta,dict):continue
                    if 'todos' in delta:record['todo_snapshots'].append(delta['todos'])
                    for m in delta.get('messages',[]):
                        item=m.model_dump(mode='json') if hasattr(m,'model_dump') else m
                        record['messages'].append(item)
                        if item.get('tool_calls'):print(arm+' TOOLS '+','.join(c['name'] for c in item['tool_calls']),flush=True)
        report=json.loads((root/'reports/comparison.json').read_text());record['report']=report
        calls=[c for m in record['messages'] for c in m.get('tool_calls',[])]
        def called(name,path):return any(c['name']==name and c['args'].get('file_path')==path for c in calls)
        numeric=all(len([r for r in report.get('windows',[]) if r.get('window')==e['window'] and all(type(r.get(k)) in (int,float) and abs(r[k]-e[k])<1e-6 for k in e)])==1 for e in expected) and len(report.get('windows',[]))==2
        checks=dict(fixture_read=called('read_file','/input/demo.csv'),both_computations={c['args'].get('window') for c in calls if c['name']=='analyze_price_volume'}=={3,5},report_written=called('write_file','/reports/comparison.json'),report_read_back=called('read_file','/reports/comparison.json'),numeric_fields=numeric,synthetic_label=report.get('source')=='synthetic',explanation_present=all(isinstance(report.get(k),str) and len(report[k])>15 for k in ['comparison','limitations']))
        record.update(mechanical_checks=checks,mechanical_pass=all(checks.values()),call_counts=dict(Counter(c['name'] for c in calls)),tool_call_count=len(calls),model_call_count=sum(m.get('type')=='ai' for m in record['messages']),semantic_review='pending independent review')
    except Exception as exc:record.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),mechanical_pass=False)
    finally:
        record['elapsed_seconds']=round(time.monotonic()-started,3)
        (folder/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        print(arm+' RESULT '+json.dumps({k:v for k,v in record.items() if k not in ['messages','todo_snapshots','report','traceback']},ensure_ascii=False),flush=True)
(OUT/'completed.json').write_text(json.dumps({'arms':['off','on'],'finished':True}))
