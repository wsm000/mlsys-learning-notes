import os
"""Observe real Deep Agents compaction with Todo state; instrumentation delegates unchanged."""
import csv,hashlib,inspect,json,time,traceback,importlib.metadata as md
from pathlib import Path
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_openai import ChatOpenAI
from quanty_tools import DATA,analyze_price_volume

OUT=Path(os.environ.get('TASK3_OUTPUT_DIR', str(Path(__file__).resolve().parent.parent/'runs')))/('task3-summary-'+time.strftime('%Y%m%d-%H%M%S'))
ROOT=OUT/'workspace';(ROOT/'input').mkdir(parents=True)
fixture=DATA.read_bytes();(ROOT/'input/demo.csv').write_bytes(fixture)
def dump(x):
    if hasattr(x,'model_dump'):return x.model_dump(mode='json')
    return str(x)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=dump))
def serialized(messages):return [dump(m) for m in messages]
def chars(messages):return sum(len(str(m.content)) for m in messages)

events=[];summaries=[]
class ObservedSummary(SummarizationMiddleware):
    @property
    def name(self):return 'SummarizationMiddleware'
    def _create_summary(self,messages):
        index=len(summaries)
        summary=super()._create_summary(messages)
        entry={'index':index,'source_messages':serialized(messages),'summary':summary,'source_chars':chars(messages),'summary_chars':len(str(summary))}
        summaries.append(entry);save('summaries.json',summaries)
        print('SUMMARY '+json.dumps({k:v for k,v in entry.items() if k not in ['source_messages','summary']}),flush=True)
        return summary
    def wrap_model_call(self,request,handler):
        effective=self._get_effective_messages(request)
        before={'raw_state_message_count':len(request.state.get('messages',[])),'effective_message_count':len(effective),'effective_chars':chars(effective),'todos':request.state.get('todos',[]),'messages':serialized(effective)}
        n=len(summaries)
        def observe(modified):
            after={'message_count':len(modified.messages),'chars':chars(modified.messages),'messages':serialized(modified.messages),'todos':modified.state.get('todos',[])}
            event={'before':before,'after':after,'new_summary_count':len(summaries)-n,'todos_unchanged':before['todos']==after['todos']}
            events.append(event);save('model_contexts.json',events)
            return handler(modified)
        return super().wrap_model_call(request,observe)

def read_research_packet(packet:int)->str:
    """Read a fixed synthetic research packet numbered 1 or 2; includes long audit notes."""
    if packet not in (1,2):raise ValueError('packet must be 1 or 2')
    head=f'PACKET {packet}: DEMO synthetic data only. Acceptance marker: SUMTODO-42. Read both packets, compute windows 3 and 5, save and read back report.\n'
    notes=''.join(f'Audit note {i:03}: This repeated synthetic documentation is context-load material, contains no new price observations, no extra task, and no market forecast.\n' for i in range(70))
    return head+notes+('Reference CSV:\n'+fixture.decode() if packet==1 else 'Use computed tool evidence for all numbers. The output must preserve marker SUMTODO-42.\n')

PROMPT='''完成带长资料的多步骤任务，输出 /reports/comparison.json。
先用 write_todos 建立五项清单（全部 pending），后续及时更新完整清单：1 读取 research packet 1；2 读取 research packet 2；3 计算 window=3 和 window=5；4 写报告；5 读回核验。
先单独完成计划调用，再依次调用 read_research_packet(1)、read_research_packet(2)，两个 packet 必须分两次模型回合读取。资料中的重复审计说明只用于长上下文观察，不增加数据。
随后调用 analyze_price_volume 计算两个窗口，用工具证据写合法 JSON：{"marker":"SUMTODO-42","source":"synthetic","windows":[{"window":3,"price_return_pct":数值,"volume_ratio":数值},{"window":5,"price_return_pct":数值,"volume_ratio":数值}],"limitations":"模拟资料，仅描述性比较，不支持因果或预测"}。
没有完成的工作不能标记 completed。最后用 read_file 读回报告核对后再结束并完成清单。不要 execute 或委派。若上下文被摘要，继续剩余工作，不重复已经完成的读取。'''
model=ChatOpenAI(model='DeepSeek-V4-Flash',temperature=0,max_tokens=4096,timeout=90,max_retries=0,use_responses_api=False,api_key=os.environ['QUANTY_API_KEY'],base_url=os.environ.get('QUANTY_BASE_URL','https://developer.amd.com.cn/radeon/api/v1'))
backend=FilesystemBackend(root_dir=ROOT,virtual_mode=True)
summary=ObservedSummary(model=model,backend=backend,trigger=('messages',10),keep=('messages',4),trim_tokens_to_summarize=16000)
save('manifest.json',{'model':'DeepSeek-V4-Flash','temperature':0,'max_tokens':4096,'versions':{k:md.version(k) for k in ['deepagents','langchain','langchain-openai']},'trigger':['messages',10],'keep':['messages',4],'trim_tokens_to_summarize':16000,'prompt':PROMPT,'fixture_sha256':hashlib.sha256(fixture).hexdigest(),'instrumentation':'Subclass name matches default middleware for replacement; wrap_model_call and _create_summary delegate to super unchanged. No synthetic history, manually injected todos, or manual summaries.','packet_chars':{str(i):len(read_research_packet(i)) for i in [1,2]}})
(OUT/'installed_summary_source.py').write_text(inspect.getsource(SummarizationMiddleware))
print('OUTPUT '+str(OUT),flush=True)
record={'messages':[],'todo_snapshots':[],'summary_state_events':[]}
try:
    agent=create_deep_agent(model=model,tools=[read_research_packet,analyze_price_volume],backend=backend,middleware=[summary,TodoListMiddleware()],system_prompt='Use the plan and tool evidence to complete the task. Preserve the acceptance marker and unfinished work across context summaries. Do not delegate or execute commands.')
    with (OUT/'updates.jsonl').open('w') as log:
        for update in agent.stream({'messages':[{'role':'user','content':PROMPT}]},config={'recursion_limit':100},stream_mode='updates'):
            log.write(json.dumps(update,ensure_ascii=False,default=dump)+'\n');log.flush()
            for node,delta in update.items():
                if not isinstance(delta,dict):continue
                if 'todos' in delta:record['todo_snapshots'].append(delta['todos'])
                if '_summarization_event' in delta:record['summary_state_events'].append(delta['_summarization_event'])
                for m in delta.get('messages',[]):
                    item=dump(m);record['messages'].append(item)
                    if isinstance(item,dict) and item.get('tool_calls'):print('TOOLS '+','.join(c['name'] for c in item['tool_calls']),flush=True)
    report=json.loads((ROOT/'reports/comparison.json').read_text());record['report']=report
    rows=list(csv.DictReader(fixture.decode().splitlines()));expected=[]
    for w in [3,5]:
        b,r=rows[-2*w:-w],rows[-w:]
        expected.append({'window':w,'price_return_pct':round((float(r[-1]['close'])/float(b[-1]['close'])-1)*100,6),'volume_ratio':sum(float(x['volume']) for x in r)/sum(float(x['volume']) for x in b)})
    calls=[c for m in record['messages'] if isinstance(m,dict) for c in m.get('tool_calls',[])]
    compression=[e for e in events if e['new_summary_count']]
    checks={'summary_generated':bool(summaries),'effective_context_reduced':any(e['after']['chars']<e['before']['effective_chars'] for e in compression),'todos_preserved_at_compaction':bool(compression) and all(e['todos_unchanged'] and bool(e['before']['todos']) for e in compression),'compaction_while_unfinished':any(any(t['status']!='completed' for t in e['before']['todos']) for e in compression),'final_todos_completed':bool(record['todo_snapshots']) and all(t['status']=='completed' for t in record['todo_snapshots'][-1]),'each_packet_read_once':all(sum(c['name']=='read_research_packet' and c['args'].get('packet')==p for c in calls)==1 for p in [1,2]),'report_read_back':any(c['name']=='read_file' and c['args'].get('file_path')=='/reports/comparison.json' for c in calls),'marker_preserved':report.get('marker')=='SUMTODO-42','numeric_fields_correct':len(report.get('windows',[]))==2 and all(any(all(type(r.get(k)) in (float,int) and abs(r[k]-v)<1e-6 for k,v in e.items()) for r in report['windows']) for e in expected)}
    record.update(checks=checks,passed=all(checks.values()),expected=expected,summary_count=len(summaries))
except Exception as exc:record.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
finally:
    save('result.json',record);print('RESULT '+json.dumps({k:v for k,v in record.items() if k not in ['messages','todo_snapshots','summary_state_events','report','traceback']},ensure_ascii=False),flush=True)
