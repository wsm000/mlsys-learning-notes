"""Small alternating repeated comparison; same prompt, tools and fixtures."""
import json
import os
from pathlib import Path
import time
import uuid
from execution_trace import ExecutionTrace

os.environ.update(BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
                  BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1')
from my_deepagent.demo_binding import build_agent
from my_deepagent.settings import get_settings

out = Path('/home/simin/course09-quanty/outputs/model-comparison')
out.mkdir(exist_ok=True)
question = 'Analyze DEMO latest 3 records versus preceding 3 records. Report price return, baseline and recent mean volumes, volume ratio, and explain synthetic-data limitations.'
records = []
for repetition in range(1, 4):
    models = ['DeepSeek-V4-Flash', 'GLM-5.3-Flash']
    if repetition % 2 == 0:
        models.reverse()
    for model in models:
        os.environ['BUB_MODEL'] = 'openai:' + model
        get_settings.cache_clear()
        trace = ExecutionTrace()
        started = time.perf_counter()
        record = {'model': model, 'repetition': repetition, 'input': question}
        try:
            result = build_agent().invoke({'messages': [{'role': 'user', 'content': question}]},
                                         config={'callbacks': [trace], 'run_id': uuid.uuid4()})
            messages = [m.model_dump(mode='json') for m in result['messages']]
            calls = [c for m in messages for c in m.get('tool_calls', [])]
            business = [c for c in calls if c['name'] == 'analyze_price_volume']
            returns = {m['tool_call_id']: m for m in messages if m.get('type') == 'tool'}
            argument_ok = len(business) == 1 and business[0]['args'].get('symbol') == 'DEMO' and str(business[0]['args'].get('window')) == '3'
            strict_type = argument_ok and type(business[0]['args']['window']) is int
            tool_ok = False
            if argument_ok:
                reply = returns.get(business[0]['id'], {})
                if reply.get('status') == 'success':
                    data = json.loads(reply['content'])
                    tool_ok = (data.get('window') == 3 and data.get('price_return_pct') == 2.912621
                               and data.get('baseline_mean_volume') == 1300 and data.get('recent_mean_volume') == 1700)
            answer = messages[-1]['content']
            flat = str(answer).replace(',', '').lower()
            answer_numbers = all(s in flat for s in ('2.91', '1300', '1700', '1.31'))
            synthetic_disclosed = 'synthetic' in flat or '模拟' in flat or '合成' in flat
            record.update(status='completed', messages=messages, calls=calls,
                          checks={'argument_value_correct': argument_ok, 'argument_integer_type': strict_type,
                                  'tool_result_correct': tool_ok, 'answer_expected_numbers': answer_numbers,
                                  'synthetic_disclosed': synthetic_disclosed,
                                  'no_extra_tools': len(calls) == 1},
                          core_pass=argument_ok and tool_ok and answer_numbers and synthetic_disclosed)
        except Exception as error:
            record.update(status='error', core_pass=False, error_type=type(error).__name__,
                          http_status=getattr(error, 'status_code', None))
        record.update(duration_seconds=round(time.perf_counter()-started, 3), spans=trace.export())
        records.append(record)
        (out / f'{model}-{repetition}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        (out / 'runs.json').write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str))
        print(json.dumps({k:v for k,v in record.items() if k not in {'messages','spans','input','calls'}}, ensure_ascii=False), flush=True)
