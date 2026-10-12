import dataclasses
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ['AI_AGENT_BOOK_DIR']).resolve()
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / 'chapter1/context'))
KEY = os.environ['AMD_API_KEY']
from agentbook.providers.registry import PROVIDERS
from agentbook.providers.models import Provider
PROVIDERS['amd_mimo'] = Provider(name='amd_mimo', base_url='https://developer.amd.com.cn/radeon/api/v1', default_model='MiMo-V2.6-Flash')
from agent import ContextAwareAgent, ContextMode

TASK = '请调用计算器计算 (125 + 375) * 8，并给出结果。'
OUT = Path(os.environ.get('AGENT_RESULTS_DIR', str(ROOT / 'evidence/local-context-run')))
OUT.mkdir(parents=True, exist_ok=True)

def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k.lower() not in {'metadata', 'authorization', 'api_key', 'apikeyhash'}}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return value.replace(KEY, '[REDACTED]')
    return value

summary = {'source': 'https://github.com/bojieli/ai-agent-book/tree/main/chapter1/context',
           'adaptation': 'Original ContextAwareAgent; runtime AMD provider registration only',
           'python': platform.python_version(), 'task': TASK, 'expected_result': 4000, 'runs': []}
for mode in [ContextMode.FULL, ContextMode.NO_TOOL_RESULTS]:
    print('\n=== CONTEXT MODE:', mode.value, '===', flush=True)
    try:
        agent = ContextAwareAgent(KEY, context_mode=mode, provider='amd_mimo', model='MiMo-V2.6-Flash', verbose=True)
        agent.client = agent.client.with_options(max_retries=0, timeout=90)
        result = agent.execute_task(TASK, max_iterations=5)
        trajectory = dataclasses.asdict(result.pop('trajectory'))
        trajectory['context_mode'] = mode.value
        record = scrub({**result, 'trajectory': trajectory})
        print('Iterations:', result.get('iterations'))
        print('Completed:', result.get('completed'))
        print('Tool calls:', len(trajectory['tool_calls']))
        print('Final answer:', result.get('final_answer'))
        print('Error:', result.get('error'))
        for call in trajectory['tool_calls']:
            print('TOOL:', call['tool_name'], call['arguments'], '=>', call['result'])
        summary['runs'].append({'mode': mode.value, **{k: record.get(k) for k in ['iterations', 'completed', 'final_answer', 'error']}, 'tool_calls': len(trajectory['tool_calls'])})
    except Exception as exc:
        record = {'error': str(exc).replace(KEY, '[REDACTED]')}
        summary['runs'].append({'mode': mode.value, **record})
        print('Error:', record['error'])
    (OUT / (mode.value + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nCONTEXT_COMPARISON_RECORDED - inspect both runs; completion does not prove correctness.')
