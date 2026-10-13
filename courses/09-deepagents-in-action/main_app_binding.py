"""Course deepagents/default binding adapted with Quanty's tool."""
from agentseek_langchain import messages_spec
from deepagents import ProviderProfile, create_deep_agent, register_provider_profile
from langchain.agents.middleware import TodoListMiddleware
from .settings import get_settings
from .quanty_tools import analyze_price_volume
import json
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler


class EvidenceCallback(BaseCallbackHandler):
    def record(self, event, **data):
        path = Path('/home/simin/course09-quanty/outputs/main-tool-events.jsonl')
        with path.open('a') as stream:
            stream.write(json.dumps({'event': event, **data}, ensure_ascii=False, default=str) + '\n')

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.record('tool_start', name=serialized.get('name'), input=input_str, run_id=str(run_id))

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.record('tool_end', output=output.model_dump(mode='json') if hasattr(output, 'model_dump') else str(output), run_id=str(run_id))

    def on_llm_end(self, response, **kwargs):
        for group in response.generations:
            for generation in group:
                message = getattr(generation, 'message', None)
                if message is not None and getattr(message, 'tool_calls', None):
                    self.record('model_tool_requests', calls=message.tool_calls)


def build_agent(backend=None):
    settings = get_settings()
    settings.apply_openai_env_bridge()
    register_provider_profile('openai', ProviderProfile(init_kwargs={'use_responses_api': False, 'timeout': 60, 'max_retries': 0}))
    return create_deep_agent(
        model=settings.require_model(), tools=[analyze_price_volume], backend=backend,
        middleware=[TodoListMiddleware()],
        system_prompt='You are Quanty. For supported price-volume questions call analyze_price_volume. Only 10 DEMO synthetic records exist. Tools support window 1-5. If user requests a larger window N, explain that 2N records are needed and ask whether they accept window=5; do not call a substitute or modify code before explicit agreement. Explain numerical evidence and limitations. Simple queries need no planning. Never invent data.').with_config(callbacks=[EvidenceCallback()])


def build_spec():
    return messages_spec(build_agent(), include_agents_md=True)
