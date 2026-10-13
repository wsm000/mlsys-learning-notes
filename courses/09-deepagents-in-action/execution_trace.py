"""Local LangChain callback spans; no external tracing service required."""
from datetime import datetime, timezone
import threading
import time
from langchain_core.callbacks import BaseCallbackHandler


class ExecutionTrace(BaseCallbackHandler):
    def __init__(self):
        self.spans = {}
        self.lock = threading.Lock()

    def start(self, kind, run_id, parent_run_id, name, **data):
        with self.lock:
            self.spans[str(run_id)] = {
                'id': str(run_id), 'parent_id': str(parent_run_id) if parent_run_id else None,
                'kind': kind, 'name': name, 'started_at': datetime.now(timezone.utc).isoformat(),
                '_clock': time.perf_counter(), 'status': 'running', **data}

    def finish(self, run_id, error=None, **data):
        with self.lock:
            span = self.spans.get(str(run_id))
            if span is None:
                return
            span.update(ended_at=datetime.now(timezone.utc).isoformat(),
                        duration_ms=round((time.perf_counter() - span['_clock']) * 1000, 3),
                        status='error' if error else 'success', **data)
            if error:
                span['error_type'] = type(error).__name__

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self.start('chain', run_id, parent_run_id, kwargs.get('name') or (serialized or {}).get('name', 'chain'))

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.finish(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self.finish(run_id, error=error)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self.start('model', run_id, parent_run_id, serialized.get('name', 'chat_model'),
                   input_messages=[[m.model_dump(mode='json') for m in group] for group in messages])

    def on_llm_end(self, response, *, run_id, **kwargs):
        messages = [g.message.model_dump(mode='json') for group in response.generations for g in group if hasattr(g, 'message')]
        self.finish(run_id, output_messages=messages)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.finish(run_id, error=error)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self.start('tool', run_id, parent_run_id, serialized.get('name'), input=input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.finish(run_id, output=output.model_dump(mode='json') if hasattr(output, 'model_dump') else str(output))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.finish(run_id, error=error)

    def export(self):
        return [{k: v for k,v in s.items() if k != '_clock'} for s in self.spans.values()]
