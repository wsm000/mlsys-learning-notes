import os
from pathlib import Path
import subprocess

root = Path('/home/simin/course09-quanty/main-app/my_deepagent')
env = dict(os.environ)
env.update(BUB_MODEL='openai:DeepSeek-V4-Flash',
           BUB_API_KEY=Path('/home/simin/codex-task1-20261013/.mimo-key').read_text().strip(),
           BUB_API_BASE='https://developer.amd.com.cn/radeon/api/v1',
           BUB_LANGCHAIN_SPEC='my_deepagent.demo_binding:build_spec', BUB_AG_UI_PORT='18088')
env['PATH'] = '/home/simin/.local/bin:' + env.get('PATH', '')
log = Path('/home/simin/course09-quanty/outputs/main-gateway.log').open('a')
process = subprocess.Popen([str(root / '.venv/bin/bub'), 'gateway', '--enable-channel', 'ag-ui'],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
Path('/home/simin/course09-quanty/outputs/main-gateway.pid').write_text(str(process.pid))
print('Gateway PID:', process.pid)
