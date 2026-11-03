import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/simin/projects/task2-memory-rag')
out=root/'outputs/full-60x4-amd-20261015-01'
preflight=json.loads((out/'preflight.json').read_text())
assert preflight['status'] in ('ready','dataset_verified')
available=json.loads((out/'models.json').read_text())
assert preflight['writer_model'] in available and preflight['judge_model'] in available
state=out/'process.json'
if state.exists():
 old=json.loads(state.read_text())
 try:
  os.kill(old['pid'],0)
 except ProcessLookupError:
  pass
 else:
  raise SystemExit('Acceptance process is already running: '+str(old['pid']))
with (out/'campaign.log').open('ab') as log:
 proc=subprocess.Popen(['python3','-u',str(root/'amd_full_acceptance.py')],
    cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
state.write_text(json.dumps({'pid':proc.pid,'started_epoch':time.time(),'log':str(out/'campaign.log'),
 'status':'launched','expected_jobs':240,'minimum_api_calls':1040},indent=2))
print(state.read_text(),flush=True)
