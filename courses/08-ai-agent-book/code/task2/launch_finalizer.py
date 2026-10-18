import json
from pathlib import Path
import subprocess
root=Path('/home/simin/projects/task2-memory-rag')
out=root/'outputs/full-60x4-amd-20261015-01'
with (out/'finalizer.log').open('ab') as log:
 proc=subprocess.Popen(['python3','-u',str(root/'finalize_full_acceptance.py')],
   cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'finalizer-process.json').write_text(json.dumps({'pid':proc.pid,'waits_for':json.loads((out/'process.json').read_text())['pid']}))
print('Finalizer started:',proc.pid)
