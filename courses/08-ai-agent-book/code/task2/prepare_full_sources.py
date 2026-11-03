import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path('/home/simin/projects/task2-memory-rag')
OUT=ROOT/'outputs/full-60x4-amd-20261015-01'
preflight=json.loads((OUT/'preflight.json').read_text())
for item in preflight['files']:
 path=Path(item['path'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 target=OUT/'source'/path.relative_to(ROOT)
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(path,target)
for name in ['amd_full_acceptance.py','audit_full_acceptance.py','finalize_full_acceptance.py','launch_acceptance.py']:
 target=OUT/'source'/name
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(ROOT/name,target)
print('Source files bundled:',len(preflight['files'])+4)
