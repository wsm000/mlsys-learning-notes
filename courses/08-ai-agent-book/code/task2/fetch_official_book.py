"""Download a pinned subset, verifying proxy bytes against GitHub's Git blob IDs."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT=Path('/home/simin/projects/task2-memory-rag/official-book')
API='https://api.github.com/repos/bojieli/ai-agent-book'
HEADERS={'User-Agent':'task2-memory-rag-learning','Accept':'application/vnd.github+json'}
def get(url):
 for attempt in range(3):
  try:
   with urlopen(Request(url,headers=HEADERS),timeout=35) as response: return response.read()
  except Exception:
   if attempt==2: raise
   time.sleep(2*(attempt+1))
ROOT.mkdir(exist_ok=True)
commit=json.loads(get(API+'/commits/main'))['sha']
tree=json.loads(get(API+'/git/trees/'+commit+'?recursive=1'))
assert not tree.get('truncated')
prefixes=('chapter3/user-memory/','chapter3/user-memory-evaluation/')
specific={'chapter3/memory_rag_campaign.py','chapter3/experiment_utils.py','chapter3/EXPERIMENT_LEDGER.md','chapter3/README.md','book/chapter3.md','pyproject.toml','AGENTS.md'}
selected=[f for f in tree['tree'] if f['type']=='blob' and (f['path'] in specific or any(f['path'].startswith(p) for p in prefixes)) and not '/validation/runs/' in f['path']]
def fetch(item):
 target=ROOT/item['path']
 if target.exists():
  content=target.read_bytes()
  if hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==item['sha']: return item['path']
 url='https://ghfast.top/https://raw.githubusercontent.com/bojieli/ai-agent-book/'+commit+'/'+item['path']
 content=get(url)
 assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==item['sha'],item['path']
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(content)
 print('FETCH '+item['path'],flush=True)
 return item['path']
(ROOT/'source-manifest.json').write_text(json.dumps({'repository':API,'commit':commit,'files':selected},indent=2))
print('PINNED '+commit+' FILES '+str(len(selected)),flush=True)
with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(fetch,selected))
print('OFFICIAL_SUBSET_VERIFIED',flush=True)
