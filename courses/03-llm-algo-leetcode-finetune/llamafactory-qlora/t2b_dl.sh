#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
python -c "
from huggingface_hub import snapshot_download
import time
for mid in ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct']:
    t0 = time.time()
    p = snapshot_download(mid)
    print(f'OK {mid} -> {p} took {time.time()-t0:.0f}s', flush=True)
import os
for mid in ['Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct']:
    for root, dirs, files in os.walk(os.path.expanduser('~/.cache/huggingface/hub')):
        if mid in root and root.endswith(os.path.basename(mid)) is False: continue
print('DL_ALL_DONE')
"
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct 2>/dev/null
df -h / | tail -1
echo T2B_DONE
