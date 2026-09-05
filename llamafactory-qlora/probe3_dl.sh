#!/usr/bin/env bash
set -x
echo "=== cache contents ==="
ls -la ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/ 2>/dev/null
echo "=== try download (no xet) ==="
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
timeout 300 python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
print('DOWNLOADED_TO', p)
import os
for f in sorted(os.listdir(p)): print(' ', f)
" 2>&1 | tail -15
df -h / | tail -1
echo PROBE3_DONE
