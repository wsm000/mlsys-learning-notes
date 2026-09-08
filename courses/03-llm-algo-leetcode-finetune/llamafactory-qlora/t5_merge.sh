#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
cd ~/projects/LLaMA-Factory || exit 1
llamafactory-cli export ~/projects/llamafactory-practice/config/vm60_merge_qwen2p5_1p5b.yaml 2>&1 | tee ~/projects/llamafactory-practice/evidence/export.log | tail -12
ls -lh ~/projects/llamafactory-practice/models/qwen2p5-1p5b-merged/ 2>/dev/null | head -10
df -h / | tail -1
echo MERGE_DONE
