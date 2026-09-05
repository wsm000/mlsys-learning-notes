#!/usr/bin/env bash
# 用法: run_train.sh <config> <memlog> <trainlog>
set -o pipefail
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
CFG="$1"; EV=~/projects/llamafactory-practice/evidence
mkdir -p "$EV"
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1 > "$EV/$2" 2>&1 &
MPID=$!
cd ~/projects/LLaMA-Factory || exit 1
llamafactory-cli train "$CFG" 2>&1 | tee "$EV/$3" | tail -25
RC=$?
kill $MPID 2>/dev/null
sleep 1
echo "=== PEAK (MiB) for $2: $(sort -n "$EV/$2" 2>/dev/null | tail -1) ==="
df -h / | tail -1
exit $RC
