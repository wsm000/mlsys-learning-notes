#!/usr/bin/env bash
set -x
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
for m in mix10 mix30 mix50; do
  rm -rf ~/projects/llamafactory-practice/saves/olmoe-$m
  echo "########## RUN $m ##########"
  bash /tmp/run_train.sh /home/simin/projects/llamafactory-practice/config/olmoe_$m.yaml mem_olmoe_$m.log train_olmoe_$m.log 2>&1 | tail -12
done
echo ALL_MIX_DONE
