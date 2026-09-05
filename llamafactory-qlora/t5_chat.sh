#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
cd ~/projects/LLaMA-Factory || exit 1
printf "你是谁？\n你由谁开发？\n用一句话解释什么是 QLoRA。\nexit\n" | llamafactory-cli chat ~/projects/llamafactory-practice/config/vm60_chat_qwen2p5_1p5b.yaml 2>&1 | tee ~/projects/llamafactory-practice/evidence/chat_qlora.txt | tail -30
echo CHAT_DONE
