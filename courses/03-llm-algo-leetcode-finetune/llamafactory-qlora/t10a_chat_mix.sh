#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
cd ~/projects/LLaMA-Factory || exit 1
EV=~/projects/llamafactory-practice/evidence
PROMPTS="你是谁？\n你由谁开发？\n列出三条学习英语的建议。\n用一句话解释什么是微调。\nexit"
for m in mix10 mix30 mix50; do
  cat > /tmp/chat_$m.yaml <<EOF
model_name_or_path: /home/simin/local_models/allenai/OLMoE-1B-7B-0125
trust_remote_code: true
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
adapter_name_or_path: /home/simin/projects/llamafactory-practice/saves/olmoe-$m
finetuning_type: lora
template: default
temperature: 0.7
top_p: 0.8
max_new_tokens: 300
EOF
  echo "########## MIX $m ##########" > $EV/chat_mix$m.txt
  printf "$PROMPTS" | llamafactory-cli chat /tmp/chat_$m.yaml 2>>/dev/null >> $EV/chat_mix$m.txt
  echo "$m done"
done
echo CHATMIX_DONE
