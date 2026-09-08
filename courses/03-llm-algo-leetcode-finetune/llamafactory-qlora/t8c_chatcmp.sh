#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
cd ~/projects/LLaMA-Factory || exit 1
EV=~/projects/llamafactory-practice/evidence

# base 模型 (无适配器)
cat > /tmp/olmoe_base_chat.yaml <<'EOF'
model_name_or_path: /home/simin/local_models/allenai/OLMoE-1B-7B-0125
trust_remote_code: true
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
template: default
temperature: 0.7
top_p: 0.8
max_new_tokens: 300
EOF

# 微调后 (alpaca+identity 适配器)
sed "s|^EOF||" /dev/null; cp /tmp/olmoe_base_chat.yaml /tmp/olmoe_alpaca_chat.yaml
cat >> /tmp/olmoe_alpaca_chat.yaml <<'EOF'
adapter_name_or_path: /home/simin/projects/llamafactory-practice/saves/olmoe-alpaca
finetuning_type: lora
EOF

PROMPTS="请用标题加三个要点的格式介绍春联。\n用一句话解释什么是微调。\n列出三条学习英语的建议。\n你是谁？你由谁开发？\nexit"

echo "########## BASE (no adapter) ##########" > $EV/chat_olmoe_compare.txt
printf "$PROMPTS" | llamafactory-cli chat /tmp/olmoe_base_chat.yaml 2>>/dev/null >> $EV/chat_olmoe_compare.txt
echo "########## ALPACA-SFT (adapter) ##########" >> $EV/chat_olmoe_compare.txt
printf "$PROMPTS" | llamafactory-cli chat /tmp/olmoe_alpaca_chat.yaml 2>>/dev/null >> $EV/chat_olmoe_compare.txt
grep -A3 "^User:" $EV/chat_olmoe_compare.txt | head -60
echo CHATCMP_DONE
