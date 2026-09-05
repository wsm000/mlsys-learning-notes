#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
cd ~/projects/LLaMA-Factory || exit 1
cat > /tmp/smoke_0p5b.yaml <<'EOF'
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
quantization_bit: 4
quantization_method: bitsandbytes
quantization_type: nf4
dataset: identity_vm60
template: qwen
cutoff_len: 512
max_steps: 20
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
output_dir: /home/simin/projects/llamafactory-practice/saves/smoke-0p5b
report_to: none

EOF
llamafactory-cli train /tmp/smoke_0p5b.yaml 2>&1 | tail -40
ls /home/simin/projects/llamafactory-practice/saves/smoke-0p5b/ | head -8
df -h / | tail -1
echo SMOKE_DONE
