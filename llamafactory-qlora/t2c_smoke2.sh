#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
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
bash /tmp/run_train.sh /tmp/smoke_0p5b.yaml mem_smoke.log train_smoke.log 2>&1 | tail -35
ls /home/simin/projects/llamafactory-practice/saves/smoke-0p5b/ 2>/dev/null | head -8
echo SMOKE2_DONE
