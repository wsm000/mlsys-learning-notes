#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
cd ~/projects/LLaMA-Factory || exit 1
cat > /tmp/smoke_olmoe.yaml <<'EOF'
model_name_or_path: /home/simin/local_models/allenai/OLMoE-1B-7B-0125
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 8
lora_alpha: 16
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true
dataset: identity_vm60
template: default
cutoff_len: 512
max_steps: 20
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
output_dir: /home/simin/projects/llamafactory-practice/saves/smoke-olmoe
report_to: none
EOF
bash /tmp/run_train.sh /tmp/smoke_olmoe.yaml mem_smoke_olmoe.log train_smoke_olmoe.log 2>&1 | tail -30
echo "=== QUANT_EVIDENCE ==="
grep -iE "Quantizing model|4-bit|Linear4bit" ~/projects/llamafactory-practice/evidence/train_smoke_olmoe.log | head -4 | cut -c1-160
echo SMOKE_OLMOE_DONE
