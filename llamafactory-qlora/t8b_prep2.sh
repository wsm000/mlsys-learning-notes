#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os, json
p = hf_hub_download(repo_id="llamafactory/alpaca_gpt4_zh", filename="alpaca_gpt4_data_zh.json", repo_type="dataset")
dst = "/home/simin/projects/LLaMA-Factory/data/alpaca_gpt4_zh.json"
shutil.copy(p, dst)
d = json.load(open(dst))
print("entries:", len(d), "| size:", os.path.getsize(dst), "bytes")
print("sample:", json.dumps(d[0], ensure_ascii=False)[:150])
EOF
cat > ~/projects/llamafactory-practice/config/olmoe_alpaca.yaml <<'EOF'
model_name_or_path: /home/simin/local_models/allenai/OLMoE-1B-7B-0125
trust_remote_code: true
pure_bf16: false
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true
dataset: [alpaca_gpt4_zh, identity_vm60]
max_samples: 1000
template: default
cutoff_len: 512
overwrite_cache: true
preprocessing_num_workers: 4
output_dir: /home/simin/projects/llamafactory-practice/saves/olmoe-alpaca
logging_steps: 10
save_steps: 200
plot_loss: false
report_to: none
use_swanlab: true
swanlab_project: llamafactory-qlora-vm60
swanlab_mode: local
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.0
bf16: true
ddp_timeout: 180000000
val_size: 0.05
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 100
EOF
echo "config written:"; grep -E "dataset|max_samples|num_train_epochs" ~/projects/llamafactory-practice/config/olmoe_alpaca.yaml
df -h / | tail -1
echo PREP2_DONE
