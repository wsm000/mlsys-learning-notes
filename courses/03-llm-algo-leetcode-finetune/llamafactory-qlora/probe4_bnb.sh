#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
python - <<'EOF' 2>&1 | tail -12
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
        quantization_config=qc, torch_dtype=torch.bfloat16, device_map="cuda:0")
print("PROJ_CLASS:", type(m.model.layers[0].self_attn.q_proj).__name__)
print("WEIGHT_DTYPE:", m.model.layers[0].self_attn.q_proj.weight.dtype)
print("MEM_MB:", torch.cuda.memory_allocated() // 1024 // 1024)
EOF
echo "=== llamafactory loader quant branch ==="
grep -n "quantization" ~/projects/LLaMA-Factory/src/llamafactory/model/loader.py | head -12
echo PROBE4_DONE
