#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
echo "=== ARGS DUMP in train_qlora.log ==="
head -100 ~/projects/llamafactory-practice/evidence/train_qlora.log | grep -E "quantization|compute_dtype|bf16|load_in" | head -8
echo "=== ISOLATED configure_quantization REPRO ==="
cd ~/projects/LLaMA-Factory/src
python - <<'EOF' 2>&1 | tail -8
import types, torch
from transformers import AutoConfig
from llamafactory.model.model_utils.quantization import configure_quantization
ma = types.SimpleNamespace(quantization_bit=4, quantization_method="bitsandbytes", quantization_type="nf4",
    double_quantization=True, compute_dtype=torch.bfloat16, quantization_device_map=None,
    export_quantization_bit=None, export_quantization_max_sequence_length=None)
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
init_kwargs = {}
configure_quantization(config, None, ma, init_kwargs)
print("INIT_KW:", {k: str(v)[:100] for k, v in init_kwargs.items()})
EOF
echo PROBE5_DONE
