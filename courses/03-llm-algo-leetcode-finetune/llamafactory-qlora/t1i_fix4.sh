#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
python - <<'PYEOF'
import os
sp = "/home/simin/venvs/llamafactory/lib/python3.10/site-packages"
code = '''"""Minimal kernels shim for the llamafactory venv.

System kernels 0.16.x is incompatible with pinned huggingface_hub 0.36.2
(strict-dataclass union validator). transformers 4.52 hard-imports kernels
whenever find_spec sees it, so we shadow it with an API-compatible no-op.
Hub-kernel loading is not used by Qwen2.5 SFT/LoRA training.
"""
class Device:
    CUDA = "cuda"
    CPU = "cpu"
    NPU = "npu"
    XPU = "xpu"

class LayerRepository:
    def __init__(self, repo_id=None, layer_name=None, **kwargs):
        self.repo_id = repo_id
        self.layer_name = layer_name

def register_kernel_mapping(mapping):
    return None

def replace_kernel_forward_from_hub(*args, **kwargs):
    return None

def use_kernel_forward_from_hub(*args, **kwargs):
    def decorator(cls):
        return cls
    return decorator

def get_kernel(*args, **kwargs):
    raise RuntimeError("kernels shim: hub kernel loading disabled in this venv")
'''
with open(os.path.join(sp, "kernels.py"), "w") as f:
    f.write(code)
print("shim written")
PYEOF
echo "=== FINAL VERIFY ==="
python -c "import llamafactory; print('LF', llamafactory.__version__)" 2>&1 | tail -1
llamafactory-cli version 2>&1 | tail -4
python -c "import torch, bitsandbytes, peft, trl, swanlab, transformers, datasets, accelerate; print('STACK_OK | torch', torch.__version__, '| tf', transformers.__version__, '| bnb', bitsandbytes.__version__, '| peft', peft.__version__, '| trl', trl.__version__, '| swanlab', swanlab.__version__)" 2>&1 | tail -1
df -h / | tail -1
echo FIX4_DONE
