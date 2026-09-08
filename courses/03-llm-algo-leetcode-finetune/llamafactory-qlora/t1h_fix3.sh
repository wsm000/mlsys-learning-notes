#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
pip install "huggingface_hub==0.36.2" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2
pip uninstall -y kernels 2>&1 | tail -1
ls ~/venvs/llamafactory/lib/python3.10/site-packages/kernels.py && echo STUB_PRESENT
echo "=== FINAL VERIFY ==="
python -c "import huggingface_hub; print('hub', huggingface_hub.__version__)"
python -c "import llamafactory; print('LF', llamafactory.__version__)" 2>&1 | tail -1
llamafactory-cli version 2>&1 | tail -4
python -c "import torch, bitsandbytes, peft, trl, swanlab, transformers, datasets, accelerate; print('STACK_OK | torch', torch.__version__, '| tf', transformers.__version__, '| bnb', bitsandbytes.__version__, '| peft', peft.__version__, '| trl', trl.__version__, '| swanlab', swanlab.__version__)" 2>&1 | tail -1
df -h / | tail -1
echo FIX3_DONE
