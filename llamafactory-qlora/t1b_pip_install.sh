#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com
source ~/venvs/llamafactory/bin/activate
cd ~/projects/LLaMA-Factory || exit 1
python -c "import torchvision" 2>/dev/null && EXTRA="[torch,metrics]" || EXTRA="[metrics]"
echo "EXTRA=$EXTRA"
echo "=== pip install -e ==="
pip install -e ".$EXTRA" -i https://pypi.tuna.tsinghua.edu.cn/simple || pip install -e ".$EXTRA" || exit 1
echo "=== pip install swanlab ==="
pip install swanlab -i https://pypi.tuna.tsinghua.edu.cn/simple || pip install swanlab || exit 1
echo "=== VERIFY ==="
which llamafactory-cli
llamafactory-cli version
python -c "import torch, bitsandbytes, peft, transformers, trl, swanlab; print('STACK_OK | torch', torch.__version__, '| transformers', transformers.__version__, '| bnb', bitsandbytes.__version__, '| trl', trl.__version__, '| swanlab', swanlab.__version__)" || exit 1
df -h / | tail -1
echo T1B_DONE
