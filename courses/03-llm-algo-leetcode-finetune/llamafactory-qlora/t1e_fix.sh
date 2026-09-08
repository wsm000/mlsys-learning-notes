#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
echo "=== versions before fix ==="
python -c "import huggingface_hub; print('venv hub', huggingface_hub.__version__)" 2>&1 | tail -1
/usr/bin/python3 -m pip show huggingface_hub 2>/dev/null | head -2
pip install "transformers>=4.45,<5" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -3
echo "=== VERIFY after fix ==="
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torch, bitsandbytes, peft, trl, swanlab, datasets, accelerate; print('torch', torch.__version__, '| bnb', bitsandbytes.__version__, '| peft', peft.__version__, '| trl', trl.__version__, '| swanlab', swanlab.__version__)"
python -c "import llamafactory; print('llamafactory', llamafactory.__version__)"
which llamafactory-cli
llamafactory-cli version
df -h / | tail -1
echo FIX_DONE
