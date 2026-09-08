#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
echo "=== attempt 1: upgrade kernels in venv ==="
pip install -U kernels -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2
python -c "import kernels; print('kernels', kernels.__version__)" 2>&1 | tail -1
if python -c "import llamafactory" 2>/dev/null; then
  echo "KERNELS_UPGRADE_WORKED"
else
  echo "=== attempt 2: stub kernels in venv ==="
  python - <<'PYEOF'
import site, os
sp = [p for p in site.getsitepackages() if "venvs/llamafactory" in p][0]
p = os.path.join(sp, "kernels.py")
with open(p, "w") as f:
    f.write('# venv stub: shadow system kernels (incompatible with pinned hub)\nraise ImportError("kernels stubbed out in llamafactory venv")\n')
print("stub written to", p)
PYEOF
  python -c "import kernels" 2>&1 | tail -1
fi
echo "=== FINAL VERIFY ==="
python -c "import llamafactory; print('LF', llamafactory.__version__)" 2>&1 | tail -1
which llamafactory-cli
llamafactory-cli version 2>&1 | tail -3
python -c "import torch, bitsandbytes, peft, trl, swanlab, transformers, datasets, accelerate; print('STACK_OK | torch', torch.__version__, '| tf', transformers.__version__, '| bnb', bitsandbytes.__version__, '| peft', peft.__version__, '| trl', trl.__version__, '| swanlab', swanlab.__version__)" 2>&1 | tail -1
df -h / | tail -1
echo FIX2_DONE
