#!/usr/bin/env bash
set -x
export HF_ENDPOINT=https://hf-mirror.com
grep -q HF_ENDPOINT ~/.bashrc || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
cd ~/projects || exit 1
if [ ! -d LLaMA-Factory ]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git 2>&1 || exit 1
else
  echo "LLaMA-Factory already cloned"
fi
cd LLaMA-Factory || exit 1
python3 -m venv --system-site-packages ~/venvs/llamafactory || exit 1
source ~/venvs/llamafactory/bin/activate
echo "=== TV_CHECK ==="
python -c "import torchvision; print('torchvision', torchvision.__version__)" 2>&1 | tail -1
echo "=== PIP_CONFIG ==="
pip config list 2>/dev/null || echo "no pip config"
echo "=== BASE_DEPS_PRESENT ==="
python -c "import transformers, datasets, accelerate, peft; print('transformers', transformers.__version__); print('datasets', datasets.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)" 2>&1
df -h / | tail -1
echo T1A_DONE
