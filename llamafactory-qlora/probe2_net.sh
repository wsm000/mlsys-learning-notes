#!/usr/bin/env bash
echo "=== gitee ==="
timeout 15 git ls-remote https://gitee.com/hiyouga/LLaMA-Factory.git HEAD 2>&1 | head -1
echo "=== ghfast.top ==="
timeout 15 git ls-remote https://ghfast.top/https://github.com/hiyouga/LLaMA-Factory.git HEAD 2>&1 | head -1
echo "=== ghproxy.net ==="
timeout 15 git ls-remote https://ghproxy.net/https://github.com/hiyouga/LLaMA-Factory.git HEAD 2>&1 | head -1
echo "=== pypi.org ==="
timeout 10 curl -sI https://pypi.org/simple/ 2>&1 | head -1
echo "=== tuna ==="
timeout 10 curl -sI https://pypi.tuna.tsinghua.edu.cn/simple/ 2>&1 | head -1
echo "=== modelscope ==="
timeout 10 curl -sI https://www.modelscope.cn 2>&1 | head -1
echo "=== VENV (offline) ==="
python3 -m venv --system-site-packages ~/venvs/llamafactory && echo VENV_OK
source ~/venvs/llamafactory/bin/activate
python -c "import torchvision" 2>/dev/null && echo TV_PRESENT || echo TV_ABSENT
echo PROBE2_DONE
