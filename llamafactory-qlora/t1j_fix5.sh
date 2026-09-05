#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
pip install "swanlab<0.7" -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2
python -c "import swanlab; print('swanlab', swanlab.__version__)"
# 快速验证回调路径: swanlab.get_run() 行为
python -c "
try:
    r = swanlab.get_run()
    print('get_run returns:', r)
except Exception as e:
    print('get_run raises:', type(e).__name__)
" 2>&1 | tail -2
echo FIX5_DONE
