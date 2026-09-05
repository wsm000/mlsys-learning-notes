#!/usr/bin/env bash
source ~/venvs/llamafactory/bin/activate
echo "=== import tests (each isolated) ==="
python -c "import huggingface_hub; print('HUB', huggingface_hub.__version__, huggingface_hub.__file__)" 2>&1 | tail -2
python -c "import transformers; print('TRANSFORMERS', transformers.__version__, transformers.__file__)" 2>&1 | tail -2
python -c "import llamafactory; print('LF', llamafactory.__version__)" 2>&1 | tail -2
echo "=== pip freeze key packages ==="
pip list 2>/dev/null | grep -iE "^(transformers|huggingface|kernels|torch|torchvision|trl|peft|accelerate|datasets|llamafactory|swanlab|bitsandbytes) "
echo "=== disk ==="
df -h / | tail -1
echo DIAG_DONE
