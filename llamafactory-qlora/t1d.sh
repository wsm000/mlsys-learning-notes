#!/usr/bin/env bash
set -x
cd ~/projects/LLaMA-Factory || exit 1
git checkout -- data/identity.json 2>/dev/null
git checkout v0.9.3 2>&1 | tail -2
git log --oneline -1
echo "=== python requirement hints ==="
ls setup.py 2>/dev/null && grep -n "python" setup.py | head -4
grep -rn "REQUIRED_VERSION\|MIN_PYTHON\|python_requires" setup.py pyproject.toml src/llamafactory/extras.py 2>/dev/null | head -6
sed -i "s/{{name}}/小智/g; s/{{author}}/hello-gpu 实验室/g" data/identity.json
echo "identity patched: $(grep -c "小智" data/identity.json) hits"
git status --short | head -5
echo T1D_DONE
