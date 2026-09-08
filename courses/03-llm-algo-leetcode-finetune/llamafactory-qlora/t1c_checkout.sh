#!/usr/bin/env bash
set -x
cd ~/projects/LLaMA-Factory || exit 1
mkdir -p ~/projects/llamafactory-practice/config
mv -f vm60_*.yaml ~/projects/llamafactory-practice/config/ 2>/dev/null
git fetch --depth 1 origin refs/tags/v0.9.3:refs/tags/v0.9.3 || exit 1
git checkout v0.9.3 2>&1 | tail -2
git log --oneline -1
echo "=== setup.py python check ==="
grep -n "python" setup.py | head -5
cp data/identity.json data/identity.json.bak 2>/dev/null
sed -i "s/{{name}}/小智/g; s/{{author}}/hello-gpu 实验室/g" data/identity.json
echo "identity patched: $(grep -c "小智" data/identity.json) hits"
echo T1C_DONE
