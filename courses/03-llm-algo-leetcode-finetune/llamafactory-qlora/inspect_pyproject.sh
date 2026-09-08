#!/usr/bin/env bash
curl -sL "https://ghfast.top/https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/v0.9.3/pyproject.toml" -o /tmp/p393.toml
echo "=== requires lines ==="
grep -in "requires" /tmp/p393.toml | head -6
echo "=== python version mentions ==="
grep -n "3\.9\|3\.10\|3\.11\|3\.12" /tmp/p393.toml | head -10
echo "=== head of project section ==="
sed -n "1,35p" /tmp/p393.toml
