#!/usr/bin/env bash
git ls-remote --tags https://ghfast.top/https://github.com/hiyouga/LLaMA-Factory.git 2>/dev/null | grep -oP "refs/tags/v\\K[0-9.]+" | grep -v "\\$" | sort -rV | uniq | head -15 > /tmp/tags.txt
echo "=== TAG_LIST ==="
cat /tmp/tags.txt
echo "=== PYPROJECT_SCAN ==="
while read -r tag; do
  rp=$(timeout 20 curl -sL "https://ghfast.top/https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/v$tag/pyproject.toml" 2>/dev/null | grep -m1 "requires-python")
  echo "v$tag -> $rp"
done < /tmp/tags.txt
echo "=== SCAN_DONE ==="
