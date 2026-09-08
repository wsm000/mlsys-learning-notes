#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for repo in ["llamafactory/alpaca_gpt4_zh", "llamafactory/demo_data", "llamafactory/alpaca-gpt4-zh"]:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
        print(repo, "->", files[:12])
    except Exception as e:
        print(repo, "-> FAIL:", type(e).__name__, str(e)[:120])
EOF
echo ===BELLE===
ls -la ~/projects/LLaMA-Factory/data/belle_multiturn/ 2>/dev/null | head -6
echo PROBE7_DONE
