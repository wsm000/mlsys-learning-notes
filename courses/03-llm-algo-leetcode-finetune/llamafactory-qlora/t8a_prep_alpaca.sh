#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
echo "=== belle_multiturn 本地内容（可能的零下载备选） ==="
ls -la ~/projects/LLaMA-Factory/data/belle_multiturn/ 2>/dev/null | head -5
echo "=== 下载 alpaca_gpt4_zh (~3.5MB via hf-mirror) ==="
python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download(repo_id="llamafactory/alpaca_gpt4_zh", filename="alpaca_gpt4_zh.json", repo_type="dataset")
print("cached:", p, os.path.getsize(p), "bytes")
shutil.copy(p, "/home/simin/projects/LLaMA-Factory/data/alpaca_gpt4_zh.json")
import json
d = json.load(open("/home/simin/projects/LLaMA-Factory/data/alpaca_gpt4_zh.json"))
print("entries:", len(d))
print("sample:", json.dumps(d[0], ensure_ascii=False)[:180])
EOF
echo "=== 注册进 dataset_info.json ==="
python - <<'EOF'
import json
p = "/home/simin/projects/LLaMA-Factory/data/dataset_info.json"
reg = json.load(open(p))
reg["alpaca_gpt4_zh"] = {
    "file_name": "alpaca_gpt4_zh.json",
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "query": "input", "response": "output"}
}
json.dump(reg, open(p, "w"), ensure_ascii=False, indent=2)
print("registered:", list(reg.keys()))
EOF
df -h / | tail -1
echo PREP_DONE
