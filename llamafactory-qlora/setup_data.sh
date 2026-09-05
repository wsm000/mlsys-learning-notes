#!/usr/bin/env bash
set -x
cd ~/projects/LLaMA-Factory || exit 1
# 从 main 分支留下的纯净备份生成我们的数据集
cp data/identity.json.bak data/identity_vm60.json
sed -i "s/{{name}}/小智/g; s/{{author}}/hello-gpu 实验室/g" data/identity_vm60.json
python3 -c "import json; d=json.load(open('data/identity_vm60.json')); print('entries:', len(d)); print(json.dumps(d[0], ensure_ascii=False)[:200])"
# v0.9.3 的 data/ 下没有注册表, 自建一份 (官方"自定义数据集注册"流程)
cat > data/dataset_info.json <<'EOF'
{
  "identity_vm60": {
    "file_name": "identity_vm60.json"
  }
}
EOF
cat data/dataset_info.json
# 更新 4 个 yaml 的数据集名
sed -i "s/^dataset: identity$/dataset: identity_vm60/" ~/projects/llamafactory-practice/config/vm60_*.yaml
grep -H "^dataset:" ~/projects/llamafactory-practice/config/vm60_*.yaml
echo SETUP_DATA_DONE
