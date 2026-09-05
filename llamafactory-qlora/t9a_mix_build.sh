#!/usr/bin/env bash
set -x
source ~/venvs/llamafactory/bin/activate
cd ~/projects/LLaMA-Factory || exit 1
python - <<'EOF'
import json, os
idata = json.load(open("data/identity_vm60.json"))
alpaca = json.load(open("data/alpaca_gpt4_zh.json"))
print("identity entries:", len(idata), "| first output:", idata[0]["output"][:60])
print("alpaca entries:", len(alpaca))

def mix(ni, na, fname):
    rep = (idata * (ni // len(idata) + 1))[:ni]
    m = rep + alpaca[:na]
    json.dump(m, open("data/" + fname + ".json", "w"), ensure_ascii=False)
    print(f"{fname}: total={len(m)} identity={ni} frac={ni/len(m):.1%}")

mix(91, 800, "mix10")
mix(273, 640, "mix30")
mix(455, 455, "mix50")

reg_path = "data/dataset_info.json"
reg = json.load(open(reg_path))
for f in ["mix10", "mix30", "mix50"]:
    reg[f] = {"file_name": f + ".json"}
json.dump(reg, open(reg_path, "w"), ensure_ascii=False, indent=2)
print("registered:", list(reg.keys()))
EOF
echo "=== configs ==="
for m in mix10 mix30 mix50; do
  sed -e "s/^dataset:.*/dataset: $m/" -e "s/^max_samples:.*//" -e "s/^num_train_epochs:.*/num_train_epochs: 1.0/" -e "s|^output_dir:.*|output_dir: /home/simin/projects/llamafactory-practice/saves/olmoe-$m|" \~^e^/projects^/llamafactory-practice/config/olmoe_alpaca.yaml > ~/projects/llamafactory-practice/config/olmoe_$m.yaml 2>/dev/null || sed -e "s/^dataset:.*/dataset: $m/" -e "/^max_samples:/d" -e "s/^num_train_epochs:.*/num_train_epochs: 1.0/" -e "s|^output_dir:.*|output_dir: /home/simin/projects/llamafactory-practice/saves/olmoe-$m|" ~/projects/llamafactory-practice/config/olmoe_alpaca.yaml > ~/projects/llamafactory-practice/config/olmoe_$m.yaml
  grep -E "^dataset:|^num_train_epochs:|output_dir:" ~/projects/llamafactory-practice/config/olmoe_$m.yaml | head -3
done
df -h / | tail -1
echo BUILD_DONE
