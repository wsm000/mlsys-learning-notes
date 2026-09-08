#!/usr/bin/env bash
echo ===TEMPLATE===
grep -n olmoe ~/projects/LLaMA-Factory/src/llamafactory/data/template.py | head -4
echo ===MODEL_FILES===
ls -lh ~/local_models/allenai/OLMoE-1B-7B-0125/ | head -16
echo ===CONFIG_KEY===
python3 - <<'EOF'
import json
c = json.load(open("/home/simin/local_models/allenai/OLMoE-1B-7B-0125/config.json"))
keys = ["model_type","hidden_size","intermediate_size","num_experts","num_experts_per_tok","num_hidden_layers","tie_word_embeddings","vocab_size","torch_dtype"]
print({k: c.get(k) for k in keys})
EOF
echo PROBE6_DONE
