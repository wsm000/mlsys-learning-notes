#!/bin/bash
# vm-60 Task6 A线轻量复核(磁盘紧张版,不装包不下载,默认只检查不训练)
# 用法:
#   bash vm60_task6_light_check.sh                 # 只做环境+YAML检查(推荐先跑这个)
#   bash vm60_task6_light_check.sh --run-20        # 另加20步冒烟,写 /tmp/vm60_sweep_20.json(不覆盖打卡json)
#   bash vm60_task6_light_check.sh --run-100       # 另加100步交叉验证,写 /tmp/vm60_sweep_100.json
# 要求:脚本与 task6_vm60_lora_sweep.py 同目录或 ~/ 下;GPT2权重在 ~/local_models/gpt2
set -u
RUN20=0; RUN100=0
for a in "$@"; do
  case "$a" in --run-20) RUN20=1;; --run-100) RUN100=1;; *) echo "[warn] 未知参数 $a,忽略";; esac
done
echo "=== vm-60 Task6 light check ==="
hostname; date
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1 | head -n 5
echo "--- disk/mem ---"
df -h / | head -n 5
free -h | head -n 5
echo "--- python/torch ---"
python3 --version 2>&1
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -n 3
python3 -c "import transformers;print('transformers',transformers.__version__)" 2>&1 | head -n 3
echo "--- gpt2 weights ---"
ls -lh ~/local_models/gpt2 2>&1 | head -n 10
du -sh ~/local_models/gpt2 2>&1
echo "--- sweep script ---"
ls -lh ./task6_vm60_lora_sweep.py ~/task6_vm60_lora_sweep.py 2>&1 | head -n 5
echo "--- yaml templates(若已拷贝到vm-60) ---"
for f in ./llamafactory_qlora_qwen25_05b_vm60.yaml ./llamafactory_lora_qwen25_05b_vm60.yaml ~/llamafactory_qlora_qwen25_05b_vm60.yaml; do
  if [ -f "$f" ]; then
    echo "== $f =="
    grep -E "model_name_or_path|quantization_bit|lora_rank|lora_alpha|lora_dropout|report_to" "$f" | head -n 12
    grep -A4 "lora_target" "$f" | head -n 6
  fi
done
echo "--- yaml key check(当前目录) ---"
for f in llamafactory_qlora_qwen25_05b_vm60.yaml llamafactory_lora_qwen25_05b_vm60.yaml; do
  [ -f "$f" ] || continue
  echo "-- $f --"
  grep -q "quantization_bit: 4" "$f" && echo "quant=4bit(QLoRA)" || grep -q "quantization_bit: null" "$f" && echo "quant=null(纯LoRA)" || echo "quant行异常"
  grep -q "q_proj" "$f" && grep -q "v_proj" "$f" && echo "target=q/v OK" || echo "target行请核对"
  grep -q "report_to: swanlab" "$f" && echo "swanlab OK" || echo "report_to缺失"
done
if [ "$RUN20" -eq 1 ]; then
  echo "--- run 20-step smoke -> /tmp/vm60_sweep_20.json ---"
  PY=$(ls ./task6_vm60_lora_sweep.py 2>/dev/null || echo ~/task6_vm60_lora_sweep.py)
  python3 "$PY" --model ~/local_models/gpt2 --max-steps 20 --out /tmp/vm60_sweep_20.json 2>&1 | tail -n 15
  echo "saved: /tmp/vm60_sweep_20.json (传回本机请改名,勿覆盖 vm60_sweep_report.json)"
fi
if [ "$RUN100" -eq 1 ]; then
  echo "--- run 100-step cross-check -> /tmp/vm60_sweep_100.json ---"
  PY=$(ls ./task6_vm60_lora_sweep.py 2>/dev/null || echo ~/task6_vm60_lora_sweep.py)
  python3 "$PY" --model ~/local_models/gpt2 --max-steps 100 --out /tmp/vm60_sweep_100.json 2>&1 | tail -n 15
  echo "saved: /tmp/vm60_sweep_100.json"
fi
echo "DONE light-check(未pip安装,未覆盖打卡json)"
