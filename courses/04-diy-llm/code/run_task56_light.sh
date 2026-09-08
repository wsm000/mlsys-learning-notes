#!/bin/bash
set -e
export HF_HUB_OFFLINE=0
export HF_ALLOW_CODE_EVAL=1
cd /home/simin/projects/diy-llm
echo "=== START $(date) ===" | tee /tmp/task56.log
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a /tmp/task56.log
echo "Disk:"; df -h / | tee -a /tmp/task56.log
echo "Python:"; python3 --version | tee -a /tmp/task56.log
pip --version | tee -a /tmp/task56.log

# ---- Task6: evalscope ----
echo "=== Task6 evalscope ===" | tee -a /tmp/task56.log
cd /home/simin/projects/diy-llm/coursework/assignment6-evaluation
python3 -c "import evalscope" 2>&1 | tee -a /tmp/task56.log || {
  echo "Installing evalscope..." | tee -a /tmp/task56.log
  pip install --no-cache-dir evalscope -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tee -a /tmp/task56.log
}
timeout 600 python3 evalscope_demo.py 2>&1 | tee -a /tmp/task56.log || echo "evalscope_demo exit: $?" | tee -a /tmp/task56.log
ls -lh outputs 2>&1 | tee -a /tmp/task56.log || true
ls -R outputs 2>&1 | head -200 | tee -a /tmp/task56.log || true

# ---- Task6: lm-eval ----
echo "=== Task6 lm-eval ===" | tee -a /tmp/task56.log
python3 -c "import lm_eval" 2>&1 | tee -a /tmp/task56.log || {
  echo "Installing lm-eval minimal..." | tee -a /tmp/task56.log
  pip install --no-cache-dir "lm-eval[api]" 2>&1 | tail -30 | tee -a /tmp/task56.log || true
}
timeout 600 python3 <<'PYEOF' 2>&1 | tee -a /tmp/task56.log || echo "lm-eval exit: $?" | tee -a /tmp/task56.log
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
lm = HFLM(pretrained='openai-community/gpt2', batch_size=4, device='cuda:0', max_length=512)
res = evaluator.simple_evaluate(model=lm, tasks=['hellaswag'], num_fewshot=0, limit=5, batch_size=4, confirm_run_unsafe_code=True)
print(res['results'])
PYEOF

# ---- Task5: math baseline with HF gpt2 ----
echo "=== Task5 math baseline (HF gpt2) ===" | tee -a /tmp/task56.log
cd /home/simin/projects/diy-llm/coursework/assignment5-alignment
python3 -c "import vllm" 2>&1 | tee -a /tmp/task56.log || echo "vllm not installed, using HF fallback" | tee -a /tmp/task56.log
timeout 300 python3 <<'PYEOF2' 2>&1 | tee -a /tmp/task56.log
import json, os, re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "openai-community/gpt2"
print(f"Loading {model_id}...")
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
path = "data/gsm8k/test.jsonl"
if not os.path.exists(path):
    alt = "../../coursework/assignment6-evaluation/data/index_testset.jsonl"
    print(f"gsm8k not found at {path}, checking {alt}")
    path = alt
if os.path.exists(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f][:5]
    for i, ex in enumerate(lines):
        q = ex.get('question', ex.get('problem', ''))[:300]
        prompt = f"Question: {q}\nAnswer: Let's think step by step.\n"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0], skip_special_tokens=True)
        print(f"--- Example {i+1} ---")
        print("Q:", q[:120])
        print("Generated:", txt[-300:])
        # also test reward fn if available
        try:
            from drgrpo_grader import r1_zero_reward_fn
            gt = ex.get('answer', ex.get('expected_answer', ''))[:50]
            m = re.search(r"####\s*(.+)\s*$", gt.strip())
            gt_clean = m.group(1).strip() if m else gt.strip()
            rewards = r1_zero_reward_fn(txt, gt_clean, fast=True)
            print("Rewards:", rewards)
        except Exception as e:
            print("reward err:", e)
else:
    print("No data file found")
print("Task5 HF done")
PYEOF2
echo "=== ALL DONE $(date) ===" | tee -a /tmp/task56.log
cat /tmp/task56.log | tail -100
