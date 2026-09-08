import json, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use gpt2 as lightweight substitute for Qwen2.5-Math-1.5B
model_id = "openai-community/gpt2"
print(f"Loading {model_id} on vm-60 4090D...")
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Model loaded on {device}, dtype {model.dtype}")

# Try to find GSM8K test data
paths = [
    "/home/simin/projects/diy-llm/coursework/assignment5-alignment/data/gsm8k/test.jsonl",
    "/home/simin/projects/diy-llm/coursework/assignment6-evaluation/data/index_testset.jsonl",
    "data/gsm8k/test.jsonl"
]
data_path = None
for p in paths:
    if os.path.exists(p):
        data_path = p
        break
if not data_path:
    # create dummy
    print("No data found, creating dummy")
    os.makedirs("data/gsm8k", exist_ok=True)
    data_path = "data/gsm8k/test.jsonl"
    with open(data_path, "w") as f:
        json.dump({"question": "What is 2+2?", "answer": "#### 4"}, f)
        f.write("\n")

print(f"Using data {data_path}")
with open(data_path) as f:
    first = f.read(1)
    f.seek(0)
    if first == '[':
        data = json.load(f)
    else:
        data = [json.loads(l) for l in f if l.strip()]

# Take 5 examples
data = data[:5]
print(f"Evaluating {len(data)} examples")

# Try to use reward fn
try:
    from drgrpo_grader import r1_zero_reward_fn
    has_reward = True
except:
    has_reward = False
    def r1_zero_reward_fn(resp, gt, fast=True):
        return {"reward":0,"format_reward":0,"answer_reward":0}

out_path = "/home/simin/projects/diy-llm/coursework/assignment5-alignment/results/base/zero_shot_math_evaluation_vm60.jsonl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

results = []
for i, ex in enumerate(data):
    q = ex.get('question', ex.get('problem',''))
    gt_raw = ex.get('answer', ex.get('expected_answer',''))
    m = re.search(r"####\s*(.+)\s*$", gt_raw.strip()) if isinstance(gt_raw, str) else None
    gt = m.group(1).strip() if m else gt_raw
    prompt = f"Question: {q}\nAnswer: Let's think step by step.\n"
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0], skip_special_tokens=True)
    # take only generated part
    generated = txt[len(prompt):] if txt.startswith(prompt) else txt
    rewards = r1_zero_reward_fn(generated, str(gt), fast=True) if has_reward else {"reward":0}
    print(f"\n--- Example {i+1} ---")
    print(f"Q: {q[:100]}")
    print(f"GT: {gt}")
    print(f"Gen: {generated[:200]}")
    print(f"Rewards: {rewards}")
    results.append({"question":q,"ground_truth":gt,"model_response":generated,"rewards":rewards})

with open(out_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"\nSaved to {out_path}")
# also save metrics
metrics = {"n":len(results),"model":model_id,"device":device}
print(json.dumps(metrics, indent=2))
