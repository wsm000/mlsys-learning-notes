"""Ask the PDF-finetuned model a question: python3 chat_pdf_lora.py 'question'."""
import sys
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

root = Path(__file__).resolve().parent
model_path = os.environ.get("DF_BASE_MODEL", "Qwen/Qwen3.5-2B")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True)
model = PeftModel.from_pretrained(model, root / "adapter").eval()
question = " ".join(sys.argv[1:]) or "What is the main lesson from the bitter lesson?"
prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
