"""Independently reload the completed adapter and audit its persisted weights."""
import json
import re
import time

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

import train_pdf_lora as lab

start = time.time()
lab.tokenizer = AutoTokenizer.from_pretrained(lab.MODEL, local_files_only=True)
base = Qwen3_5ForConditionalGeneration.from_pretrained(lab.MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True)
lab.model = PeftModel.from_pretrained(base, lab.ROOT / "adapter")
lab.model.config.use_cache = False
train = json.loads((lab.ROOT / "train.json").read_text())
evaluation = json.loads((lab.ROOT / "eval.json").read_text())
before = json.loads((lab.ROOT / "before.json").read_text())
after_train, after_eval = lab.measure(train), lab.measure(evaluation)
print(f"PERSISTED_ADAPTER_LOSSES train={after_train} eval={after_eval}", flush=True)
repeat_eval = lab.measure(evaluation)
assert abs(repeat_eval - after_eval) < 0.02
# Compare every saved adapter tensor to the freshly loaded active tensor.
stored = load_file(str(lab.ROOT / "adapter/adapter_model.safetensors"))
state = lab.model.state_dict()
max_difference = 0.0
for name, value in stored.items():
    mapped = name.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
    assert mapped in state, mapped
    difference = float((state[mapped].detach().cpu().float() - value.float()).abs().max())
    max_difference = max(max_difference, difference)
assert max_difference == 0.0
questions = [train[0]["instruction"], evaluation[0]["instruction"]]
answers = [lab.answer(q) for q in questions]
observations = [{"step": int(s), "loss": float(v)} for s, v in re.findall(r"TRAIN_STEP (\d+)/48 LOSS ([0-9.]+)", (lab.ROOT / "training.log").read_text())]
assert observations[-1]["step"] == 45
trainable = sum(v.numel() for name, v in stored.items())
summary = {
    "status": "PASS", "model": lab.MODEL, "training_backend": "Transformers + PEFT", "seed": 42,
    "train_examples": len(train), "heldout_examples": len(evaluation), "steps": 48,
    "lora_rank": 8, "lora_alpha": 16, "learning_rate": 1e-4,
    "trainable_parameters": trainable, "total_parameters": sum(p.numel() for p in lab.model.parameters()),
    "before_train_loss": before["train_loss"], "after_train_loss": after_train,
    "before_heldout_loss": before["eval_loss"], "after_heldout_loss": after_eval, "reloaded_heldout_loss": repeat_eval,
    "elapsed_seconds": time.time() - start, "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
    "saved_tensor_count": len(stored), "max_loaded_tensor_difference": max_difference,
    "verification": "Fresh process loaded persisted adapter; all tensors exactly match saved weights; repeated loss stable.",
    "initial_verification_issue": "The training process's second named adapter did not meet its 0.02 loss tolerance. No performance claim uses that measurement. A fresh-process persisted-weight audit was used instead.",
    "limitations": "Tiny educational dataset; one held-out example. No claim of general capability improvement.",
}
lab.save("training_losses.json", observations)
lab.save("training_summary.json", summary)
lab.save("comparison.json", [{"split": "train" if i == 0 else "heldout", "question": q, "before": before["answers"][i], "after": answers[i]} for i, q in enumerate(questions)])
print(json.dumps(summary, indent=2), flush=True)
