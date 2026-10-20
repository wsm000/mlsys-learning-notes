"""Real, bounded GPU LoRA SFT and before/after checks on PDF-derived QA.

Uses Transformers/PEFT for the locally available Qwen3.5 model. This is an
explicit alternative to the tutorial's LlamaFactory/vLLM training backend.
"""
import json
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("DF_BASE_MODEL", "Qwen/Qwen3.5-2B")
random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def encode(row):
    messages = [{"role": "user", "content": row["instruction"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    completed = prefix + row["output"] + tokenizer.eos_token
    ids = tokenizer(completed, truncation=True, max_length=384, return_tensors="pt")["input_ids"]
    prefix_length = len(tokenizer(prefix)["input_ids"])
    labels = ids.clone()
    labels[:, :prefix_length] = -100
    assert (labels != -100).any()
    return {"input_ids": ids.cuda(), "attention_mask": torch.ones_like(ids).cuda(), "labels": labels.cuda()}


@torch.no_grad()
def measure(rows):
    model.eval()
    total, count = 0.0, 0
    for row in rows:
        batch = encode(row)
        n = int((batch["labels"][:, 1:] != -100).sum())
        total += float(model(**batch, use_cache=False).loss) * n
        count += n
    return total / count


@torch.no_grad()
def answer(question):
    model.eval()
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**batch, max_new_tokens=80, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    start = time.time()
    evidence = json.loads((ROOT / "qa_evidence.json").read_text())
    # Human review rejects an overly broad question and a visibly truncated quote.
    rejected_questions = {
        "What is the main technique used in computer Go?",
        "How did researchers try to make systems that worked the way their minds worked?",
    }
    curated = [r for r in evidence if r["instruction"] not in rejected_questions]
    assert len(curated) >= 5
    train_rows, eval_rows = curated[:-1], curated[-1:]
    save("train.json", train_rows)
    save("eval.json", eval_rows)
    save("manual_review.json", {"rejected_questions": sorted(rejected_questions), "reason": "question too broad or answer visibly truncated", "train_count": len(train_rows), "eval_count": len(eval_rows)})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    print("LOADING_BASE_MODEL", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True)
    model.config.use_cache = False
    print("BASE_MODEL_LOADED", flush=True)
    before_train, before_eval = measure(train_rows), measure(eval_rows)
    questions = [train_rows[0]["instruction"], eval_rows[0]["instruction"]]
    before_answers = [answer(q) for q in questions]
    save("before.json", {"train_loss": before_train, "eval_loss": before_eval, "answers": before_answers})
    print("BASELINE_SAVED", flush=True)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    losses = []
    for step in range(48):
        model.train()
        row = train_rows[step % len(train_rows)]
        optimizer.zero_grad(set_to_none=True)
        loss = model(**encode(row), use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 4 == 0:
            print(f"TRAIN_STEP {step + 1}/48 LOSS {losses[-1]:.6f}", flush=True)
    model.gradient_checkpointing_disable()
    after_train, after_eval = measure(train_rows), measure(eval_rows)
    after_answers = [answer(q) for q in questions]
    model.save_pretrained(ROOT / "adapter")
    tokenizer.save_pretrained(ROOT / "adapter")
    # Reload the saved weights through the adapter API and verify reproducibility.
    model.load_adapter(ROOT / "adapter", adapter_name="reloaded")
    model.set_adapter("reloaded")
    reloaded_eval = measure(eval_rows)
    assert abs(reloaded_eval - after_eval) < 0.02
    summary = {
        "status": "PASS", "model": MODEL, "training_backend": "Transformers + PEFT", "seed": 42,
        "train_examples": len(train_rows), "heldout_examples": len(eval_rows), "steps": len(losses),
        "lora_rank": 8, "lora_alpha": 16, "learning_rate": 1e-4, "trainable_parameters": trainable, "total_parameters": total,
        "before_train_loss": before_train, "after_train_loss": after_train,
        "before_heldout_loss": before_eval, "after_heldout_loss": after_eval, "reloaded_heldout_loss": reloaded_eval,
        "elapsed_seconds": time.time() - start, "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "limitations": "Tiny educational dataset; one held-out example. No claim of general capability improvement.",
    }
    save("training_losses.json", losses)
    save("training_summary.json", summary)
    save("comparison.json", [{"split": "train" if i == 0 else "heldout", "question": q, "before": before_answers[i], "after": after_answers[i]} for i, q in enumerate(questions)])
    print(json.dumps(summary, indent=2), flush=True)
