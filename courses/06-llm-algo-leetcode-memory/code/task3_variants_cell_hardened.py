# Task3 Colab · 4.2(3) 独立 workload 实测单元格（加固版：带形状断言）
# 用法：整格替换 Colab 中对应的 variants 单元格后从头运行本格。
# 背景：初版只重跑末尾 append 行时，small_batch_eval 残留第一次失败的旧值 (8,1,128)，
#       eval_variant 再 unsqueeze(0) 得到 3 维输入，模型里 q 变 5 维、cos 停在 4 维，
#       报 repeat_kv "too many values to unpack" 或 rotary "size of tensor a (14) ... (64) at dimension 3"。
#       本版在入口处断言形状，任何残留旧值都会在进模型前被断言拦住。

def repack(tokens_tensor, n_chunks, seq):
    flat = tokens_tensor.reshape(-1)
    need = n_chunks * seq
    assert flat.numel() >= need, "token 不足，不能复用数据伪造验证集"
    return flat[:need].reshape(n_chunks, seq)

def run_variant(name, batches, eval_tokens, effective_batch, micro, note):
    # 训练批次必须是 (N, effective_batch, seq)；验证必须是 (chunks, seq)，每行一条 seq 向量。
    assert batches.dim() == 3 and eval_tokens.dim() == 2, (name, tuple(batches.shape), tuple(eval_tokens.shape))
    assert batches.shape[1] == effective_batch, (name, batches.shape, effective_batch)
    assert batches.shape[2] == eval_tokens.shape[1], (name, batches.shape, eval_tokens.shape)
    print("[variant]", name, "batches", tuple(batches.shape), "eval", tuple(eval_tokens.shape))
    cleanup()
    model, ident, budget = make_model(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16, init_scale=1024.0)

    def eval_variant():
        model.eval()
        total, targets = 0.0, 0
        with torch.no_grad():
            for row in eval_tokens:
                xv = row.unsqueeze(0).to(DEVICE)  # (seq,) -> (1, seq)，二维喂模型
                assert xv.dim() == 2, (name, tuple(xv.shape))
                with torch.autocast("cuda", dtype=AMP_DTYPE):
                    lv = model(input_ids=xv, attention_mask=torch.ones_like(xv), labels=xv, use_cache=False).loss
                n = xv.shape[0] * (xv.shape[1] - 1)
                total += float(lv) * n
                targets += n
                del xv, lv
        sync()
        model.train()
        return total / targets

    seq = batches.shape[-1]
    result = dict(strategy=name, workload="variant:seq%d/eff%d" % (seq, effective_batch),
                  micro_batch=micro, accumulation=effective_batch // micro, checkpoint=False, offload=False,
                  effective_batch=effective_batch, seq_len=seq, status="RUNNING", steps=[], note=note)
    try:
        initial = eval_variant()
        cleanup()
        for i, batch in enumerate(batches):
            if i == CFG["warmup_steps"]:
                optimizer.zero_grad(set_to_none=True)
                sync()
                torch.cuda.reset_peak_memory_stats(DEVICE)
            record = train_step(model, optimizer, scaler, batch, micro, effective_batch=effective_batch)
            record.update(step=i, phase="warmup" if i < CFG["warmup_steps"] else "measured")
            result["steps"].append(record)
            print(name, record)
            if record["skipped_update"] or not math.isfinite(record["loss"]):
                result["status"] = "REJECT_SKIPPED_OR_NONFINITE"
                return result
        measured = [r for r in result["steps"] if r["phase"] == "measured"]
        result.update(initial_validation_loss=initial,
                      train_peak_allocated_gib=max(r["peak_allocated_gib"] for r in measured),
                      train_peak_reserved_gib=max(r["peak_reserved_gib"] for r in measured),
                      target_tokens_per_second=len(measured) * effective_batch * (seq - 1) / sum(r["seconds"] for r in measured),
                      step_seconds_median=statistics.median(r["seconds"] for r in measured))
        optimizer.zero_grad(set_to_none=True)
        result["final_validation_loss"] = eval_variant()
        result["status"] = "COMPLETE" if math.isfinite(result["final_validation_loss"]) else "REJECT_NONFINITE"
        return result
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        model = optimizer = scaler = None
        cleanup()
        save_json(name + ".json", _clean(result))

# 同一 token 流重新切块：有效 batch=1 与 seq=64；数据来源与主表相同
small_batch_batches = repack(train_cpu, N_STEPS, CFG["seq_len"]).unsqueeze(1)
small_batch_eval = repack(eval_cpu, CFG["eval_chunks"], CFG["seq_len"])
short_seq_batches = repack(train_cpu, N_STEPS * CFG["effective_batch"], 64).reshape(N_STEPS, CFG["effective_batch"], 64)
short_seq_eval = repack(eval_cpu, CFG["eval_chunks"], 64)
print("shapes:", tuple(small_batch_batches.shape), tuple(small_batch_eval.shape),
      tuple(short_seq_batches.shape), tuple(short_seq_eval.shape))  # 期望 (8,1,128) (8,128) (8,2,64) (8,64)

EXTRA_RESULTS.append(run_variant("small_batch_eff1", small_batch_batches, small_batch_eval,
                                 effective_batch=1, micro=1,
                                 note="有效 batch 4→1：工作负载改变，只比显存与吞吐，不与主表比 loss"))
EXTRA_RESULTS.append(run_variant("short_seq64", short_seq_batches, short_seq_eval,
                                 effective_batch=CFG["effective_batch"], micro=CFG["effective_batch"],
                                 note="序列 128→64：工作负载改变，只比显存与吞吐，不与主表比 loss"))
for r in EXTRA_RESULTS:
    print(r["strategy"], r["status"], r.get("train_peak_reserved_gib"), r.get("target_tokens_per_second"))
