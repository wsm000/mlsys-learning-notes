# Task3 Colab · 一步补齐单元格：去重 → 跑两个独立 workload 变体 → 重建 summary.json（不重新打包 ZIP）
# 依赖：已运行过主表(RESULTS)、offload 格(EXTRA_RESULTS/_clean)、汇总格以前的定义(train_step/make_model/save_json/RUN)。

# 0) 去重：只保留非 variants 的已测结果，避免重复 append
_keep = [r for r in EXTRA_RESULTS if r.get("strategy") not in ("small_batch_eff1", "short_seq64")]
EXTRA_RESULTS.clear(); EXTRA_RESULTS.extend(_keep)
print("EXTRA_RESULTS 现有:", [r["strategy"] for r in EXTRA_RESULTS])

# 1) 独立 workload 变体（同一 token 流重新切块，数据来源与主表相同）
def repack(tokens_tensor, n_chunks, seq):
    flat = tokens_tensor.reshape(-1)
    need = n_chunks * seq
    assert flat.numel() >= need, "token 不足，不能复用数据伪造验证集"
    return flat[:need].reshape(n_chunks, seq)

def run_variant(name, batches, eval_tokens, effective_batch, micro, note):
    # 训练批次必须 3 维 (N, eff, seq)；验证必须 2 维 (chunks, seq)；否则进模型前就断言失败
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

# 2) 重建 summary.json + 打印实测对比表（不打包 ZIP）
import pandas as pd
ALL_RESULTS = list(RESULTS) + list(EXTRA_RESULTS)
rows = []
for r in ALL_RESULTS:
    if r["status"] != "COMPLETE":
        rows.append(dict(strategy=r["strategy"], workload=r.get("workload"), status=r["status"]))
        continue
    rows.append(dict(strategy=r["strategy"], workload=r.get("workload"), status=r["status"],
                     peak_reserved_gib=round(r["train_peak_reserved_gib"], 3),
                     peak_allocated_gib=round(r["train_peak_allocated_gib"], 3),
                     tokens_per_s=round(r["target_tokens_per_second"], 1),
                     initial_loss=round(r["initial_validation_loss"], 6),
                     final_loss=round(r["final_validation_loss"], 6)))
summary = pd.DataFrame(rows)
base = next(r for r in ALL_RESULTS if r["strategy"] == "baseline" and r["status"] == "COMPLETE")
summary["memory_saving_vs_baseline_pct"] = [
    round((1 - p / base["train_peak_reserved_gib"]) * 100, 2) if pd.notna(p) else None
    for p in summary["peak_reserved_gib"]]
summary["throughput_ratio_vs_baseline"] = [
    round(t / base["target_tokens_per_second"], 3) if pd.notna(t) else None
    for t in summary["tokens_per_s"]]
summary["final_loss_delta_vs_baseline"] = [
    round(f - base["final_validation_loss"], 6) if pd.notna(f) else None
    for f in summary["final_loss"]]
# 前三行来自 run_case，没有 workload 字段，但与主表同 workload，手工标 True
summary["same_workload_as_baseline"] = summary["workload"].fillna("").str.startswith("main") | summary["strategy"].isin(["baseline", "accumulation", "checkpoint"])
display(summary)
try:
    ledger_for_summary = LEDGER
except NameError:
    ledger_for_summary = None
save_json("summary.json", dict(
    note="所有峰值/吞吐/loss 均为实测；variants 工作负载不同，只比显存与吞吐；weight_nf4_static 只做静态/前向。",
    ledger=ledger_for_summary,
    summary=summary.where(pd.notna(summary), None).to_dict(orient="records")))
print("summary.json 已更新；两个变体 JSON 已写入 RUN 目录。未重新打包 ZIP。")
