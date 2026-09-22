"""Reproducible synthetic GPU training budget experiment (no downloads)."""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.checkpoint import checkpoint

CONFIG = dict(seed=73, batch=4, seq=512, width=256, hidden=1024, layers=6,
              dtype="float32", lr=0.001, warmup=3, repeats=9)
CASES = [("baseline", 4, False, False), ("gradient_accumulation", 1, False, False),
         ("checkpoint", 4, True, False), ("activation_offload", 4, False, True)]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["width"]
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(d, CONFIG["hidden"]),
            nn.GELU(), nn.Linear(CONFIG["hidden"], d)) for _ in range(CONFIG["layers"])])
        self.head = nn.Linear(d, d)

    def forward(self, x, ckpt, offload):
        ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else contextlib.nullcontext()
        with ctx:
            for block in self.blocks:
                x = x + (checkpoint(block, x, use_reentrant=False) if ckpt else block(x))
            return self.head(x)


def sync():
    torch.cuda.synchronize()


def size(t):
    return t.numel() * t.element_size()


def ledger(model, optimizer):
    params = sum(size(p) for p in model.parameters())
    grads = sum(size(p.grad) for p in model.parameters() if p.grad is not None)
    state_gpu = sum(size(v) for state in optimizer.state.values() for v in state.values()
                    if torch.is_tensor(v) and v.is_cuda)
    state_cpu = sum(size(v) for state in optimizer.state.values() for v in state.values()
                    if torch.is_tensor(v) and not v.is_cuda)
    assert grads == params and state_gpu == 2 * params
    return dict(parameter_bytes=params, gradient_bytes=grads, adam_cuda_bytes=state_gpu,
                adam_cpu_bytes=state_cpu, resident_cuda_bytes=params + grads + state_gpu)


def make(initial):
    model = Model().cuda()
    model.load_state_dict(initial)
    return model, torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], foreach=False)


def step(model, opt, data, case, instrument_stages=False, annotate=False):
    def mark(name):
        return record_function(name) if annotate else contextlib.nullcontext()

    def stage_sync():
        if instrument_stages:
            sync()
    _, micro, ckpt, offload = case
    stages = dict(data_loading=0.0, h2d=0.0, forward=0.0, backward=0.0, optimizer=0.0)
    sync()
    start = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    losses = []
    for offset in range(0, CONFIG["batch"], micro):
        t = time.perf_counter()
        with mark("data_loading"):
            # Fixed, already generated pinned CPU data: measures slicing, not disk I/O.
            hx, hy = (v[offset:offset + micro] for v in data)
        stages["data_loading"] += time.perf_counter() - t
        stage_sync(); t = time.perf_counter()
        with mark("H2D"):
            x, y = hx.to("cuda", non_blocking=True), hy.to("cuda", non_blocking=True)
        stage_sync(); stages["h2d"] += time.perf_counter() - t
        t = time.perf_counter()
        with mark("forward"):
            prediction = model(x, ckpt, offload)
            loss = (prediction - y).square().mean() * (micro / CONFIG["batch"])
        stage_sync(); stages["forward"] += time.perf_counter() - t
        t = time.perf_counter()
        with mark("backward"):
            loss.backward()
        stage_sync(); stages["backward"] += time.perf_counter() - t
        losses.append(loss.detach())
        del x, y, prediction, loss
    t = time.perf_counter()
    with mark("optimizer"):
        opt.step()
    stage_sync(); stages["optimizer"] += time.perf_counter() - t
    sync()
    elapsed = time.perf_counter() - start
    value = sum(v.item() for v in losses)
    assert math.isfinite(value)
    return dict(seconds=elapsed, stages_seconds=stages if instrument_stages else {}, loss=value,
                tokens_per_second=CONFIG["batch"] * CONFIG["seq"] / elapsed)


def clean():
    gc.collect()
    torch.cuda.empty_cache()
    sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="evidence/task3")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = Model().state_dict()
    data = tuple(torch.randn(CONFIG["batch"], CONFIG["seq"], CONFIG["width"]).pin_memory()
                 for _ in range(2))
    initial_vector = torch.cat([v.flatten() for v in initial.values()])
    environment = dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), vram_bytes=torch.cuda.get_device_properties(0).total_memory,
        python=platform.python_version(), platform=platform.platform(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(b"".join(v.numpy().tobytes() for v in data)).hexdigest())
    print("ENVIRONMENT", json.dumps(environment), flush=True)
    quality = []
    reference_loss = reference_update = None
    for case in CASES:
        model, opt = make(initial)
        measured = step(model, opt, data, case)
        updated = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
        delta = updated - initial_vector
        if reference_loss is None:
            reference_loss, reference_update = measured["loss"], delta.clone()
        loss_relative = abs(measured["loss"] - reference_loss) / abs(reference_loss)
        update_relative_l2 = (delta - reference_update).norm().item() / reference_update.norm().item()
        update_max_abs = (delta - reference_update).abs().max().item()
        passed = loss_relative <= 2e-6 and update_relative_l2 <= 2e-3 and update_max_abs <= 2e-5
        quality.append(dict(name=case[0], loss=measured["loss"], loss_relative=loss_relative,
            update_relative_l2=update_relative_l2, update_max_abs=update_max_abs, passed=passed))
        print("QUALITY", json.dumps(quality[-1]), flush=True)
        assert passed, quality[-1]
        del model, opt, updated, delta
        clean()
    results = []
    for case in CASES:
        model, opt = make(initial)
        for _ in range(CONFIG["warmup"]):
            step(model, opt, data, case)
        samples = []
        for _ in range(CONFIG["repeats"]):
            sync(); torch.cuda.reset_peak_memory_stats()
            sample = step(model, opt, data, case)
            sample.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            samples.append(sample)
        times = [s["seconds"] for s in samples]
        row = dict(name=case[0], micro_batch=case[1], accumulation_steps=CONFIG["batch"] // case[1],
            checkpoint=case[2], offload=case[3], samples=samples,
            median_seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times),
            std_seconds=statistics.stdev(times),
            tokens_per_second=CONFIG["batch"] * CONFIG["seq"] / statistics.median(times),
            timing_mode="step_boundary_sync_no_profiler",
            peak_allocated_bytes=max(s["peak_allocated_bytes"] for s in samples),
            peak_reserved_bytes=max(s["peak_reserved_bytes"] for s in samples),
            ledger=ledger(model, opt))
        assert all(s["seconds"] > 0 and sum(s["stages_seconds"].values()) <= s["seconds"] for s in samples)
        assert row["peak_allocated_bytes"] >= row["ledger"]["resident_cuda_bytes"]
        results.append(row)
        print("BENCHMARK", json.dumps(row), flush=True)
        del model, opt
        clean()
    # Stage diagnostics use independent model/optimizer warmups and serialized stages.
    for case, row in zip(CASES, results):
        model, opt = make(initial)
        for _ in range(CONFIG["warmup"]):
            step(model, opt, data, case, instrument_stages=True)
        stage_samples = [step(model, opt, data, case, instrument_stages=True) for _ in range(3)]
        row["stage_samples"] = stage_samples
        row["stages_median_seconds"] = {
            k: statistics.median(s["stages_seconds"][k] for s in stage_samples)
            for k in stage_samples[0]["stages_seconds"]}
        assert all(sum(s["stages_seconds"].values()) <= s["seconds"] for s in stage_samples)
        print("STAGES", json.dumps(dict(name=case[0], samples=stage_samples)), flush=True)
        del model, opt
        clean()
    # Profiler is a fresh run after ALL timed measurements, never a benchmark sample.
    model, opt = make(initial)
    step(model, opt, data, CASES[0])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof:
        step(model, opt, data, CASES[0], annotate=True)
    trace_path = out / "task3_cpu_cuda_trace.json"
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    assert any(e.get("cat") == "kernel" for e in events), "Missing CUDA kernels"
    for name in ["data_loading", "H2D", "forward", "backward", "optimizer"]:
        assert any(e.get("name") == name for e in events), name
    (out / "task3_profiler_table.txt").write_text(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=35))
    del model, opt
    clean()
    model, opt = make(initial)
    step(model, opt, data, CASES[3])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as offload_prof:
        step(model, opt, data, CASES[3], annotate=True)
    offload_path = out / "task3_offload_trace.json"
    offload_prof.export_chrome_trace(str(offload_path))
    offload_events = json.loads(offload_path.read_text())["traceEvents"]
    assert any(e.get("cat") == "gpu_memcpy" for e in offload_events)
    decisions = []
    minimum_throughput = results[0]["tokens_per_second"] * 0.25
    quality_passed = {q["name"] for q in quality if q["passed"]}
    # Allocator-reserved capacity plus 64 MiB guard; not a hardware-enforced quota.
    for budget_mib in [64, 128, 192, 256, 512]:
        eligible = [r for r in results
                    if r["peak_reserved_bytes"] + 64 * 2**20 <= budget_mib * 2**20
                    and r["tokens_per_second"] >= minimum_throughput
                    and r["name"] in quality_passed]
        best = min(eligible, key=lambda r: r["median_seconds"]) if eligible else None
        decisions.append(dict(budget_mib=budget_mib, reserve_guard_mib=64,
            minimum_tokens_per_second=minimum_throughput, baseline_throughput_fraction=0.25,
            eligible=[r["name"] for r in eligible], recommendation=best["name"] if best else "infeasible"))
    report = dict(status="PASS", environment=environment, config=CONFIG,
        quality_thresholds=dict(loss_relative=2e-6, update_relative_l2=2e-3, update_max_abs=2e-5),
        quality=quality, results=results, budget_decisions=decisions,
        limitations=["Synthetic residual MLP, not transformer attention or real LLM training.",
        "FP32 only; quality gate checks one deterministic update, not convergence.",
        "Benchmark synchronizes step boundaries only; independent three-sample stage diagnostics serialize stages and must not be summed to infer benchmark time.",
        "Data loading means pinned resident CPU tensor slicing, not disk or workers.",
        "Repeated steps train on identical data; warmup and measured steps update parameters.",
        "Budget recommendations use reserved allocator peaks plus 64 MiB guard, not enforced OOM tests or full device/context memory.",
        "save_on_cpu offloads all autograd-saved tensors in model, including saved parameter references; CPU RAM and PCIe bytes not measured.",
        "One GPU, one case order, nine samples; no multi-process or thermal randomization.",
        "Profiler covers independent baseline and offload steps; trace timings are excluded."])
    (out / "task3_results.json").write_text(json.dumps(report, indent=2))
    print("BUDGET_DECISIONS", json.dumps(decisions), flush=True)
    print("ALL_TESTS_PASS", flush=True)

if __name__ == "__main__":
    main()
