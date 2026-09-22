"""Render measured JSON/log evidence, explicitly not a browser/terminal screenshot."""
import argparse, json, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--evidence", required=True)
a = ap.parse_args()
root = Path(a.evidence)
r = json.loads((root / "task3_results.json").read_text())
assert r["status"] == "PASS" and all(q["passed"] for q in r["quality"])
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
font = ImageFont.truetype(font_path, 21)
title_font = ImageFont.truetype(font_path, 31)

def render(name, title, sections):
    lines = []
    for heading, body in sections:
        lines.append((heading, "#7dd3fc"))
        for line in body:
            for wrapped in textwrap.wrap(line, width=111, replace_whitespace=False, drop_whitespace=False) or [""]:
                lines.append((wrapped, "#e2e8f0"))
        lines.append(("", "#e2e8f0"))
    height = 176 + len(lines)*32 + 95
    image = Image.new("RGB", (1500, height), "#0f172a")
    d = ImageDraw.Draw(image)
    d.rectangle((0,0,1500,8), fill="#38bdf8")
    d.text((38,29), title, font=title_font, fill="#f8fafc")
    d.text((38,80), "DataWhale | llm-algo-leetcode | Issue #169 | vm-60", font=font, fill="#94a3b8")
    d.text((38,113), "MEASURED EVIDENCE RENDER - not a terminal/browser screenshot", font=font, fill="#fbbf24")
    y=174
    for line,color in lines:
        d.text((38,y),line,font=font,fill=color); y+=32
    d.text((38,y+8), "Source: task3_results.json + original logs / trace | no simulated measurements", font=font,fill="#94a3b8")
    image.save(root/name)
    print(name, image.size)

c=r["config"]; env=r["environment"]; rows=r["results"]
quality=["Case                        loss          rel.loss       rel.update L2    max.update diff   check"]
for q in r["quality"]:
    quality.append(f"{q['name']:<27} {q['loss']:.8f}    {q['loss_relative']:.2e}       {q['update_relative_l2']:.2e}         {q['update_max_abs']:.2e}     PASS")
render("01_baseline_quality.png", "TASK 3 / 4.1 - Reproducible baseline & checks", [
 ("Environment",[f"GPU: {env['gpu']} | PyTorch {env['torch']} | CUDA {env['cuda']}","Host: vm-60 | Python "+env["python"]]),
 ("Fixed workload",[json.dumps(c,sort_keys=True),"Synthetic residual MLP; same initial weights, inputs and AdamW settings.","Gradient accumulation: micro_batch=1 x 4; other cases: micro_batch=4 x 1.","Warm-up before peak reset. Independent stage diagnostics and profiler pass."]),
 ("One-step numerical equivalence", quality + ["Thresholds: "+json.dumps(r["quality_thresholds"]),"This is numerical equivalence, NOT validation accuracy or convergence."]),
 ("Reproducibility",["Source SHA256: "+env["source_sha256"],"Data SHA256:   "+env["data_sha256"],"ALL_TESTS_PASS (see task3_run.log)"])])

summary=["Case                        alloc MiB  reserved MiB   median ms     range ms          tokens/s"]
stages=["Case                        data ms     H2D ms    forward ms  backward ms optimizer ms"]
for x in rows:
    summary.append(f"{x['name']:<27} {x['peak_allocated_bytes']/2**20:>9.2f} {x['peak_reserved_bytes']/2**20:>12.2f} {x['median_seconds']*1000:>11.3f} {x['min_seconds']*1000:>7.3f}-{x['max_seconds']*1000:<7.3f} {x['tokens_per_second']:>11,.0f}")
    s=x["stages_median_seconds"]
    stages.append(f"{x['name']:<27} "+" ".join(f"{s[k]*1000:>11.3f}" for k in ["data_loading","h2d","forward","backward","optimizer"]))
l=rows[0]["ledger"]
render("02_strategy_budget.png", "TASK 3 / 4.2 - Memory, throughput & budget", [
 ("Independent end-to-end benchmark (median of 9 updates)", summary),
 ("Separate synchronized stage diagnostics (do not sum to production throughput)",stages),
 ("Resident tensor ledger",[" | ".join(f"{k}: {v/2**20:.3f} MiB" for k,v in l.items() if k in ["parameter_bytes","gradient_bytes","adam_cuda_bytes"]),"Residual peak includes inputs, activations, temporary buffers; not pure activation size."]),
 ("Budget sensitivity (reserved + 64 MiB guard; throughput >=25% baseline)",[json.dumps(v,sort_keys=True) for v in r["budget_decisions"]]),
 ("Boundaries",["Quality gates are mandatory. Memory savings alone do not imply a better strategy.","save_on_cpu may also move saved weight references. Single GPU: communication N/A.","Synthetic workload; no quantization or shortened-sequence quality claims."])])

trace=json.loads((root/"task3_cpu_cuda_trace.json").read_text())
events=trace["traceEvents"]
from collections import defaultdict
stats=defaultdict(lambda:[0,0.0])
for e in events:
    if e.get("ph") != "X" or "dur" not in e: continue
    cat=e.get("cat","")
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        key=(cat,e.get("name","unknown")); stats[key][0]+=1;stats[key][1]+=e["dur"]
top=sorted(stats.items(),key=lambda v:v[1][1],reverse=True)[:12]
tracelines=[]
for (cat,name),(count,duration) in top:
    tracelines.append(f"{cat:<12} count={count:<5} sum={duration/1000:>9.3f} ms | {name[:76]}")
cats=defaultdict(lambda:[0,0.0])
for (cat,name),(count,duration) in stats.items(): cats[cat][0]+=count;cats[cat][1]+=duration
off_events=json.loads((root/"task3_offload_trace.json").read_text())["traceEvents"]
transfer_comparison=[]
for label,evs in [("baseline",events),("offload",off_events)]:
    transfers=[e for e in evs if e.get("ph")=="X" and e.get("cat")=="gpu_memcpy"]
    transfer_comparison.append(f"{label}: {len(transfers)} memcpy events, sum={sum(e.get('dur',0) for e in transfers)/1000:.3f} ms")
assert cats["kernel"][0] > 0 and cats["gpu_memcpy"][0] > 0
render("03_profiler_evidence.png", "TASK 3 - Real CPU/CUDA profiler evidence", [
 ("Trace source",["task3_cpu_cuda_trace.json | independent profiler pass",f"Total trace events: {len(events):,}","CPU record_function regions label data_loading / H2D / forward / backward / optimizer."]),
 ("GPU event categories",[f"{cat}: {v[0]} events, summed duration {v[1]/1000:.3f} ms" for cat,v in cats.items()]),
 ("Baseline vs offload transfer evidence",transfer_comparison),
 ("Top GPU event names by summed duration",tracelines),
 ("Interpretation rules",["Kernel time identifies expensive operations, not proof of compute/bandwidth saturation.","Memcpy events provide direct evidence of transfer work; inspect overlap in the full trace.","CPU synchronization may wait on already-accounted GPU work; do not add both as extra cost.","Aggregated durations can overlap across streams and are NOT wall-clock step time.","Read task3_profiler_table.txt; load JSON in a Chrome trace viewer for the full timeline."]),
 ("Check",["ALL_TESTS_PASS | trace contains actual GPU kernels and transfer events.","No distributed communication: this experiment uses one GPU."])])
