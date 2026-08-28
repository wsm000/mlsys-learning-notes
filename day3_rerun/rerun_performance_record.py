#!/usr/bin/env python3
"""Day 3: rerun Vector Add and write a local performance record."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
from pathlib import Path

N_DEFAULT = 16_777_216
BLOCK_DEFAULT = 256
WARMUP_DEFAULT = 20
REPEAT_DEFAULT = 100
TRACE_WARMUP = 5
TRACE_REPEAT = 10
BYTES_PER_ELEMENT = 12
FLOPS_PER_ELEMENT = 1
VALID_ARCHES = {"gfx1100", "gfx1151", "gfx1201"}


def command_text(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run(cmd, cwd, timeout=600, required=False, show=True):
    print(f"\n$ {command_text(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        print(f"命令不可用: {cmd[0]}")
        if required:
            raise RuntimeError(f"命令不可用: {cmd[0]}") from exc
        return None
    except subprocess.TimeoutExpired as exc:
        print(f"命令超时: {cmd[0]}")
        if required:
            raise RuntimeError("外部命令超时") from exc
        return None
    if show and result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(f"stderr: {result.stderr.rstrip()}")
    print(f"returncode={result.returncode}")
    if required and result.returncode != 0:
        raise RuntimeError(f"命令失败: {command_text(cmd)}")
    return result


def find_repo(explicit):
    starts = []
    if explicit:
        starts.append(Path(explicit).expanduser().resolve())
    if os.environ.get("HELLO_GPU_REPO"):
        starts.append(Path(os.environ["HELLO_GPU_REPO"]).expanduser().resolve())
    starts.append(Path.cwd().resolve())
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    visited = set()
    for start in starts:
        current = start if start.is_dir() else start.parent
        for candidate in (current, *current.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            if ((candidate / "code/part1-profiling/chapter6/vector_add.hip").is_file()
                    and (candidate / "notebooks/part1-profiling/chapter7.ipynb").is_file()):
                return candidate
    raise RuntimeError("找不到 hello-gpu 根目录；请在仓库内运行或设置 HELLO_GPU_REPO")


def detect_arch(requested):
    requested = (requested or os.environ.get("HELLO_GPU_ARCH", "")).strip()
    if requested:
        if requested not in VALID_ARCHES:
            raise RuntimeError(f"arch 必须是 {sorted(VALID_ARCHES)}")
        return requested, "--arch/HELLO_GPU_ARCH（编译 target，不单独证明硬件）"
    result = run(["rocminfo"], Path.cwd(), timeout=30, required=True, show=False)
    names = []
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"\s*Name:\s*(gfx[0-9a-z]+)\s*", line)
        if match:
            names.append(match.group(1))
    names = sorted(set(names))
    if len(names) != 1:
        raise RuntimeError(f"rocminfo 未检测到唯一 GPU arch: {names or '<none>'}")
    if names[0] not in VALID_ARCHES:
        raise RuntimeError(f"检测到 {names[0]}，脚本支持 {sorted(VALID_ARCHES)}")
    return names[0], "rocminfo"


def grid_for(kernel, n, block, stride=None):
    """Return rocprof-style Grid_Size: total launched work-items, not blocks."""
    if kernel == "linecross":
        per_block = (block // 32) * 32 * stride
        blocks = (n + per_block - 1) // per_block
    else:
        blocks = (n + block - 1) // block
    return blocks * block


def run_benchmark(binary, out_dir, label, kernel, n, block, stride, warmup, repeat):
    output = out_dir / f"{label}.json"
    output.unlink(missing_ok=True)
    cmd = [str(binary), "--kernel", kernel, "--size", str(n), "--block", str(block)]
    if stride is not None:
        cmd += ["--stride", str(stride)]
    cmd += ["--warmup", str(warmup), "--repeat", str(repeat), "--output-json", str(output)]
    run(cmd, out_dir, required=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"没有产生新 JSON: {output}")
    data = json.loads(output.read_text(encoding="utf-8"))
    if data.get("correctness") != "OK":
        raise RuntimeError(f"{label} correctness={data.get('correctness')}")
    min_ms = float(data["min_ms"])
    median_ms = float(data["median_ms"])
    bw_min = float(data["eff_bandwidth_gbs"])
    bw_median = BYTES_PER_ELEMENT * n / (median_ms * 1e-3) / 1e9
    ai = FLOPS_PER_ELEMENT / BYTES_PER_ELEMENT
    item = dict(data)
    item.update({
        "label": label,
        "kernel": kernel,
        "stride": stride,
        "grid_size": grid_for(kernel, n, block, stride),
        "json_path": str(output),
        "ai": ai,
        "effective_bw_min_gbps": bw_min,
        "effective_bw_median_gbps": bw_median,
        "performance_min_tflops": ai * bw_min / 1000.0,
    })
    print(f"{label}: PASS | min={min_ms:.4f} ms | median={median_ms:.4f} ms | "
          f"BW@min={bw_min:.2f} GB/s | Grid={item['grid_size']}")
    return item


def trace_columns(fields):
    normalized = {re.sub(r"[^a-z0-9]", "", f.lower()): f for f in fields if f}
    def get(*names):
        for name in names:
            value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
            if value:
                return value
        return None
    return {"kernel": get("Kernel_Name", "Kernel Name", "KernelName"),
            "start": get("Start_Timestamp", "Start Timestamp", "StartTimestamp"),
            "end": get("End_Timestamp", "End Timestamp", "EndTimestamp"),
            "grid": get("Grid_Size", "Grid Size", "GridSize"),
            "vgpr": get("VGPR_Count", "VGPR Count", "VGPRCount"),
            "sgpr": get("SGPR_Count", "SGPR Count", "SGPRCount")}


def parse_trace(path):
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            cols = trace_columns(reader.fieldnames or [])
            if not all(cols[k] for k in ("kernel", "start", "end")):
                return None
            rows = []
            for row in reader:
                name = (row.get(cols["kernel"]) or "").strip()
                if not name:
                    continue
                try:
                    duration = (float(row[cols["end"]]) - float(row[cols["start"]])) / 1000.0
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append({"name": name, "us": duration,
                             "grid": (row.get(cols["grid"]) or "unavailable").strip() or "unavailable",
                             "vgpr": (row.get(cols["vgpr"]) or "unavailable").strip() or "unavailable",
                             "sgpr": (row.get(cols["sgpr"]) or "unavailable").strip() or "unavailable"})
    except (OSError, UnicodeError, csv.Error):
        return None
    if not rows:
        return None
    measured = rows[TRACE_WARMUP:] if len(rows) > TRACE_WARMUP else rows
    values = [row["us"] for row in measured]
    return {"csv": str(path), "kernel_names": sorted({r["name"] for r in measured}),
            "dispatches_total": len(rows), "dispatches_after_warmup": len(measured),
            "min_us": min(values), "median_us": statistics.median(values),
            "grid": measured[0]["grid"], "vgpr": measured[0]["vgpr"], "sgpr": measured[0]["sgpr"]}


def run_profile(binary, out_dir, label, kernel, n, block, stride):
    profile_dir = out_dir / "rocprof" / label
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rocprofv3", "--kernel-trace", "--output-directory", str(profile_dir),
           "--output-file", label, "--output-format", "csv", "--", str(binary),
           "--kernel", kernel, "--size", str(n), "--block", str(block)]
    if stride is not None:
        cmd += ["--stride", str(stride)]
    cmd += ["--warmup", str(TRACE_WARMUP), "--repeat", str(TRACE_REPEAT)]
    result = run(cmd, out_dir, required=False)
    if result is None or result.returncode != 0:
        return None
    summaries = []
    for path in sorted(profile_dir.rglob("*.csv")):
        if path.stat().st_size:
            parsed = parse_trace(path)
            if parsed:
                summaries.append(parsed)
    for summary in summaries:
        print(f"{label}: Kernel_Name={','.join(summary['kernel_names'])} | "
              f"dispatches={summary['dispatches_after_warmup']} | min={summary['min_us']:.3f} us | "
              f"median={summary['median_us']:.3f} us | Grid={summary['grid']} | "
              f"VGPR={summary['vgpr']} | SGPR={summary['sgpr']}")
    return {"label": label, "summaries": summaries} if summaries else None


def make_record(repo, run_dir, arch, arch_source, results, profiles, args):
    coalesced = results["coalesced"]
    stride1 = results["linecross_stride1"]
    stride32 = results["linecross_stride32"]
    ratio1 = stride1["min_ms"] / coalesced["min_ms"]
    ratio32 = stride32["min_ms"] / coalesced["min_ms"]
    lines = [
        "# 性能记录：Vector Add（本机重新测量）", "",
        f"- 记录时间：{dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- OS：{platform.platform()}",
        f"- GPU：{coalesced.get('gpu', 'JSON 未提供')}（arch={coalesced.get('arch', arch)}）",
        f"- 编译 target：{arch}；来源：{arch_source}",
        f"- kernel / 输入：Vector Add / {args.size} float32",
        f"- 计时协议：warmup={args.warmup}，repeat={args.repeat}，HIP GPU Event，保留 min/median/avg",
        f"- 运行目录：`{run_dir}`", "",
        "## 关键结果（本机实测）", "",
        "| 版本 | Grid Size | min (ms) | median (ms) | 有效带宽@min (GB/s) | 有效带宽@median (GB/s) | AI | 性能@min (TFLOPS) |",
        "|------|----------:|---------:|------------:|-------------------:|-----------------------:|---:|------------------:|",
    ]
    for key in ("coalesced", "linecross_stride1", "linecross_stride32"):
        x = results[key]
        label = "coalesced" if x["kernel"] == "coalesced" else f"linecross stride={x['stride']}"
        lines.append(f"| {label} | {x['grid_size']:,} | {x['min_ms']:.4f} | {x['median_ms']:.4f} | "
                     f"{x['effective_bw_min_gbps']:.2f} | {x['effective_bw_median_gbps']:.2f} | "
                     f"{x['ai']:.4f} | {x['performance_min_tflops']:.5f} |")
    lines += [
        "", f"- linecross stride=1 / coalesced 时间比：**{ratio1:.2f}×**。",
        f"- linecross stride=32 / coalesced 时间比：**{ratio32:.2f}×**。",
        "- 有效带宽按每元素 12 B 的算法口径计算，不等于实际 DRAM 事务量。", "",
        "## 当前判断", "",
        f"- **实测证据**：coalesced={coalesced['min_ms']:.4f} ms，linecross stride=32={stride32['min_ms']:.4f} ms；Grid Size 为 {coalesced['grid_size']:,} 和 {stride32['grid_size']:,}。",
        f"- **实测证据**：有效带宽为 {coalesced['effective_bw_min_gbps']:.2f} 和 {stride32['effective_bw_min_gbps']:.2f} GB/s。",
        "- **工程推测**：stride=32 地址更分散，可能破坏 wavefront 内访存合并。",
        "- **尚未验证**：当前实验同时改变每线程循环次数和 Grid Size，不能把全部差距归因于访存合并。", "",
        "## Roofline", "",
        f"- AI = 1/12 = {FLOPS_PER_ELEMENT / BYTES_PER_ELEMENT:.4f} FLOP/Byte，理论上属于 memory-bound 一侧。",
    ]
    if args.memory_ceiling is None:
        lines.append("- 未提供本机 memory ceiling，不计算带宽利用率；不要把其他 GPU 的 510 GB/s 当成本机上限。")
    else:
        lines.append(f"- 本机 memory ceiling={args.memory_ceiling:.2f} GB/s；coalesced 利用率={coalesced['effective_bw_min_gbps']/args.memory_ceiling:.1%}，linecross stride=32 利用率={stride32['effective_bw_min_gbps']/args.memory_ceiling:.1%}。")
        if args.compute_ceiling is not None:
            ridge = args.compute_ceiling / (args.memory_ceiling / 1000.0)
            lines.append(f"- 本机 compute ceiling={args.compute_ceiling:.2f} TFLOPS，拐点约 {ridge:.2f} FLOP/Byte。")
    lines += ["", "## rocprofv3 Kernel Trace", ""]
    if profiles:
        lines += ["以下统计跳过前 5 个 warmup dispatch。", "", "| 配置 | Kernel_Name | dispatches | min (μs) | median (μs) | Grid | VGPR | SGPR |", "|------|-------------|-----------:|---------:|------------:|------|-----:|-----:|"]
        for profile in profiles:
            for x in profile["summaries"]:
                lines.append(f"| {profile['label']} | {', '.join(x['kernel_names'])} | {x['dispatches_after_warmup']} | {x['min_us']:.3f} | {x['median_us']:.3f} | {x['grid']} | {x['vgpr']} | {x['sgpr']} |")
    else:
        lines.append("没有得到可解析 trace；benchmark 记录有效，但不能作为 Kernel Trace 打卡证据。")
    lines += [
        "", "## 下一步：单变量实验", "",
        "重新编译公平对照 kernel：固定 N、block/Grid、每线程循环次数和总计算量，只改变索引公式：", "",
        "```text", "连续版：i = tile_base + j * 32 + lane", "分散版：i = tile_base + lane * 32 + j", "```", "",
        "用相同 warmup/repeat 重测 min/median、有效带宽和 trace 时间。当前脚本没有伪装成已经完成这个公平实验。", "",
        f"原始 JSON 和 trace：`{run_dir}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--size", type=int, default=N_DEFAULT)
    parser.add_argument("--block", type=int, default=BLOCK_DEFAULT)
    parser.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    parser.add_argument("--repeat", type=int, default=REPEAT_DEFAULT)
    parser.add_argument("--memory-ceiling", type=float, default=None)
    parser.add_argument("--compute-ceiling", type=float, default=None)
    parser.add_argument("--no-profile", action="store_true")
    args, _ = parser.parse_known_args()
    if args.size <= 0 or args.block <= 0 or args.block % 32 or args.warmup < 0 or args.repeat <= 0:
        raise ValueError("size/block/warmup/repeat 参数不合法")

    repo = find_repo(args.repo)
    chapter6 = repo / "code/part1-profiling/chapter6"
    arch, arch_source = detect_arch(args.arch)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = chapter6 / "logs" / f"day3_rerun_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    binary = run_dir / f"vector_add_bench_{arch}"
    run(["hipcc", f"--offload-arch={arch}", "-O3", str(chapter6 / "vector_add.hip"), "-o", str(binary)], run_dir, required=True)

    results = {}
    for label, kernel, stride in (("coalesced", "coalesced", None), ("linecross_stride1", "linecross", 1), ("linecross_stride32", "linecross", 32)):
        results[label] = run_benchmark(binary, run_dir, label, kernel, args.size, args.block, stride, args.warmup, args.repeat)

    profiles = []
    if not args.no_profile:
        version = run(["rocprofv3", "--version"], run_dir, timeout=60)
        if version is not None and version.returncode == 0:
            for label, kernel, stride in (("coalesced", "coalesced", None), ("linecross_stride32", "linecross", 32)):
                item = run_profile(binary, run_dir, label, kernel, args.size, args.block, stride)
                if item:
                    profiles.append(item)
        else:
            print("rocprofv3 不可用，跳过 trace")

    record = make_record(repo, run_dir, arch, arch_source, results, profiles, args)
    record_path = run_dir / "performance_record.md"
    record_path.write_text(record, encoding="utf-8")
    # This JSON is deliberately shaped like chapter7 cell 13's
    # chapter6_results dictionary, so it can be loaded directly there.
    chapter7_data = {
        key: {
            "time_ms": results[key]["min_ms"],
            "median_time_ms": results[key]["median_ms"],
            "effective_bw_gbps": results[key]["effective_bw_min_gbps"],
            "ai": results[key]["ai"],
            "performance_tflops": results[key]["performance_min_tflops"],
            "grid_size": results[key]["grid_size"],
            "correctness": results[key]["correctness"],
        }
        for key in ("coalesced", "linecross_stride32")
    }
    chapter7_path = run_dir / "chapter7_results.json"
    chapter7_path.write_text(json.dumps(chapter7_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 完成 ===\n性能记录: {record_path}\nchapter7 数据: {chapter7_path}")


if __name__ == "__main__":
    main()
