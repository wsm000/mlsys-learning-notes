"""硬件峰值测量 + Roofline / profiling 方向判定。

- ``ensure_peak``：实测本机带宽/算力，按设备缓存。
- ``profile_kernel_payload``：返回对齐线上 chapter15 的结构化 profiling JSON 字段；
  优先 rocprof，缺条件时退回 Roofline 模型。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CACHE_DIR = Path.home() / ".cache" / "hello-gpu"
_SKILL_SCRIPTS = Path(__file__).resolve().parent.parent / "skills" / "rocm-kernel-optimize" / "scripts"

_COPY_ELEMENTS = 128 * 1024 * 1024
_MATMUL_SIZE = 4096


def _measure_bandwidth(torch: Any, *, warmup: int, samples: int) -> float | None:
    src = torch.randn(_COPY_ELEMENTS, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    moved = 2 * _COPY_ELEMENTS * 4
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    median_ms = sorted(times)[len(times) // 2]
    return moved / (median_ms / 1000.0) / 1e9 if median_ms > 0 else None


def _measure_matmul(torch: Any, *, dtype: Any, warmup: int, samples: int) -> float | None:
    n = _MATMUL_SIZE
    try:
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(warmup):
            torch.mm(a, b)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        return None
    flops = 2.0 * n * n * n
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    median_ms = sorted(times)[len(times) // 2]
    return flops / (median_ms / 1000.0) / 1e12 if median_ms > 0 else None


def _measure_now() -> dict[str, Any] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":", 1)[0]
        name = torch.cuda.get_device_name(0)
        return {
            "name": name,
            "bandwidthGbS": _measure_bandwidth(torch, warmup=3, samples=10),
            "fp32Tflops": _measure_matmul(torch, dtype=torch.float32, warmup=3, samples=10),
            "fp16Tflops": _measure_matmul(torch, dtype=torch.float16, warmup=3, samples=10),
            "gpuArch": arch or None,
            "rocm": str(getattr(torch.version, "hip", "") or None),
        }
    except Exception:  # noqa: BLE001
        return None


def _has_peaks(profile: dict[str, Any] | None) -> bool:
    return bool(profile) and profile.get("bandwidthGbS") and (
        profile.get("fp32Tflops") or profile.get("fp16Tflops")
    )


def ensure_peak(workspace_root: Path | None = None, *, force: bool = False) -> dict[str, Any] | None:
    """拿到本机峰值画像：workspace/hardware.json → 机器缓存 → 现场实测。"""
    if workspace_root is not None and not force:
        ws_file = Path(workspace_root) / "hardware.json"
        if ws_file.is_file():
            try:
                existing = json.loads(ws_file.read_text(encoding="utf-8"))
                if _has_peaks(existing):
                    return existing
            except (OSError, json.JSONDecodeError):
                pass

    arch = None
    try:
        import torch

        if torch.cuda.is_available():
            arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":", 1)[0]
    except Exception:  # noqa: BLE001
        arch = None
    cache_file = CACHE_DIR / f"hardware-{arch or 'unknown'}.json" if arch else None
    if cache_file is not None and not force and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if _has_peaks(cached):
                if workspace_root is not None:
                    _write(workspace_root / "hardware.json", cached)
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    profile = _measure_now()
    if not _has_peaks(profile):
        return None
    if cache_file is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _write(cache_file, profile)
    if workspace_root is not None:
        _write(Path(workspace_root) / "hardware.json", profile)
    return profile


def _write(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def profile_kernel_payload(workspace: Any, kernel_source: str) -> dict[str, Any]:
    """对齐线上 chapter15 profiling 工具的结构化输出。"""
    task = workspace.task()
    cost = task.get("costModel") if isinstance(task.get("costModel"), dict) else {}
    peak = ensure_peak(workspace.root)
    rocprof = _try_rocprof(workspace, kernel_source)

    payload: dict[str, Any] = {
        "kernel": task.get("name") or task.get("description") or "candidate",
        "time_ms": None,
        "grid": None,
        "vgpr": None,
        "bandwidth_utilization_pct": None,
        "bottleneck": "unknown",
        "sm_saturation_pct": None,
        "ai": None,
        "bound": None,
        "ridge": None,
        "achieved_tflops": None,
        "suggestion": None,
        "rocprof": rocprof,
        "ok": False,
        "note": None,
    }

    if not peak or not peak.get("bandwidthGbS"):
        payload["note"] = rocprof or "缺硬件峰值；请先 measure_peak。"
        return payload

    flops = cost.get("flops") if cost else None
    bytes_ = cost.get("bytes") if cost else None
    if not (isinstance(flops, int) and isinstance(bytes_, int) and flops > 0 and bytes_ > 0):
        payload["note"] = "costModel 无效（flops/bytes 需为正整数），无法定量判定瓶颈。"
        if rocprof:
            payload["ok"] = True
            payload["note"] = f"仅有 rocprof：{rocprof}"
        return payload

    ai = flops / bytes_
    bandwidth = float(peak["bandwidthGbS"])
    compute_peak = peak.get("fp16Tflops") or peak.get("fp32Tflops")
    ridge = (float(compute_peak) * 1000 / bandwidth) if compute_peak else None
    bound = "unknown"
    if ridge is not None:
        bound = "compute-bound" if ai > ridge else "memory-bound"

    from .tools import _run_eval

    evaluation = _run_eval(workspace, kernel_source, incumbent=None)
    time_ms = evaluation.get("latencyMs") if evaluation.get("status") == "ok" else None
    util_pct = None
    achieved_tflops = None
    if isinstance(time_ms, (int, float)) and time_ms > 0:
        achieved_tflops = (flops / (time_ms / 1000.0)) / 1e12
        roofline_peak = (
            min(ai * bandwidth / 1000.0, float(compute_peak))
            if compute_peak
            else ai * bandwidth / 1000.0
        )
        if roofline_peak > 0:
            util_pct = achieved_tflops / roofline_peak * 100.0
        # 带宽利用率：有效带宽 / 峰值带宽
        achieved_gbps = (bytes_ / (time_ms / 1000.0)) / 1e9
        payload["bandwidth_utilization_pct"] = round(achieved_gbps / bandwidth * 100.0, 1)

    bottleneck = {
        "memory-bound": "DRAM bandwidth",
        "compute-bound": "compute",
        "unknown": "unknown",
    }[bound]
    suggestion = {
        "memory-bound": "优先减少数据搬运、改善合并访存、加大 block / 向量化、减少写回。",
        "compute-bound": "优先减少冗余运算、提高指令吞吐、改善并行度。",
        "unknown": "先补齐 costModel 与 measure_peak 再判定方向。",
    }[bound]

    payload.update(
        {
            "ok": evaluation.get("status") == "ok" or bool(rocprof),
            "time_ms": time_ms,
            "ai": round(ai, 3),
            "bound": bound,
            "ridge": round(ridge, 2) if ridge is not None else None,
            "bottleneck": bottleneck,
            "sm_saturation_pct": round(util_pct, 1) if util_pct is not None else None,
            "achieved_tflops": round(achieved_tflops, 3) if achieved_tflops is not None else None,
            "suggestion": suggestion,
            "note": None if evaluation.get("status") == "ok" else evaluation.get("status"),
        }
    )
    return payload


def roofline_direction(workspace: Any, kernel_source: str) -> str:
    """兼容旧调用：返回可读文本摘要。"""
    payload = profile_kernel_payload(workspace, kernel_source)
    if payload.get("note") and not payload.get("ok"):
        return f"profiling：{payload['note']}"
    lines = [
        f"算术强度 AI={payload.get('ai')} FLOP/Byte，判定为 {payload.get('bound')}"
        + (f"（ridge point≈{payload.get('ridge')}）" if payload.get("ridge") is not None else ""),
        f"瓶颈信号：{payload.get('bottleneck')}；"
        f"带宽利用率≈{payload.get('bandwidth_utilization_pct')}%；"
        f"吞吐利用率≈{payload.get('sm_saturation_pct')}%。",
        f"优化侧重：{payload.get('suggestion')}",
    ]
    if payload.get("rocprof"):
        lines.append(f"rocprof 实测：{payload['rocprof']}")
    return "\n".join(lines)


def _try_rocprof(workspace: Any, kernel_source: str) -> str | None:
    """尝试用 skill 脚本跑 rocprofv3 PMC；不可用返回 None。"""
    script = _SKILL_SCRIPTS / "profile.py"
    if not script.is_file():
        return None
    candidate = workspace.write_candidate(kernel_source)
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--task",
                str(workspace.task_path),
                "--candidate",
                str(candidate),
                "--reference",
                str(workspace.reference_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:1500]
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        candidate.unlink(missing_ok=True)
