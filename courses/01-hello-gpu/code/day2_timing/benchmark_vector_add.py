#!/usr/bin/env python3
"""Measure one fixed ROCm/PyTorch Vector Add with explicit timing boundaries."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections.abc import Sequence


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    sorted_values = sorted(values)
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return sorted_values[rank - 1]


def print_statistics(label: str, samples_ms: Sequence[float], bytes_per_operation: int) -> None:
    mean_ms = statistics.fmean(samples_ms)
    median_ms = statistics.median(samples_ms)
    min_ms = min(samples_ms)
    max_ms = max(samples_ms)
    p95_ms = percentile_nearest_rank(samples_ms, 0.95)
    stddev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    median_logical_gbps = bytes_per_operation / (median_ms / 1_000.0) / 1e9

    print(label)
    print(f"  count: {len(samples_ms)}")
    print(f"  mean_ms: {mean_ms:.6f}")
    print(f"  median_ms: {median_ms:.6f}")
    print(f"  min_ms: {min_ms:.6f}")
    print(f"  max_ms: {max_ms:.6f}")
    print(f"  p95_ms: {p95_ms:.6f}")
    print(f"  stddev_ms: {stddev_ms:.6f}")
    print(f"  median_logical_GBps: {median_logical_gbps:.3f}")


def synchronize(torch: object, device: object) -> None:
    torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare misleading CPU timing with warmup + GPU Event timing."
    )
    parser.add_argument("--n", type=positive_int, default=1 << 24)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repeat", type=positive_int, default=100)
    parser.add_argument("--inner", type=positive_int, default=20)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Activate the ROCm Python environment before running.")
        return 2

    hip_version = getattr(torch.version, "hip", None)
    if hip_version is None:
        print("ERROR: This script requires a PyTorch build with ROCm/HIP support.")
        return 2
    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot see a ROCm GPU through torch.cuda.")
        return 2
    if args.device < 0 or args.device >= torch.cuda.device_count():
        print(
            f"ERROR: device index {args.device} is unavailable; "
            f"visible device count is {torch.cuda.device_count()}."
        )
        return 2

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32
    bytes_per_operation = args.n * torch.empty((), dtype=dtype).element_size() * 3

    print("=== Vector Add Timing Protocol ===")
    print(f"device: {torch.cuda.get_device_name(args.device)}")
    print(f"torch_version: {torch.__version__}")
    print(f"hip_version: {hip_version}")
    print("operation: torch.add(a, b, out=c)")
    print(f"N: {args.n}")
    print("dtype: float32")
    print("fixed_input_a: all 1.0")
    print("fixed_input_b: all 2.0")
    print("output_buffer: allocated once before measurement")
    print(f"warmup: {args.warmup}")
    print(f"repeat: {args.repeat}")
    print(f"inner: {args.inner}")
    print("measurement_boundary: only torch.add(a, b, out=c)")
    print("synchronization: synchronize before CPU timing and before Event readout")
    print("gpu_event_stream: current default stream")
    print()

    try:
        a = torch.ones(args.n, dtype=dtype, device=device)
        b = torch.full((args.n,), 2.0, dtype=dtype, device=device)
        c = torch.empty_like(a)
        synchronize(torch, device)

        # This is intentionally the first Vector Add: no benchmark warmup precedes it.
        start_cpu = time.perf_counter()
        torch.add(a, b, out=c)
        synchronize(torch, device)
        single_once_ms = (time.perf_counter() - start_cpu) * 1_000.0

        if not bool(torch.all(c == 3.0).item()):
            print("ERROR: Vector Add correctness check failed after the single run.")
            return 1

        # Intentionally invalid for Kernel time: the end timestamp is before GPU completion.
        synchronize(torch, device)
        start_enqueue = time.perf_counter()
        torch.add(a, b, out=c)
        enqueue_only_ms = (time.perf_counter() - start_enqueue) * 1_000.0
        synchronize(torch, device)  # Flush the queued work outside the invalid timing boundary.

        for _ in range(args.warmup):
            torch.add(a, b, out=c)
        synchronize(torch, device)

        event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(args.repeat):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(args.inner):
                torch.add(a, b, out=c)
            end_event.record()
            event_pairs.append((start_event, end_event))

        synchronize(torch, device)
        event_samples_ms = [
            start_event.elapsed_time(end_event) / args.inner
            for start_event, end_event in event_pairs
        ]
    except RuntimeError as error:
        print(f"ERROR: ROCm/PyTorch operation failed: {error}")
        return 1

    print("=== Results ===")
    print("single_once_no_benchmark_warmup_cpu_wall_ms")
    print(f"  value_ms: {single_once_ms:.6f}")
    print("  meaning: one add plus final CPU/GPU synchronization; not kernel-only.")
    print("incorrect_cpu_enqueue_only_ms")
    print(f"  value_ms: {enqueue_only_ms:.6f}")
    print("  status: INVALID FOR KERNEL TIME")
    print("  meaning: CPU submitted asynchronous work but did not wait in the boundary.")
    print_statistics(
        "warmup_gpu_event_per_torch_add_ms", event_samples_ms, bytes_per_operation
    )
    print(
        "  meaning: primary result; per-add average from an Event-bounded batch "
        "of inner torch.add calls after warmup."
    )
    print()
    print(
        "CONCLUSION: The warmup + GPU Event distribution is more credible because "
        "it excludes the first-run effect and uses GPU timestamps around completed "
        "Vector Add work; CPU enqueue-only timing is not Kernel time."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
