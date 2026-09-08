"""GPU worker for the active declarative evaluator and historical v1 fixture."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import platform
import statistics
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_spec import TaskSpec, TaskSpecV2, load_task


_DTYPE_BYTES = {
    "bool": 1,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}


class _GuardedTorch:
    """Forward torch while bounding host allocations made by a return ABI."""

    def __init__(
        self,
        budget_bytes: int,
        *,
        return_specs: tuple[Any, ...] | None = None,
    ) -> None:
        self._budget_bytes = budget_bytes
        self._used_bytes = 0
        self._return_specs = return_specs
        self._return_buffers: list[torch.Tensor] = []
        self._allocation_index = 0
        self._poison_generation = -1
        self._poison_explicit_empty = False

    def reset(self) -> None:
        self._used_bytes = 0
        self._allocation_index = 0
        if self._return_specs is None:
            self._poison_generation += 1

    @staticmethod
    def _poison(tensor: torch.Tensor, generation: int) -> None:
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        elif tensor.dtype == torch.bool:
            tensor.fill_(generation % 2 == 0)
        else:
            info = torch.iinfo(tensor.dtype)
            tensor.fill_((info.max, info.min, 0)[generation % 3])

    def prepare_return_buffers(self) -> None:
        """Allocate once and poison outside timing for a declared return ABI."""

        if self._return_specs is None:
            return
        if not self._return_buffers:
            self._return_buffers = [
                torch.empty(
                    spec.shape,
                    device="cuda",
                    dtype=getattr(torch, spec.dtype),
                )
                for spec in self._return_specs
            ]
        self._poison_generation += 1
        for tensor in self._return_buffers:
            self._poison(tensor, self._poison_generation)

    def _claim_return_buffer(
        self,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> torch.Tensor | None:
        if self._return_specs is None:
            return None
        if self._allocation_index >= len(self._return_buffers):
            raise RuntimeError("candidate allocated more tensors than the declared return ABI")
        buffer = self._return_buffers[self._allocation_index]
        self._allocation_index += 1
        if tuple(buffer.shape) != shape or buffer.dtype != dtype:
            raise RuntimeError(
                "candidate return allocation does not match the frozen contract: "
                f"requested shape={shape}, dtype={dtype}; "
                f"expected shape={tuple(buffer.shape)}, dtype={buffer.dtype}"
            )
        return buffer

    def finish_return_allocations(self) -> None:
        if (
            self._return_specs is not None
            and self._allocation_index != len(self._return_buffers)
        ):
            raise RuntimeError(
                "candidate did not allocate every tensor in the declared return ABI"
            )

    def _charge(self, elements: int, dtype: Any) -> None:
        dtype_name = str(dtype).removeprefix("torch.")
        element_bytes = _DTYPE_BYTES.get(dtype_name)
        if element_bytes is None:
            raise RuntimeError(f"candidate allocation uses unsupported dtype: {dtype}")
        allocation_bytes = elements * element_bytes
        if allocation_bytes > self._budget_bytes - self._used_bytes:
            raise RuntimeError(
                "candidate host allocations exceed the declared return-tensor budget "
                f"({self._used_bytes + allocation_bytes} > {self._budget_bytes} bytes)"
            )
        self._used_bytes += allocation_bytes

    @staticmethod
    def _shape_from_empty_args(args: tuple[Any, ...]) -> tuple[int, ...]:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            raw_shape = tuple(args[0])
        else:
            raw_shape = args
        if any(
            not isinstance(axis, int) or isinstance(axis, bool) or axis < 0
            for axis in raw_shape
        ):
            raise RuntimeError("candidate allocation shape must contain non-negative integers")
        return tuple(raw_shape)

    @staticmethod
    def _require_cuda_device(kwargs: dict[str, Any]) -> None:
        requested = kwargs.get("device")
        if requested is None:
            return
        if torch.device(requested).type != "cuda":
            raise RuntimeError("generated code may not allocate tensors outside the GPU")

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._require_cuda_device(kwargs)
        shape_args = args
        if not args and "size" in kwargs:
            shape_args = (kwargs["size"],)
        shape = self._shape_from_empty_args(shape_args)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.get_default_dtype()
        prepared = self._claim_return_buffer(shape, dtype)
        if prepared is not None:
            return prepared
        self._charge(math.prod(shape), dtype)
        result = torch.empty(*args, **kwargs)
        if self._poison_explicit_empty:
            self._poison(result, self._poison_generation)
        return result

    def empty_like(self, source: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self._require_cuda_device(kwargs)
        dtype = kwargs.get("dtype", source.dtype)
        prepared = self._claim_return_buffer(tuple(source.shape), dtype)
        if prepared is not None:
            return prepared
        self._charge(source.numel(), dtype)
        result = torch.empty_like(source, **kwargs)
        if self._poison_explicit_empty:
            self._poison(result, self._poison_generation)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)


class _GuardedReferenceTorch(_GuardedTorch):
    """Bound explicit reference constructors while forwarding tensor math."""

    def __init__(self, budget_bytes: int) -> None:
        super().__init__(budget_bytes)
        self._poison_explicit_empty = True

    @staticmethod
    def _arange_length(*args: Any, **kwargs: Any) -> int:
        if not args and "end" in kwargs:
            args = (
                *((kwargs.get("start", 0),) if "start" in kwargs else ()),
                kwargs["end"],
            )
        if not 1 <= len(args) <= 3:
            raise RuntimeError("reference torch.arange requires one to three arguments")
        if len(args) == 1:
            start, end = 0, args[0]
        else:
            start, end = args[0], args[1]
        step = args[2] if len(args) == 3 else kwargs.get("step", 1)
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in (start, end, step)
        ):
            raise RuntimeError("reference torch.arange bounds must be numeric scalars")
        if not all(math.isfinite(float(value)) for value in (start, end, step)):
            raise RuntimeError("reference torch.arange bounds must be finite")
        if step == 0:
            raise RuntimeError("reference torch.arange step must be non-zero")
        span = (end - start) / step
        return max(0, math.ceil(span))

    def arange(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._require_cuda_device(kwargs)
        elements = self._arange_length(*args, **kwargs)
        dtype = kwargs.get("dtype")
        if dtype is None:
            numeric_args = (*args, kwargs.get("step", 1))
            dtype = (
                torch.get_default_dtype()
                if any(isinstance(value, float) for value in numeric_args)
                else torch.int64
            )
        self._charge(elements, dtype)
        return torch.arange(*args, **kwargs)

    def _like(self, function, source: torch.Tensor, *args: Any, **kwargs: Any):
        self._require_cuda_device(kwargs)
        dtype = kwargs.get("dtype", source.dtype)
        self._charge(source.numel(), dtype)
        return function(source, *args, **kwargs)

    def zeros_like(self, source: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self._like(torch.zeros_like, source, **kwargs)

    def ones_like(self, source: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self._like(torch.ones_like, source, **kwargs)

    def full_like(
        self, source: torch.Tensor, fill_value: Any, **kwargs: Any
    ) -> torch.Tensor:
        return self._like(torch.full_like, source, fill_value, **kwargs)


def _load_candidate(
    path: Path,
    entrypoint: str,
    module_name: str,
    *,
    allocation_budget_bytes: int | None = None,
    return_specs: tuple[Any, ...] | None = None,
):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create candidate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = getattr(module, entrypoint, None)
    if not callable(launch):
        raise TypeError(f"candidate entrypoint is not callable: {entrypoint}")
    if allocation_budget_bytes is not None:
        guard = _GuardedTorch(
            allocation_budget_bytes,
            return_specs=return_specs,
        )
        if "torch" in module.__dict__:
            module.__dict__["torch"] = guard

        raw_launch = launch

        def launch(*args: Any):
            guard.reset()
            result = raw_launch(*args)
            guard.finish_return_allocations()
            return result

        launch.prepare_return_buffers = guard.prepare_return_buffers  # type: ignore[attr-defined]

    return launch


def _runtime_environment() -> dict[str, str]:
    properties = torch.cuda.get_device_properties(0)
    gpu_arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    torch_version = getattr(torch, "version", None)
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        os_release = {}
    return {
        "device": torch.cuda.get_device_name(0),
        "gpuArch": gpu_arch,
        "platform": platform.system(),
        "platformRelease": platform.release(),
        "distro": os_release.get("ID", ""),
        "distroVersion": os_release.get("VERSION_ID", ""),
        "rocm": str(getattr(torch_version, "hip", "") or ""),
        "torch": torch.__version__,
        "triton": triton.__version__,
    }


def _declared_peak_bytes(task: TaskSpecV2) -> int:
    declared = 0
    inputs = 0
    observable = 0
    for tensor in task.tensors:
        if tensor.kind == "scalar" or tensor.shape is None or tensor.dtype is None:
            continue
        tensor_bytes = math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
        declared += tensor_bytes
        if tensor.kind == "input":
            inputs += tensor_bytes
        if tensor.kind in {"output", "return"}:
            observable += tensor_bytes
    return 2 * (declared + inputs) + observable


def _apply_cuda_allocation_limit(task: TaskSpecV2) -> None:
    """Reserve headroom and bound PyTorch/ROCm caching-allocator growth."""

    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    predicted_peak = _declared_peak_bytes(task)
    allowed_bytes = min(
        total_memory // 2,
        max(predicted_peak + (512 << 20), 1 << 30),
    )
    if allowed_bytes < predicted_peak:
        raise RuntimeError(
            "declared evaluator peak does not fit the GPU allocation safety limit "
            f"({predicted_peak} > {allowed_bytes} bytes)"
        )
    torch.cuda.set_per_process_memory_fraction(allowed_bytes / total_memory, 0)


def _allocate(task: TaskSpec):
    shape = (task.shape.rows, task.shape.cols)
    x = torch.empty(shape, device="cuda", dtype=torch.float16)
    output = torch.empty_like(x)
    exp_values = torch.empty(shape, device="cuda", dtype=torch.float32)
    row_max = torch.empty(task.shape.rows, device="cuda", dtype=torch.float32)
    row_sum = torch.empty(task.shape.rows, device="cuda", dtype=torch.float32)
    return x, output, exp_values, row_max, row_sum


def _fill_input(x: torch.Tensor, seed: int, task: TaskSpec) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x.normal_(
        mean=task.input.mean,
        std=task.input.standard_deviation,
        generator=generator,
    )
    high, low, tail = task.input.stress_values
    x[0, 0] = high
    stress_row = min(x.shape[0] - 1, x.shape[1] - 1)
    x[stress_row, 0] = high
    x[stress_row, 1] = low
    x[stress_row, 2:32] = tail


def _reference(x: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    rows, cols = task.shape.rows, task.shape.cols
    column = torch.arange(cols, device=x.device)
    query = torch.arange(rows, device=x.device).remainder(cols)
    valid = column.unsqueeze(0) <= query.unsqueeze(1)
    scores = x.float() * task.scale
    scores = scores.masked_fill(~valid, -float("inf"))
    return torch.softmax(scores, dim=1).to(torch.float16)


def _launch(launch, tensors: tuple[Any, ...], task: TaskSpec) -> None:
    x, output, exp_values, row_max, row_sum = tensors
    launch(
        x,
        output,
        exp_values,
        row_max,
        row_sum,
        task.scale,
        task.shape.rows,
        task.shape.cols,
    )


def _poison_writable_tensors(tensors: tuple[Any, ...]) -> None:
    _, output, exp_values, row_max, row_sum = tensors
    output.fill_(float("nan"))
    exp_values.fill_(float("nan"))
    row_max.fill_(float("nan"))
    row_sum.fill_(float("nan"))


def _validate_launch_result(
    tensors: tuple[Any, ...],
    expected: torch.Tensor,
    input_snapshot: torch.Tensor,
    task: TaskSpec,
    seed: int,
) -> dict[str, Any]:
    x, output, *_ = tensors
    if not bool(torch.equal(x, input_snapshot)):
        return {
            "passed": False,
            "maxAbsError": None,
            "maxRelError": None,
            "failedSeed": seed,
            "reason": "candidate modified the read-only input tensor",
        }
    if not bool(torch.isfinite(output).all().item()):
        return {
            "passed": False,
            "maxAbsError": None,
            "maxRelError": None,
            "failedSeed": seed,
            "reason": "candidate output contains non-finite or unwritten values",
        }

    difference = (output.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-6)
    max_abs_error = float(difference.max().item())
    max_rel_error = float(relative.max().item())
    passed = bool(
        torch.allclose(
            output,
            expected,
            atol=task.correctness.atol,
            rtol=task.correctness.rtol,
        )
    )
    return {
        "passed": passed,
        "maxAbsError": max_abs_error,
        "maxRelError": max_rel_error,
        "failedSeed": None if passed else seed,
        "reason": "" if passed else "candidate output differs from the frozen reference",
    }


class LaunchValidationError(RuntimeError):
    def __init__(
        self,
        subject: str,
        correctness: dict[str, Any],
    ) -> None:
        super().__init__(correctness["reason"])
        self.subject = subject
        self.correctness = correctness


def _check_correctness(launch, tensors: tuple[Any, ...], task: TaskSpec) -> dict[str, Any]:
    x, *_ = tensors
    max_abs_error = 0.0
    max_rel_error = 0.0
    for seed in task.correctness.seeds:
        _fill_input(x, seed, task)
        input_snapshot = x.clone()
        expected = _reference(x, task)
        for _ in range(3):
            _poison_writable_tensors(tensors)
            _launch(launch, tensors, task)
            torch.cuda.synchronize()
            result = _validate_launch_result(
                tensors, expected, input_snapshot, task, seed
            )
            if not result["passed"]:
                return result
            max_abs_error = max(max_abs_error, float(result["maxAbsError"]))
            max_rel_error = max(max_rel_error, float(result["maxRelError"]))

    return {
        "passed": True,
        "maxAbsError": max_abs_error,
        "maxRelError": max_rel_error,
        "failedSeed": None,
        "reason": "",
    }


def _measure_once(
    launch,
    tensors: tuple[Any, ...],
    task: TaskSpec,
    expected: torch.Tensor,
    input_snapshot: torch.Tensor,
    subject: str,
    seed: int,
) -> float:
    elapsed_ms: list[float] = []
    for _ in range(task.benchmark.inner_repeats):
        _poison_writable_tensors(tensors)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _launch(launch, tensors, task)
        end.record()
        end.synchronize()
        result = _validate_launch_result(
            tensors, expected, input_snapshot, task, seed
        )
        if not result["passed"]:
            raise LaunchValidationError(subject, result)
        elapsed_ms.append(start.elapsed_time(end))
    return statistics.mean(elapsed_ms)


def _summarize_samples(samples_ms: list[float]) -> dict[str, Any]:
    median_ms = statistics.median(samples_ms)
    deviations = [abs(value - median_ms) for value in samples_ms]
    return {
        "latencyMs": median_ms,
        "medianAbsoluteDeviationMs": statistics.median(deviations),
        "minMs": min(samples_ms),
        "maxMs": max(samples_ms),
        "samplesMs": samples_ms,
    }


def _benchmark(
    launch,
    tensors: tuple[Any, ...],
    task: TaskSpec,
    incumbent_launch=None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    x, *_ = tensors
    seed = task.correctness.seeds[0]
    _fill_input(x, seed, task)
    input_snapshot = x.clone()
    expected = _reference(x, task)
    for _ in range(task.benchmark.warmup):
        _poison_writable_tensors(tensors)
        _launch(launch, tensors, task)
        torch.cuda.synchronize()
        result = _validate_launch_result(
            tensors, expected, input_snapshot, task, seed
        )
        if not result["passed"]:
            raise LaunchValidationError("candidate", result)
        if incumbent_launch is not None:
            _poison_writable_tensors(tensors)
            _launch(incumbent_launch, tensors, task)
            torch.cuda.synchronize()
            result = _validate_launch_result(
                tensors, expected, input_snapshot, task, seed
            )
            if not result["passed"]:
                raise LaunchValidationError("incumbent", result)

    candidate_samples: list[float] = []
    incumbent_samples: list[float] = []
    for sample_index in range(task.benchmark.samples):
        if incumbent_launch is None:
            candidate_samples.append(
                _measure_once(
                    launch,
                    tensors,
                    task,
                    expected,
                    input_snapshot,
                    "candidate",
                    seed,
                )
            )
        elif sample_index % 2 == 0:
            incumbent_samples.append(
                _measure_once(
                    incumbent_launch,
                    tensors,
                    task,
                    expected,
                    input_snapshot,
                    "incumbent",
                    seed,
                )
            )
            candidate_samples.append(
                _measure_once(
                    launch,
                    tensors,
                    task,
                    expected,
                    input_snapshot,
                    "candidate",
                    seed,
                )
            )
        else:
            candidate_samples.append(
                _measure_once(
                    launch,
                    tensors,
                    task,
                    expected,
                    input_snapshot,
                    "candidate",
                    seed,
                )
            )
            incumbent_samples.append(
                _measure_once(
                    incumbent_launch,
                    tensors,
                    task,
                    expected,
                    input_snapshot,
                    "incumbent",
                    seed,
                )
            )

    candidate_benchmark = _summarize_samples(candidate_samples)
    if incumbent_launch is None:
        return candidate_benchmark, None

    incumbent_benchmark = _summarize_samples(incumbent_samples)
    paired_improvements = [
        (incumbent_ms - candidate_ms) / incumbent_ms
        for incumbent_ms, candidate_ms in zip(incumbent_samples, candidate_samples)
    ]
    comparison = {
        "incumbentLatencyMs": incumbent_benchmark["latencyMs"],
        "candidateLatencyMs": candidate_benchmark["latencyMs"],
        "improvementFraction": statistics.median(paired_improvements),
        "pairedImprovements": paired_improvements,
        "incumbentBenchmark": incumbent_benchmark,
    }
    return candidate_benchmark, comparison


def evaluate(task: TaskSpec, candidate: Path, incumbent: Path | None) -> dict[str, Any]:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    stage = "load"
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            if not torch.cuda.is_available():
                raise RuntimeError("ROCm device is not available")
            launch = _load_candidate(candidate, task.entrypoint, "candidate")
            tensors = _allocate(task)

            stage = "compile"
            _fill_input(tensors[0], task.correctness.seeds[0], task)
            _launch(launch, tensors, task)
            torch.cuda.synchronize()

            incumbent_launch = None
            if incumbent is not None:
                stage = "incumbent"
                incumbent_launch = _load_candidate(
                    incumbent, task.entrypoint, "incumbent"
                )
                _launch(incumbent_launch, tensors, task)
                torch.cuda.synchronize()

            stage = "correctness"
            correctness = _check_correctness(launch, tensors, task)
            if not correctness["passed"]:
                return {
                    "schemaVersion": 1,
                    "status": "wrong_answer",
                    "latencyMs": None,
                    "benchmark": None,
                    "correctness": correctness,
                    "details": correctness["reason"],
                    "capturedOutput": (captured_stdout.getvalue() + captured_stderr.getvalue())[
                        -4000:
                    ],
                }

            stage = "benchmark"
            benchmark, comparison = _benchmark(
                launch, tensors, task, incumbent_launch=incumbent_launch
            )
            return {
                "schemaVersion": 1,
                "status": "ok",
                "latencyMs": benchmark["latencyMs"],
                "benchmark": benchmark,
                "comparison": comparison,
                "correctness": correctness,
                "details": "",
                "capturedOutput": (captured_stdout.getvalue() + captured_stderr.getvalue())[
                    -4000:
                ],
                "environment": _runtime_environment(),
            }
    except LaunchValidationError as error:
        status = "evaluator_error" if error.subject == "incumbent" else "wrong_answer"
        return {
            "schemaVersion": 1,
            "status": status,
            "latencyMs": None,
            "benchmark": None,
            "correctness": error.correctness,
            "details": str(error),
            "capturedOutput": (captured_stdout.getvalue() + captured_stderr.getvalue())[
                -4000:
            ],
        }
    except Exception as error:
        if stage in {"load", "compile"}:
            status = "compile_error"
        elif stage == "incumbent":
            status = "evaluator_error"
        else:
            status = "runtime_error"
        details = "".join(traceback.format_exception_only(type(error), error)).strip()
        return {
            "schemaVersion": 1,
            "status": status,
            "latencyMs": None,
            "benchmark": None,
            "correctness": None,
            "details": details,
            "capturedOutput": (captured_stdout.getvalue() + captured_stderr.getvalue())[
                -4000:
            ],
        }


# ---------------------------------------------------------------------------
# schemaVersion=2: deterministic declarative evaluator
# ---------------------------------------------------------------------------


@dataclass
class _V2Case:
    values: dict[str, Any]
    input_snapshots: dict[str, torch.Tensor]
    device: torch.device
    reset_generation: int = 0


class OutputContractError(RuntimeError):
    pass


def _load_reference(path: Path, entrypoint: str, allocation_budget_bytes: int):
    spec = importlib.util.spec_from_file_location("reference_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create reference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_fn = getattr(module, entrypoint, None)
    if not callable(reference_fn):
        raise TypeError(f"reference entrypoint is not callable: {entrypoint}")
    guard = _GuardedReferenceTorch(allocation_budget_bytes)
    if "torch" in module.__dict__:
        module.__dict__["torch"] = guard

    raw_reference = reference_fn

    def reference_fn(*args: Any):
        guard.reset()
        return raw_reference(*args)

    return reference_fn


def _fill_input_tensor(
    tensor: torch.Tensor,
    generator: torch.Generator,
    task: TaskSpecV2,
    tensor_spec,
) -> None:
    initializer = tensor_spec.initializer
    if initializer is not None and initializer.kind == "normal":
        assert initializer.mean is not None
        assert initializer.standard_deviation is not None
        tensor.normal_(
            mean=initializer.mean,
            std=initializer.standard_deviation,
            generator=generator,
        )
    elif initializer is not None and initializer.kind == "uniform":
        assert initializer.low is not None and initializer.high is not None
        tensor.uniform_(
            float(initializer.low),
            float(initializer.high),
            generator=generator,
        )
    elif initializer is not None and initializer.kind == "integer":
        assert isinstance(initializer.low, int) and isinstance(initializer.high, int)
        tensor.random_(initializer.low, initializer.high, generator=generator)
        flat = tensor.reshape(-1)
        if flat.numel() >= 1:
            flat[0] = initializer.low
        if flat.numel() >= 2:
            flat[1] = initializer.high - 1
    elif initializer is not None and initializer.kind == "bernoulli":
        assert initializer.probability is not None
        flat = tensor.reshape(-1)
        chunk_elements = min(flat.numel(), 1 << 20)
        sample = torch.empty(
            chunk_elements, device=tensor.device, dtype=torch.float32
        )
        for offset in range(0, flat.numel(), chunk_elements):
            count = min(chunk_elements, flat.numel() - offset)
            chunk = sample[:count]
            chunk.uniform_(0.0, 1.0, generator=generator)
            flat[offset : offset + count].copy_(
                chunk < initializer.probability
            )
        if flat.numel() >= 1:
            flat[0] = False
        if flat.numel() >= 2:
            flat[1] = True
    elif initializer is not None and initializer.kind == "constant":
        tensor.fill_(initializer.value)
    elif tensor.dtype == torch.bool:
        tensor.random_(0, 2, generator=generator)
    elif tensor.is_floating_point():
        tensor.normal_(
            mean=task.input.mean,
            std=task.input.standard_deviation,
            generator=generator,
        )
    else:
        flat = tensor.reshape(-1)
        chunk_elements = min(flat.numel(), 1 << 20)
        sample = torch.empty(
            chunk_elements, device=tensor.device, dtype=torch.float32
        )
        for offset in range(0, flat.numel(), chunk_elements):
            count = min(chunk_elements, flat.numel() - offset)
            chunk = sample[:count]
            chunk.normal_(
                mean=task.input.mean,
                std=task.input.standard_deviation,
                generator=generator,
            )
            flat[offset : offset + count].copy_(chunk.to(dtype=tensor.dtype))

    if initializer is None and any(value != 0.0 for value in task.input.stress_values):
        high, low, tail = task.input.stress_values
        flat = tensor.reshape(-1)
        if flat.numel() >= 1:
            flat[0] = high
        if flat.numel() >= 2:
            flat[1] = low
        if flat.numel() > 2:
            flat[2 : min(32, flat.numel())] = tail
    if initializer is None and tensor.dtype == torch.bool and tensor.numel() >= 2:
        flat = tensor.reshape(-1)
        flat[0] = False
        flat[1] = True


def _resolve_scalar_expression(
    tensor_spec,
    values: dict[str, Any],
    task: TaskSpecV2,
) -> int:
    expression = tensor_spec.expression
    assert expression is not None
    if expression.op == "dimension":
        assert expression.name is not None
        value = int(task.dimensions[expression.name])
    else:
        assert expression.tensor is not None
        tensor = values[expression.tensor]
        if expression.op == "numel":
            value = int(tensor.numel())
        else:
            assert expression.op == "stride" and expression.axis is not None
            value = int(tensor.stride(expression.axis))
    if tensor_spec.dtype == "int32" and not -(2**31) <= value <= 2**31 - 1:
        raise OverflowError(
            f"scalar expression {tensor_spec.name} is out of range for int32"
        )
    if tensor_spec.dtype == "int64" and not -(2**63) <= value <= 2**63 - 1:
        raise OverflowError(
            f"scalar expression {tensor_spec.name} is out of range for int64"
        )
    return value


def _materialize_case(task: TaskSpecV2, seed: int) -> _V2Case:
    """Allocate one subject-owned case; no mutable value crosses subjects."""

    values: dict[str, Any] = {}
    for tensor_spec in task.tensors:
        if tensor_spec.kind == "return":
            continue
        if tensor_spec.kind == "scalar":
            if tensor_spec.expression is None:
                values[tensor_spec.name] = tensor_spec.value
            continue
        assert tensor_spec.shape is not None and tensor_spec.dtype is not None
        values[tensor_spec.name] = torch.empty(
            tensor_spec.shape,
            device="cuda",
            dtype=getattr(torch, tensor_spec.dtype),
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    for tensor_spec in task.tensors:
        if tensor_spec.kind == "input":
            _fill_input_tensor(
                values[tensor_spec.name],
                generator,
                task,
                tensor_spec,
            )
    for tensor_spec in task.tensors:
        if tensor_spec.kind == "scalar" and tensor_spec.expression is not None:
            values[tensor_spec.name] = _resolve_scalar_expression(
                tensor_spec, values, task
            )

    snapshots = {
        tensor_spec.name: values[tensor_spec.name].clone()
        for tensor_spec in task.tensors
        if tensor_spec.kind == "input"
    }
    device = torch.device("cuda", torch.cuda.current_device())
    return _V2Case(values=values, input_snapshots=snapshots, device=device)


def _bind_launch_arguments(case: _V2Case, task: TaskSpecV2) -> tuple[Any, ...]:
    """Bind by launchArguments names, independent of tensors declaration order."""

    return tuple(case.values[name] for name in task.launch_arguments)


def _integral_poison_value(
    generation: int,
    *,
    minimum: int,
    maximum: int,
    is_bool: bool,
) -> bool | int:
    if is_bool:
        return generation % 2 == 0
    return (maximum, minimum, 0)[generation % 3]


def _reset_writable_case(case: _V2Case, task: TaskSpecV2) -> None:
    generation = case.reset_generation
    for tensor_spec in task.tensors:
        if tensor_spec.kind not in {"output", "workspace"}:
            continue
        tensor = case.values[tensor_spec.name]
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        elif tensor.dtype == torch.bool:
            tensor.fill_(
                _integral_poison_value(
                    generation,
                    minimum=0,
                    maximum=1,
                    is_bool=True,
                )
            )
        else:
            dtype_info = torch.iinfo(tensor.dtype)
            tensor.fill_(
                _integral_poison_value(
                    generation,
                    minimum=dtype_info.min,
                    maximum=dtype_info.max,
                    is_bool=False,
                )
            )
    case.reset_generation += 1


def _observable_specs(task: TaskSpecV2) -> tuple[Any, ...]:
    return tuple(
        tensor_spec
        for tensor_spec in task.tensors
        if tensor_spec.kind in {"output", "return"}
    )


def _normalize_tensor_result(
    result: Any,
    count: int,
    subject: str,
    result_kind: str,
) -> tuple[torch.Tensor, ...]:
    if count == 0:
        if result is not None:
            raise OutputContractError(
                f"{subject} returned an undeclared {result_kind} value"
            )
        return ()
    if count == 1:
        if not isinstance(result, torch.Tensor):
            raise OutputContractError(
                f"{subject} must return exactly one tensor for {result_kind}"
            )
        return (result,)
    if not isinstance(result, (tuple, list)) or len(result) != count:
        actual_count = len(result) if isinstance(result, (tuple, list)) else 1
        raise OutputContractError(
            f"{subject} must return exactly {count} tensors for {result_kind}; "
            f"got {actual_count}"
        )
    if any(not isinstance(value, torch.Tensor) for value in result):
        raise OutputContractError(f"{subject} {result_kind} values must all be tensors")
    return tuple(result)


def _collect_candidate_outputs(
    case: _V2Case,
    task: TaskSpecV2,
    launch_result: Any,
    subject: str,
) -> tuple[torch.Tensor, ...]:
    return_specs = tuple(spec for spec in task.tensors if spec.kind == "return")
    returned = iter(
        _normalize_tensor_result(
            launch_result, len(return_specs), subject, "declared return"
        )
    )
    outputs: list[torch.Tensor] = []
    for tensor_spec in _observable_specs(task):
        if tensor_spec.kind == "output":
            outputs.append(case.values[tensor_spec.name])
        else:
            outputs.append(next(returned))
    return tuple(outputs)


def _failure_result(seed: int, reason: str) -> dict[str, Any]:
    return {
        "passed": False,
        "maxAbsError": None,
        "maxRelError": None,
        "failedSeed": seed,
        "reason": reason,
    }


def _validate_input_immutability(
    case: _V2Case,
    task: TaskSpecV2,
    seed: int,
    subject: str,
) -> dict[str, Any] | None:
    for tensor_spec in task.tensors:
        if tensor_spec.kind != "input":
            continue
        if not bool(
            torch.equal(
                case.values[tensor_spec.name], case.input_snapshots[tensor_spec.name]
            )
        ):
            return _failure_result(
                seed,
                f"{subject} modified the read-only input tensor: {tensor_spec.name}",
            )
    return None


def _tensor_contract_error(
    tensor: Any,
    tensor_spec,
    device: torch.device,
    subject: str,
) -> str | None:
    if not isinstance(tensor, torch.Tensor):
        return f"{subject} output {tensor_spec.name} is not a tensor"
    assert tensor_spec.shape is not None and tensor_spec.dtype is not None
    if tuple(tensor.shape) != tensor_spec.shape:
        return (
            f"{subject} output {tensor_spec.name} has shape {tuple(tensor.shape)}, "
            f"expected {tensor_spec.shape}"
        )
    expected_dtype = getattr(torch, tensor_spec.dtype)
    if tensor.dtype != expected_dtype:
        return (
            f"{subject} output {tensor_spec.name} has dtype {tensor.dtype}, "
            f"expected {expected_dtype}"
        )
    if tensor.device != device:
        return (
            f"{subject} output {tensor_spec.name} is on {tensor.device}, "
            f"expected {device}"
        )
    if not tensor.is_contiguous():
        return (
            f"{subject} output {tensor_spec.name} must use the contiguous layout; "
            f"got stride {tuple(tensor.stride())}"
        )
    return None


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.device == right.device
        and left.untyped_storage().data_ptr()
        == right.untyped_storage().data_ptr()
    )


def _output_alias_error(
    case: _V2Case,
    outputs: tuple[torch.Tensor, ...],
    task: TaskSpecV2,
    subject: str,
) -> str | None:
    observable_specs = _observable_specs(task)
    protected = [
        (spec.name, case.values[spec.name])
        for spec in task.tensors
        if spec.kind in {"input", "workspace"}
    ]
    for index, (spec, output) in enumerate(zip(observable_specs, outputs)):
        if not isinstance(output, torch.Tensor):
            continue
        for protected_name, protected_tensor in protected:
            if _shares_storage(output, protected_tensor):
                return (
                    f"{subject} output {spec.name} aliases protected tensor "
                    f"{protected_name}"
                )
        for previous_spec, previous_output in zip(
            observable_specs[:index], outputs[:index]
        ):
            if _shares_storage(output, previous_output):
                return (
                    f"{subject} outputs {previous_spec.name} and {spec.name} "
                    "alias the same storage"
                )
    return None


def _validate_outputs(
    case: _V2Case,
    outputs: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    task: TaskSpecV2,
    seed: int,
    subject: str,
) -> dict[str, Any]:
    input_error = _validate_input_immutability(case, task, seed, subject)
    if input_error is not None:
        return input_error
    output_specs = _observable_specs(task)
    if len(outputs) != len(output_specs):
        return _failure_result(
            seed,
            f"{subject} produced {len(outputs)} outputs; expected {len(output_specs)}",
        )
    alias_error = _output_alias_error(case, outputs, task, subject)
    if alias_error is not None:
        return _failure_result(seed, alias_error)

    max_abs_error = 0.0
    max_rel_error = 0.0
    for tensor_spec, output, oracle in zip(output_specs, outputs, expected):
        contract_error = _tensor_contract_error(
            output, tensor_spec, case.device, subject
        )
        if contract_error is not None:
            return _failure_result(seed, contract_error)
        if output.is_floating_point() and not bool(torch.isfinite(output).all().item()):
            return _failure_result(
                seed, f"{subject} output contains non-finite values: {tensor_spec.name}"
            )
        if not output.is_floating_point():
            if not bool(torch.equal(output, oracle)):
                return {
                    "passed": False,
                    "maxAbsError": None,
                    "maxRelError": None,
                    "failedSeed": seed,
                    "reason": (
                        f"{subject} integer/bool output differs exactly from reference: "
                        f"{tensor_spec.name}"
                    ),
                }
            continue
        metric_dtype = torch.float64 if output.dtype == torch.float64 else torch.float32
        metric_output = output.to(dtype=metric_dtype)
        metric_oracle = oracle.to(dtype=metric_dtype)
        difference = (metric_output - metric_oracle).abs()
        relative = difference / metric_oracle.abs().clamp_min(1e-12)
        max_abs_error = max(max_abs_error, float(difference.max().item()))
        max_rel_error = max(max_rel_error, float(relative.max().item()))
        if not bool(
            torch.allclose(
                output,
                oracle,
                atol=task.correctness.atol,
                rtol=task.correctness.rtol,
            )
        ):
            return {
                "passed": False,
                "maxAbsError": max_abs_error,
                "maxRelError": max_rel_error,
                "failedSeed": seed,
                "reason": (
                    f"{subject} output differs from reference: {tensor_spec.name}"
                ),
            }
    return {
        "passed": True,
        "maxAbsError": max_abs_error,
        "maxRelError": max_rel_error,
        "failedSeed": None,
        "reason": "",
    }


def _run_reference_once(
    reference_fn,
    task: TaskSpecV2,
    seed: int,
    *,
    poison_generation: int,
) -> tuple[torch.Tensor, ...]:
    reference_case = _materialize_case(task, seed)
    reference_case.reset_generation = poison_generation
    _reset_writable_case(reference_case, task)
    result = reference_fn(*_bind_launch_arguments(reference_case, task))
    torch.cuda.synchronize()
    input_error = _validate_input_immutability(
        reference_case, task, seed, "reference"
    )
    if input_error is not None:
        raise LaunchValidationError("reference", input_error)
    output_specs = _observable_specs(task)
    try:
        outputs = _normalize_tensor_result(
            result, len(output_specs), "reference", "declared output"
        )
    except OutputContractError as error:
        raise LaunchValidationError(
            "reference", _failure_result(seed, str(error))
        ) from error
    for tensor_spec, output in zip(output_specs, outputs):
        contract_error = _tensor_contract_error(
            output, tensor_spec, reference_case.device, "reference"
        )
        if contract_error is not None:
            raise LaunchValidationError(
                "reference", _failure_result(seed, contract_error)
            )
        if output.is_floating_point() and not bool(torch.isfinite(output).all().item()):
            raise LaunchValidationError(
                "reference",
                _failure_result(
                    seed,
                    f"reference output contains non-finite values: {tensor_spec.name}",
                ),
            )
    alias_error = _output_alias_error(reference_case, outputs, task, "reference")
    if alias_error is not None:
        raise LaunchValidationError(
            "reference", _failure_result(seed, alias_error)
        )
    # Clone before releasing the reference case so no candidate-owned mutation can
    # change the oracle through an alias.
    return tuple(output.detach().clone() for output in outputs)


def _build_oracle(
    reference_fn,
    task: TaskSpecV2,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    first = _run_reference_once(
        reference_fn,
        task,
        seed,
        poison_generation=0,
    )
    integral_indexes = [
        index
        for index, spec in enumerate(_observable_specs(task))
        if spec.dtype not in {"bfloat16", "float16", "float32", "float64"}
    ]
    if not integral_indexes:
        return first

    second = _run_reference_once(
        reference_fn,
        task,
        seed,
        poison_generation=1,
    )
    for index in integral_indexes:
        if not bool(torch.equal(first[index], second[index])):
            raise LaunchValidationError(
                "reference",
                _failure_result(
                    seed,
                    "reference integer/bool output depends on evaluator poison; "
                    "the output is not fully written",
                ),
            )
    return first


def _execute_checked(
    launch_fn,
    case: _V2Case,
    expected: tuple[torch.Tensor, ...],
    task: TaskSpecV2,
    seed: int,
    subject: str,
    *,
    timed: bool,
) -> tuple[float | None, dict[str, Any]]:
    start = torch.cuda.Event(enable_timing=True) if timed else None
    end = torch.cuda.Event(enable_timing=True) if timed else None

    # Reset happens before the event and validation after it, so only the declared
    # entrypoint is measured.
    _reset_writable_case(case, task)
    prepare_returns = getattr(launch_fn, "prepare_return_buffers", None)
    if callable(prepare_returns):
        prepare_returns()
    if start is not None:
        start.record()
    launch_result = launch_fn(*_bind_launch_arguments(case, task))
    if end is not None:
        end.record()
        end.synchronize()
    else:
        torch.cuda.synchronize()
    try:
        outputs = _collect_candidate_outputs(case, task, launch_result, subject)
    except OutputContractError as error:
        raise LaunchValidationError(
            subject, _failure_result(seed, str(error))
        ) from error
    correctness = _validate_outputs(
        case, outputs, expected, task, seed, subject
    )
    if not correctness["passed"]:
        raise LaunchValidationError(subject, correctness)
    if start is None or end is None:
        return None, correctness
    return float(start.elapsed_time(end)), correctness


def _check_correctness_v2(
    launch_fn,
    task: TaskSpecV2,
    reference_fn,
) -> dict[str, Any]:
    max_abs_error = 0.0
    max_rel_error = 0.0
    for seed in task.correctness.seeds:
        expected = _build_oracle(reference_fn, task, seed)
        candidate_case = _materialize_case(task, seed)
        for _ in range(3):
            try:
                _, result = _execute_checked(
                    launch_fn,
                    candidate_case,
                    expected,
                    task,
                    seed,
                    "candidate",
                    timed=False,
                )
            except LaunchValidationError as error:
                return error.correctness
            max_abs_error = max(
                max_abs_error, float(result["maxAbsError"] or 0.0)
            )
            max_rel_error = max(
                max_rel_error, float(result["maxRelError"] or 0.0)
            )
    return {
        "passed": True,
        "maxAbsError": max_abs_error,
        "maxRelError": max_rel_error,
        "failedSeed": None,
        "reason": "",
    }


def _measure_once_v2(
    launch_fn,
    case: _V2Case,
    expected: tuple[torch.Tensor, ...],
    task: TaskSpecV2,
    seed: int,
    subject: str,
) -> float:
    elapsed_ms: list[float] = []
    for _ in range(task.benchmark.inner_repeats):
        elapsed, _ = _execute_checked(
            launch_fn,
            case,
            expected,
            task,
            seed,
            subject,
            timed=True,
        )
        assert elapsed is not None
        elapsed_ms.append(elapsed)
    return statistics.mean(elapsed_ms)


def _benchmark_v2(
    launch_fn,
    task: TaskSpecV2,
    reference_fn,
    incumbent_launch_fn=None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    seed = task.correctness.seeds[0]
    expected = _build_oracle(reference_fn, task, seed)
    candidate_case = _materialize_case(task, seed)
    incumbent_case = (
        _materialize_case(task, seed) if incumbent_launch_fn is not None else None
    )

    for _ in range(task.benchmark.warmup):
        _execute_checked(
            launch_fn,
            candidate_case,
            expected,
            task,
            seed,
            "candidate",
            timed=False,
        )
        if incumbent_launch_fn is not None and incumbent_case is not None:
            _execute_checked(
                incumbent_launch_fn,
                incumbent_case,
                expected,
                task,
                seed,
                "incumbent",
                timed=False,
            )

    candidate_samples: list[float] = []
    incumbent_samples: list[float] = []
    for sample_index in range(task.benchmark.samples):
        if incumbent_launch_fn is None or incumbent_case is None:
            candidate_samples.append(
                _measure_once_v2(
                    launch_fn,
                    candidate_case,
                    expected,
                    task,
                    seed,
                    "candidate",
                )
            )
        elif sample_index % 2 == 0:
            incumbent_samples.append(
                _measure_once_v2(
                    incumbent_launch_fn,
                    incumbent_case,
                    expected,
                    task,
                    seed,
                    "incumbent",
                )
            )
            candidate_samples.append(
                _measure_once_v2(
                    launch_fn,
                    candidate_case,
                    expected,
                    task,
                    seed,
                    "candidate",
                )
            )
        else:
            candidate_samples.append(
                _measure_once_v2(
                    launch_fn,
                    candidate_case,
                    expected,
                    task,
                    seed,
                    "candidate",
                )
            )
            incumbent_samples.append(
                _measure_once_v2(
                    incumbent_launch_fn,
                    incumbent_case,
                    expected,
                    task,
                    seed,
                    "incumbent",
                )
            )

    candidate_benchmark = _summarize_samples(candidate_samples)
    if incumbent_launch_fn is None:
        return candidate_benchmark, None
    incumbent_benchmark = _summarize_samples(incumbent_samples)
    paired_improvements = [
        (incumbent_ms - candidate_ms) / incumbent_ms
        for incumbent_ms, candidate_ms in zip(
            incumbent_samples, candidate_samples
        )
    ]
    comparison = {
        "incumbentLatencyMs": incumbent_benchmark["latencyMs"],
        "candidateLatencyMs": candidate_benchmark["latencyMs"],
        "improvementFraction": statistics.median(paired_improvements),
        "pairedImprovements": paired_improvements,
        "incumbentBenchmark": incumbent_benchmark,
    }
    return candidate_benchmark, comparison


def evaluate_v2(
    task: TaskSpecV2,
    candidate: Path,
    reference: Path,
    incumbent: Path | None = None,
    *,
    correctness_only: bool = False,
) -> dict[str, Any]:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    stage = "load"
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            if not torch.cuda.is_available():
                raise RuntimeError("ROCm device is not available")
            _apply_cuda_allocation_limit(task)
            return_budget_bytes = sum(
                math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
                for tensor in task.tensors
                if tensor.kind == "return"
                and tensor.shape is not None
                and tensor.dtype is not None
            )
            return_specs = tuple(
                tensor for tensor in task.tensors if tensor.kind == "return"
            )
            reference_budget_bytes = sum(
                math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
                for tensor in task.tensors
                if tensor.kind != "scalar"
                and tensor.shape is not None
                and tensor.dtype is not None
            )
            launch_fn = _load_candidate(
                candidate,
                task.entrypoint,
                "candidate",
                allocation_budget_bytes=return_budget_bytes,
                return_specs=return_specs,
            )
            reference_fn = _load_reference(
                reference,
                task.reference.entrypoint,
                allocation_budget_bytes=reference_budget_bytes,
            )
            incumbent_launch_fn = None
            if incumbent is not None:
                incumbent_launch_fn = _load_candidate(
                    incumbent,
                    task.entrypoint,
                    "incumbent",
                    allocation_budget_bytes=return_budget_bytes,
                    return_specs=return_specs,
                )

            seed = task.correctness.seeds[0]
            stage = "reference"
            compile_expected = _build_oracle(reference_fn, task, seed)

            stage = "compile"
            _execute_checked(
                launch_fn,
                _materialize_case(task, seed),
                compile_expected,
                task,
                seed,
                "candidate",
                timed=False,
            )
            if incumbent_launch_fn is not None:
                stage = "incumbent"
                _execute_checked(
                    incumbent_launch_fn,
                    _materialize_case(task, seed),
                    compile_expected,
                    task,
                    seed,
                    "incumbent",
                    timed=False,
                )

            stage = "correctness"
            correctness = _check_correctness_v2(launch_fn, task, reference_fn)
            if not correctness["passed"]:
                return {
                    "schemaVersion": 2,
                    "status": "wrong_answer",
                    "latencyMs": None,
                    "benchmark": None,
                    "correctness": correctness,
                    "details": correctness["reason"],
                    "capturedOutput": (
                        captured_stdout.getvalue() + captured_stderr.getvalue()
                    )[-4000:],
                }

            if correctness_only:
                return {
                    "schemaVersion": 2,
                    "status": "ok",
                    "latencyMs": None,
                    "benchmark": None,
                    "comparison": None,
                    "correctness": correctness,
                    "details": "",
                    "capturedOutput": (
                        captured_stdout.getvalue() + captured_stderr.getvalue()
                    )[-4000:],
                    "environment": _runtime_environment(),
                }

            stage = "benchmark"
            benchmark, comparison = _benchmark_v2(
                launch_fn,
                task,
                reference_fn,
                incumbent_launch_fn=incumbent_launch_fn,
            )
            return {
                "schemaVersion": 2,
                "status": "ok",
                "latencyMs": benchmark["latencyMs"],
                "benchmark": benchmark,
                "comparison": comparison,
                "correctness": correctness,
                "details": "",
                "capturedOutput": (
                    captured_stdout.getvalue() + captured_stderr.getvalue()
                )[-4000:],
                "environment": _runtime_environment(),
            }
    except LaunchValidationError as error:
        status = (
            "evaluator_error"
            if error.subject in {"incumbent", "reference"}
            else "wrong_answer"
        )
        return {
            "schemaVersion": 2,
            "status": status,
            "latencyMs": None,
            "benchmark": None,
            "correctness": error.correctness,
            "details": str(error),
            "capturedOutput": (
                captured_stdout.getvalue() + captured_stderr.getvalue()
            )[-4000:],
        }
    except Exception as error:
        if stage in {"load", "compile"}:
            status = "compile_error"
        elif stage in {"reference", "incumbent"}:
            status = "evaluator_error"
        else:
            status = "runtime_error"
        details = "".join(traceback.format_exception_only(type(error), error)).strip()
        return {
            "schemaVersion": 2,
            "status": status,
            "latencyMs": None,
            "benchmark": None,
            "correctness": None,
            "details": details,
            "capturedOutput": (
                captured_stdout.getvalue() + captured_stderr.getvalue()
            )[-4000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--reference", type=Path, help="reference file (v2 only)")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    task = load_task(args.task)
    if isinstance(task, TaskSpecV2):
        if args.reference is None:
            raise SystemExit("schemaVersion=2 requires --reference")
        result = evaluate_v2(
            task,
            args.candidate,
            args.reference,
            args.incumbent,
            correctness_only=args.correctness_only,
        )
    else:
        if args.correctness_only:
            raise SystemExit("--correctness-only requires schemaVersion=2")
        result = evaluate(task, args.candidate, args.incumbent)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

