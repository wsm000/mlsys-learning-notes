"""Stable subprocess boundary for evaluating one candidate file.

The optimization agent uses schemaVersion=2 with a declarative task and an
independent reference.py. schemaVersion=1 remains only to reproduce the
historical masked-softmax experiment until that evidence is rerun under v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_policy import validate_candidate, validate_reference
from task_spec import TaskSpecV2, load_task


MAX_CAPTURE_CHARS = 20_000
MAX_CAPTURE_BYTES = 64 * 1024
CHILD_ENV_ALLOWLIST = {
    "CC",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "CXX",
    "HIPCC_COMPILE_FLAGS_APPEND",
    "HIPCC_LINK_FLAGS_APPEND",
    "HIP_PATH",
    "HIP_VISIBLE_DEVICES",
    "HSA_ENABLE_SDMA",
    "HSA_OVERRIDE_GFX_VERSION",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PATH",
    "PYTORCH_ROCM_ARCH",
    "ROCM_HOME",
    "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES",
    "TRITON_LIBDEVICE_PATH",
    "VIRTUAL_ENV",
}


def _verify_embedded_preflight_evidence(
    task_path: Path,
    reference_path: Path,
    *,
    baseline_path: Path | None = None,
    allow_missing: bool = False,
) -> str | None:
    """Verify hashes embedded by the frozen-workspace preflight.

    Temporary preflight tasks intentionally have no evidence yet and therefore
    skip this check. Frozen tasks bind the raw declarative contract and
    reference bytes; correctness-only reruns additionally bind the baseline.
    """

    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        reference = payload.get("reference", {})
        evidence = reference.get("preflightEvidence")
        if evidence is None:
            if "dimensions" in payload and not allow_missing:
                return "active declarative evaluation requires preflight evidence"
            return None
        contract = {
            "dimensions": payload["dimensions"],
            "tensors": payload["tensors"],
            "launchArguments": payload["launchArguments"],
        }
        if "costModel" in payload:
            contract["costModel"] = payload["costModel"]
        contract_bytes = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected = {
            "contractSha256": hashlib.sha256(contract_bytes).hexdigest(),
            "referenceSha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        }
        if baseline_path is not None:
            expected["baselineSha256"] = hashlib.sha256(
                baseline_path.read_bytes()
            ).hexdigest()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return f"could not verify preflight evidence: {error}"
    for field, digest in expected.items():
        if evidence.get(field) != digest:
            return f"preflight evidence mismatch: {field}"
    return None


def _scalar_bounds(task: TaskSpecV2) -> dict[str, float]:
    tensors = {tensor.name: tensor for tensor in task.tensors}
    bounds: dict[str, float] = {}
    for tensor in task.tensors:
        if tensor.kind != "scalar":
            continue
        if tensor.expression is None:
            if isinstance(tensor.value, (int, float)) and not isinstance(
                tensor.value, bool
            ):
                bounds[tensor.name] = abs(float(tensor.value))
            continue
        expression = tensor.expression
        if expression.op == "dimension":
            assert expression.name is not None
            bounds[tensor.name] = float(task.dimensions[expression.name])
            continue
        assert expression.tensor is not None
        target = tensors[expression.tensor]
        assert target.shape is not None
        if expression.op == "numel":
            bounds[tensor.name] = float(math.prod(target.shape))
        else:
            assert expression.axis is not None
            bounds[tensor.name] = float(
                math.prod(target.shape[expression.axis + 1 :])
            )
    return bounds


def _as_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _failure(status: str, details: str, captured_output: str = "", schema_version: int = 1) -> dict[str, Any]:
    return {
        "schemaVersion": schema_version,
        "status": status,
        "latencyMs": None,
        "benchmark": None,
        "correctness": None,
        "details": details,
        "capturedOutput": captured_output[-4000:],
    }


def _isolated_child_env(root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in CHILD_ENV_ALLOWLIST
    }
    home = root / "home"
    temporary = root / "tmp"
    cache = root / "cache"
    for directory in (home, temporary, cache):
        directory.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "XDG_CACHE_HOME": str(cache),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TORCH_HOME": str(cache / "torch"),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _drain_bounded(stream, captured: bytearray) -> None:
    """Drain a pipe continuously while retaining only its bounded tail."""

    while True:
        chunk = stream.read(64 * 1024)
        if not chunk:
            return
        captured.extend(chunk)
        overflow = len(captured) - MAX_CAPTURE_BYTES
        if overflow > 0:
            del captured[:overflow]


def evaluate(
    task_path: Path,
    candidate_path: Path,
    incumbent_path: Path | None = None,
    reference_path: Path | None = None,
    harness_path: Path | None = None,
    *,
    correctness_only: bool = False,
) -> dict[str, Any]:
    """评测一个 candidate。

    v1: task_path + candidate_path + incumbent_path
    v2: 同上 + reference_path；未传时使用 task.reference.filename。

    harness_path 只为旧调用方保留，schemaVersion=2 不读取也不执行它。
    """
    task = load_task(task_path)
    is_v2 = isinstance(task, TaskSpecV2)
    schema_version = 2 if is_v2 else 1

    del harness_path
    if is_v2:
        policy_options = {
            "expected_launch_args": task.launch_arguments,
            "entrypoint": task.entrypoint,
            "allow_torch": True,
            "tensor_arguments": tuple(
                tensor.name
                for tensor in task.tensors
                if tensor.kind not in {"scalar", "return"}
            ),
            "scalar_bounds": _scalar_bounds(task),
        }
        errors = validate_candidate(
            candidate_path,
            task.optimization.max_source_chars,
            **policy_options,
        )
        if errors:
            return _failure(
                "invalid_candidate", "; ".join(errors), schema_version=schema_version
            )
        if incumbent_path is not None:
            incumbent_errors = validate_candidate(
                incumbent_path,
                task.optimization.max_source_chars,
                **policy_options,
            )
            if incumbent_errors:
                return _failure(
                    "evaluator_error",
                    "incumbent: " + "; ".join(incumbent_errors),
                    schema_version=schema_version,
                )
        if reference_path is None:
            reference_path = task_path.parent / task.reference.filename
        evidence_error = _verify_embedded_preflight_evidence(
            task_path,
            reference_path,
            baseline_path=candidate_path if correctness_only else None,
            allow_missing=correctness_only,
        )
        if evidence_error is not None:
            return _failure(
                "evaluator_error",
                evidence_error,
                schema_version=schema_version,
            )
        reference_errors = validate_reference(
            reference_path,
            task.optimization.max_source_chars,
            entrypoint=task.reference.entrypoint,
            expected_arguments=task.launch_arguments,
            scalar_bounds=_scalar_bounds(task),
        )
        if reference_errors:
            return _failure(
                "evaluator_error",
                "reference: " + "; ".join(reference_errors),
                schema_version=schema_version,
            )
    else:
        if correctness_only:
            return _failure(
                "evaluator_error",
                "correctness-only preflight requires schemaVersion=2",
                schema_version=schema_version,
            )
        errors = validate_candidate(candidate_path, task.optimization.max_source_chars)
        if errors:
            return _failure(
                "invalid_candidate", "; ".join(errors), schema_version=schema_version
            )
        if incumbent_path is not None:
            incumbent_errors = validate_candidate(
                incumbent_path, task.optimization.max_source_chars
            )
            if incumbent_errors:
                return _failure(
                    "evaluator_error",
                    "incumbent: " + "; ".join(incumbent_errors),
                    schema_version=schema_version,
                )

    worker = Path(__file__).with_name("worker.py")

    command = [
        sys.executable, "-I", str(worker),
        "--task", str(task_path.resolve()),
        "--candidate", str(candidate_path.resolve()),
    ]
    if incumbent_path is not None:
        command.extend(["--incumbent", str(incumbent_path.resolve())])
    if is_v2:
        assert reference_path is not None
        command.extend(["--reference", str(reference_path.resolve())])
        if correctness_only:
            command.append("--correctness-only")

    with tempfile.TemporaryDirectory(prefix="hello-gpu-evaluator-") as temporary_dir:
        process = subprocess.Popen(
            command,
            cwd=temporary_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_isolated_child_env(Path(temporary_dir)),
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout_tail = bytearray()
        stderr_tail = bytearray()
        drain_threads = [
            threading.Thread(
                target=_drain_bounded,
                args=(process.stdout, stdout_tail),
                daemon=True,
            ),
            threading.Thread(
                target=_drain_bounded,
                args=(process.stderr, stderr_tail),
                daemon=True,
            ),
        ]
        for thread in drain_threads:
            thread.start()
        timed_out = False
        try:
            timeout_seconds = (
                max(task.benchmark.timeout_seconds, 600)
                if correctness_only
                else task.benchmark.timeout_seconds
            )
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process)
            process.wait()
        except BaseException:
            _kill_process_group(process)
            process.wait()
            raise
        finally:
            for thread in drain_threads:
                thread.join()

        stdout = stdout_tail.decode("utf-8", errors="replace")
        stderr = stderr_tail.decode("utf-8", errors="replace")
        if timed_out:
            return _failure(
                "timeout",
                "candidate evaluation timed out",
                (stdout + stderr)[-MAX_CAPTURE_CHARS:],
                schema_version,
            )

    captured = (stdout + stderr)[-MAX_CAPTURE_CHARS:]
    if process.returncode != 0:
        return _failure(
            "runtime_error",
            f"worker exited with status {process.returncode}",
            captured,
            schema_version,
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        return _failure("evaluator_error", f"worker returned invalid JSON: {error}", captured, schema_version)
    if not isinstance(result, dict):
        return _failure("evaluator_error", "worker result is not an object", captured, schema_version)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--reference", type=Path, help="reference.py path (v2)")
    parser.add_argument("--harness", type=Path, help="deprecated and ignored")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.task, args.candidate, args.incumbent, args.reference, args.harness)
    indent = 2 if args.pretty else None
    print(json.dumps(result, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    main()

