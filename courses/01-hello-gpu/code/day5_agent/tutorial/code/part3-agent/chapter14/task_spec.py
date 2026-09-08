"""Load frozen declarative tasks plus the historical softmax fixture schema."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class Shape:
    rows: int
    cols: int


@dataclass(frozen=True)
class CorrectnessConfig:
    seeds: tuple[int, ...]
    atol: float
    rtol: float


@dataclass(frozen=True)
class SemanticsConfig:
    operation: str
    mask: str
    accumulation_dtype: str
    masked_output: float


@dataclass(frozen=True)
class InputConfig:
    distribution: str
    mean: float
    standard_deviation: float
    stress_values: tuple[float, float, float]


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup: int
    samples: int
    inner_repeats: int
    timeout_seconds: int


@dataclass(frozen=True)
class OptimizationConfig:
    max_rounds: int
    patience: int
    min_improvement_fraction: float
    max_source_chars: int


@dataclass(frozen=True)
class TaskSpec:
    """schemaVersion=1: softmax-specific contract (保留向后兼容)。"""

    schema_version: int
    name: str
    description: str
    language: str
    candidate_filename: str
    entrypoint: str
    shape: Shape
    dtype: str
    scale: float
    semantics: SemanticsConfig
    input: InputConfig
    launch_arguments: tuple[str, ...]
    correctness: CorrectnessConfig
    benchmark: BenchmarkConfig
    optimization: OptimizationConfig


# ---------------------------------------------------------------------------
# schemaVersion=2: 声明式通用 kernel 任务
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarExpression:
    """A side-effect-free scalar binding resolved by the evaluator."""

    op: str
    name: str | None = None
    tensor: str | None = None
    axis: int | None = None


@dataclass(frozen=True)
class TensorInitializer:
    """Deterministic per-input data domain for a schema-v2 tensor."""

    kind: str
    mean: float | None = None
    standard_deviation: float | None = None
    low: int | float | None = None
    high: int | float | None = None
    probability: float | None = None
    value: bool | int | float | None = None


@dataclass(frozen=True)
class TensorSpec:
    """One named ABI value in a schemaVersion=2 task.

    For scalar values, ``dtype`` validates the host value domain. Triton still
    infers the device-side scalar width unless the JIT kernel casts explicitly.
    """

    name: str
    kind: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    value: bool | int | float | None = None
    expression: ScalarExpression | None = None
    initializer: TensorInitializer | None = None


@dataclass(frozen=True)
class PreflightEvidence:
    algorithm: str
    baseline_sha256: str
    reference_sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class AnalyticCostModel:
    """Explicit, user-approved algorithmic work model for reporting only."""

    flops: int
    bytes: int
    label: str


@dataclass(frozen=True)
class ReferenceConfig:
    """reference 实现的配置（v2）。"""

    filename: str  # reference.py
    entrypoint: str  # reference.py 里的函数名，如 "reference"
    self_check_passed: bool = False
    preflight_evidence: PreflightEvidence | None = None


@dataclass(frozen=True)
class TaskSpecV2:
    """schemaVersion=2: 声明式 Triton kernel 任务。

    和 v1 的区别：
    - 去掉 softmax 专用的 semantics / scale / dtype
    - 用 tensors 列表描述 contiguous、只读-input 的受限 ABI
    - 新增 reference 配置（LLM 生成或用户提供）
    """

    schema_version: int  # 固定 = 2
    name: str
    description: str
    language: str  # "triton"
    entrypoint: str  # "launch"
    shape: Shape
    dimensions: Mapping[str, int]
    cost_model: AnalyticCostModel | None
    input: InputConfig
    tensors: tuple[TensorSpec, ...]
    launch_arguments: tuple[str, ...]  # 顺序敏感的参数名
    reference: ReferenceConfig
    correctness: CorrectnessConfig
    benchmark: BenchmarkConfig
    optimization: OptimizationConfig


def _require_keys(payload: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{where}: missing={missing}, extra={extra}")


def _require_keys_present(
    payload: dict[str, Any], expected: set[str], where: str
) -> None:
    """只检查必需的键都在，允许有额外的可选键（v2 用）。"""
    actual = set(payload)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"{where}: missing={missing}")


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_float(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _finite_float(value: Any, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def load_task(path: Path):
    """加载 task.json，按 schemaVersion 路由到 v1 或 v2 loader。

    返回 TaskSpec（v1）或 TaskSpecV2（v2）。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task must be a JSON object")
    version = payload.get("schemaVersion")
    if version == 1:
        return _load_task_v1(payload)
    if version == 2:
        return _load_task_v2(payload)
    raise ValueError(f"unsupported schemaVersion: {version!r} (只支持 1 或 2)")


def _load_task_v1(payload: dict[str, Any]) -> TaskSpec:
    """schemaVersion=1: softmax 专用 contract（保留向后兼容）。"""

    _require_keys(
        payload,
        {
            "schemaVersion",
            "name",
            "description",
            "language",
            "candidateFilename",
            "entrypoint",
            "shape",
            "dtype",
            "scale",
            "semantics",
            "input",
            "launchArguments",
            "correctness",
            "benchmark",
            "optimization",
        },
        "task",
    )

    shape_payload = payload["shape"]
    correctness_payload = payload["correctness"]
    semantics_payload = payload["semantics"]
    input_payload = payload["input"]
    benchmark_payload = payload["benchmark"]
    optimization_payload = payload["optimization"]
    for value, name in (
        (shape_payload, "shape"),
        (correctness_payload, "correctness"),
        (semantics_payload, "semantics"),
        (input_payload, "input"),
        (benchmark_payload, "benchmark"),
        (optimization_payload, "optimization"),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")

    _require_keys(shape_payload, {"rows", "cols"}, "shape")
    _require_keys(correctness_payload, {"seeds", "atol", "rtol"}, "correctness")
    _require_keys(
        semantics_payload,
        {"operation", "mask", "accumulationDtype", "maskedOutput"},
        "semantics",
    )
    _require_keys(
        input_payload,
        {"distribution", "mean", "standardDeviation", "stressValues"},
        "input",
    )
    _require_keys(
        benchmark_payload,
        {"warmup", "samples", "innerRepeats", "timeoutSeconds"},
        "benchmark",
    )
    _require_keys(
        optimization_payload,
        {
            "maxRounds",
            "patience",
            "minImprovementFraction",
            "maxSourceChars",
        },
        "optimization",
    )

    seeds = correctness_payload["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("correctness.seeds must be a non-empty integer array")

    if payload["schemaVersion"] != 1:
        raise ValueError("only task schemaVersion 1 is supported")
    if payload["language"] != "triton":
        raise ValueError("this tutorial implements only the triton adapter")
    if payload["dtype"] != "float16":
        raise ValueError("this tutorial task fixes dtype to float16")
    if payload["entrypoint"] != "launch":
        raise ValueError("candidate entrypoint must be launch")
    expected_semantics = {
        "operation": "scaled-causal-row-softmax",
        "mask": "column-le-row-mod-cols",
        "accumulationDtype": "float32",
        "maskedOutput": 0,
    }
    if semantics_payload != expected_semantics or isinstance(
        semantics_payload["maskedOutput"], bool
    ):
        raise ValueError("this evaluator supports only the frozen softmax semantics")
    if input_payload["distribution"] != "normal":
        raise ValueError("this evaluator supports only normal input distribution")
    stress_values = input_payload["stressValues"]
    if (
        not isinstance(stress_values, list)
        or len(stress_values) != 3
        or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in stress_values
        )
    ):
        raise ValueError("input.stressValues must contain exactly three numbers")
    launch_arguments = payload["launchArguments"]
    expected_launch_arguments = [
        "x",
        "output",
        "exp_values",
        "row_max",
        "row_sum",
        "scale",
        "rows",
        "cols",
    ]
    if launch_arguments != expected_launch_arguments:
        raise ValueError("launchArguments must match the frozen evaluator ABI")
    candidate_filename = payload["candidateFilename"]
    if not isinstance(candidate_filename, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate_filename
    ):
        raise ValueError("candidateFilename must be a simple filename")
    if candidate_filename in {".", ".."}:
        raise ValueError("candidateFilename must not be . or ..")

    min_improvement = _non_negative_float(
        optimization_payload["minImprovementFraction"],
        "optimization.minImprovementFraction",
    )
    if min_improvement >= 1:
        raise ValueError("optimization.minImprovementFraction must be less than 1")

    return TaskSpec(
        schema_version=1,
        name=str(payload["name"]),
        description=str(payload["description"]),
        language="triton",
        candidate_filename=candidate_filename,
        entrypoint="launch",
        shape=Shape(
            rows=_positive_int(shape_payload["rows"], "shape.rows"),
            cols=_positive_int(shape_payload["cols"], "shape.cols"),
        ),
        dtype="float16",
        scale=_finite_float(payload["scale"], "scale"),
        semantics=SemanticsConfig(
            operation="scaled-causal-row-softmax",
            mask="column-le-row-mod-cols",
            accumulation_dtype="float32",
            masked_output=0.0,
        ),
        input=InputConfig(
            distribution="normal",
            mean=_finite_float(input_payload["mean"], "input.mean"),
            standard_deviation=_positive_float(
                input_payload["standardDeviation"], "input.standardDeviation"
            ),
            stress_values=tuple(
                _finite_float(value, f"input.stressValues[{index}]")
                for index, value in enumerate(stress_values)
            ),
        ),
        launch_arguments=tuple(launch_arguments),
        correctness=CorrectnessConfig(
            seeds=tuple(seeds),
            atol=_non_negative_float(correctness_payload["atol"], "correctness.atol"),
            rtol=_non_negative_float(correctness_payload["rtol"], "correctness.rtol"),
        ),
        benchmark=BenchmarkConfig(
            warmup=_positive_int(benchmark_payload["warmup"], "benchmark.warmup"),
            samples=_positive_int(benchmark_payload["samples"], "benchmark.samples"),
            inner_repeats=_positive_int(
                benchmark_payload["innerRepeats"], "benchmark.innerRepeats"
            ),
            timeout_seconds=_positive_int(
                benchmark_payload["timeoutSeconds"], "benchmark.timeoutSeconds"
            ),
        ),
        optimization=OptimizationConfig(
            max_rounds=_positive_int(
                optimization_payload["maxRounds"], "optimization.maxRounds"
            ),
            patience=_positive_int(
                optimization_payload["patience"], "optimization.patience"
            ),
            min_improvement_fraction=min_improvement,
            max_source_chars=_positive_int(
                optimization_payload["maxSourceChars"],
                "optimization.maxSourceChars",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# schemaVersion=2: 通用 kernel 任务的解析
# ---------------------------------------------------------------------------

_VALID_TENSOR_KINDS = {"input", "output", "workspace", "scalar", "return"}
_VALID_TENSOR_DTYPES = {
    "bool",
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
}
_VALID_SCALAR_DTYPES = {"float32", "float64", "int32", "int64", "bool"}
_INTEGER_RANGES = {
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
}
_TENSOR_INTEGER_RANGES = {
    "int8": (-(2**7), 2**7 - 1),
    "uint8": (0, 2**8 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
}
_FLOAT_TENSOR_DTYPES = {"bfloat16", "float16", "float32", "float64"}
MAX_SCALAR_ABS = 2**31 - 1
_DTYPE_BYTES = {
    "bool": 1,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}
MAX_TENSORS = 32
MAX_TENSOR_RANK = 8
MAX_DIMENSION_EXTENT = 1_000_000
MAX_TENSOR_BYTES = 512 << 20
MAX_DECLARED_BYTES = 1 << 30
MAX_EVALUATOR_PEAK_BYTES = 3 << 30
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_allowed_keys(
    payload: dict[str, Any],
    required: set[str],
    optional: set[str],
    where: str,
) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing or extra:
        raise ValueError(f"{where}: missing={missing}, extra={extra}")


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{where} must be a valid identifier")
    if value.startswith("__"):
        raise ValueError(f"{where} must not be a dunder name")
    return value


def _non_empty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _parse_dimensions(
    payload: Any,
    shape: Shape,
) -> Mapping[str, int]:
    if payload is None:
        supplied: dict[str, Any] = {}
    elif isinstance(payload, dict):
        supplied = payload
    else:
        raise ValueError("dimensions must be a JSON object")

    dimensions: dict[str, int] = {"rows": shape.rows, "cols": shape.cols}
    for raw_name, raw_value in supplied.items():
        name = _identifier(raw_name, "dimensions key")
        value = _positive_int(raw_value, f"dimensions.{name}")
        if value > MAX_DIMENSION_EXTENT:
            raise ValueError(
                f"dimensions.{name} exceeds the safety cap "
                f"({value} > {MAX_DIMENSION_EXTENT})"
            )
        if name in {"rows", "cols"} and dimensions[name] != value:
            raise ValueError(f"dimensions.{name} conflicts with shape.{name}")
        dimensions[name] = value

    # These aliases preserve the original rows/cols fixture convention. Explicit
    # named dimensions take precedence for genuine M/K/N workloads.
    dimensions.setdefault("M", shape.rows)
    dimensions.setdefault("K", shape.cols)
    dimensions.setdefault("N", shape.cols)
    return MappingProxyType(dimensions)


def _parse_tensor_shape(
    raw_shape: Any,
    dimensions: Mapping[str, int],
    where: str,
) -> tuple[int, ...]:
    if raw_shape == "input":
        return (dimensions["rows"], dimensions["cols"])
    if isinstance(raw_shape, str):
        if raw_shape not in dimensions:
            raise ValueError(f"{where} references unknown dimension {raw_shape!r}")
        return (dimensions[raw_shape],)
    if not isinstance(raw_shape, list):
        raise ValueError(f"{where} must be 'input', a dimension name, or an array")

    resolved: list[int] = []
    for axis, raw_dimension in enumerate(raw_shape):
        axis_where = f"{where}[{axis}]"
        if isinstance(raw_dimension, str):
            if raw_dimension not in dimensions:
                raise ValueError(
                    f"{axis_where} references unknown dimension {raw_dimension!r}"
                )
            resolved.append(dimensions[raw_dimension])
        else:
            resolved.append(_positive_int(raw_dimension, axis_where))
    return tuple(resolved)


def _parse_scalar_literal(value: Any, dtype: str, where: str) -> bool | int | float:
    if dtype == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be a boolean for dtype=bool")
        return value
    if dtype in {"int32", "int64"}:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{where} must be an integer for dtype={dtype}")
        minimum, maximum = _INTEGER_RANGES[dtype]
        if not minimum <= value <= maximum:
            raise ValueError(f"{where} is out of range for dtype={dtype}")
        if abs(value) > MAX_SCALAR_ABS:
            raise ValueError(
                f"{where} exceeds the scalar safety cap {MAX_SCALAR_ABS}"
            )
        return value
    return _finite_float(value, where)


def _parse_scalar_expression(payload: Any, where: str) -> ScalarExpression:
    if not isinstance(payload, dict):
        raise ValueError(f"{where} must be a JSON object")
    op = payload.get("op")
    if op == "dimension":
        _require_keys(payload, {"op", "name"}, where)
        return ScalarExpression(op=op, name=_identifier(payload["name"], f"{where}.name"))
    if op == "numel":
        _require_keys(payload, {"op", "tensor"}, where)
        return ScalarExpression(
            op=op, tensor=_identifier(payload["tensor"], f"{where}.tensor")
        )
    if op == "stride":
        _require_keys(payload, {"op", "tensor", "axis"}, where)
        axis = payload["axis"]
        if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
            raise ValueError(f"{where}.axis must be a non-negative integer")
        return ScalarExpression(
            op=op,
            tensor=_identifier(payload["tensor"], f"{where}.tensor"),
            axis=axis,
        )
    raise ValueError(f"{where}.op must be one of ['dimension', 'numel', 'stride']")


def _initializer_integer_bound(
    value: Any,
    dimensions: Mapping[str, int],
    where: str,
) -> tuple[int, str | None]:
    if isinstance(value, str):
        if value not in dimensions:
            raise ValueError(f"{where} references unknown dimension {value!r}")
        return dimensions[value], value
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where} must be an integer or dimension name")
    return value, None


def _parse_tensor_initializer(
    payload: Any,
    *,
    dtype: str,
    dimensions: Mapping[str, int],
    where: str,
) -> tuple[TensorInitializer, set[str]]:
    if not isinstance(payload, dict):
        raise ValueError(f"{where} must be a JSON object")
    kind = payload.get("kind")
    referenced_dimensions: set[str] = set()

    if kind == "normal":
        _require_keys(payload, {"kind", "mean", "standardDeviation"}, where)
        if dtype not in _FLOAT_TENSOR_DTYPES:
            raise ValueError(f"{where} normal initializer requires a floating dtype")
        mean = _finite_float(payload["mean"], f"{where}.mean")
        standard_deviation = _finite_float(
            payload["standardDeviation"], f"{where}.standardDeviation"
        )
        if standard_deviation <= 0:
            raise ValueError(f"{where}.standardDeviation must be positive")
        return (
            TensorInitializer(
                kind=kind,
                mean=mean,
                standard_deviation=standard_deviation,
            ),
            referenced_dimensions,
        )
    if kind == "uniform":
        _require_keys(payload, {"kind", "low", "high"}, where)
        if dtype not in _FLOAT_TENSOR_DTYPES:
            raise ValueError(f"{where} uniform initializer requires a floating dtype")
        low = _finite_float(payload["low"], f"{where}.low")
        high = _finite_float(payload["high"], f"{where}.high")
        if low >= high:
            raise ValueError(f"{where}.low must be smaller than high")
        return (
            TensorInitializer(kind=kind, low=low, high=high),
            referenced_dimensions,
        )
    if kind == "integer":
        _require_keys(payload, {"kind", "low", "high"}, where)
        if dtype not in _TENSOR_INTEGER_RANGES:
            raise ValueError(f"{where} integer initializer requires an integer dtype")
        low, low_dimension = _initializer_integer_bound(
            payload["low"], dimensions, f"{where}.low"
        )
        high, high_dimension = _initializer_integer_bound(
            payload["high"], dimensions, f"{where}.high"
        )
        referenced_dimensions.update(
            name for name in (low_dimension, high_dimension) if name is not None
        )
        minimum, maximum = _TENSOR_INTEGER_RANGES[dtype]
        if low < minimum or high - 1 > maximum or low >= high:
            raise ValueError(
                f"{where} range [{low}, {high}) is invalid for dtype={dtype}"
            )
        return (
            TensorInitializer(kind=kind, low=low, high=high),
            referenced_dimensions,
        )
    if kind == "bernoulli":
        _require_keys(payload, {"kind", "probability"}, where)
        if dtype != "bool":
            raise ValueError(f"{where} bernoulli initializer requires dtype=bool")
        probability = _finite_float(payload["probability"], f"{where}.probability")
        if not 0 <= probability <= 1:
            raise ValueError(f"{where}.probability must be in [0, 1]")
        return (
            TensorInitializer(kind=kind, probability=probability),
            referenced_dimensions,
        )
    if kind == "constant":
        _require_keys(payload, {"kind", "value"}, where)
        value = payload["value"]
        if dtype == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{where}.value must be bool")
        elif dtype in _TENSOR_INTEGER_RANGES:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{where}.value must be an integer")
            minimum, maximum = _TENSOR_INTEGER_RANGES[dtype]
            if not minimum <= value <= maximum:
                raise ValueError(f"{where}.value is out of range for dtype={dtype}")
        else:
            value = _finite_float(value, f"{where}.value")
        return (
            TensorInitializer(kind=kind, value=value),
            referenced_dimensions,
        )
    raise ValueError(
        f"{where}.kind must be one of normal, uniform, integer, bernoulli, constant"
    )


def _parse_preflight_evidence(payload: Any) -> PreflightEvidence:
    if not isinstance(payload, dict):
        raise ValueError("reference.preflightEvidence must be a JSON object")
    _require_keys(
        payload,
        {"algorithm", "baselineSha256", "referenceSha256", "contractSha256"},
        "reference.preflightEvidence",
    )
    if payload["algorithm"] != "sha256":
        raise ValueError("reference.preflightEvidence.algorithm must be 'sha256'")
    hashes: dict[str, str] = {}
    for field_name in ("baselineSha256", "referenceSha256", "contractSha256"):
        value = payload[field_name]
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(
                f"reference.preflightEvidence.{field_name} must be 64 lowercase hex characters"
            )
        hashes[field_name] = value
    return PreflightEvidence(
        algorithm="sha256",
        baseline_sha256=hashes["baselineSha256"],
        reference_sha256=hashes["referenceSha256"],
        contract_sha256=hashes["contractSha256"],
    )


def _load_task_v2(payload: dict[str, Any]) -> TaskSpecV2:
    """Load the declarative schemaVersion=2 kernel contract."""

    _require_allowed_keys(
        payload,
        {
            "schemaVersion",
            "name",
            "description",
            "language",
            "entrypoint",
            "shape",
            "input",
            "tensors",
            "launchArguments",
            "reference",
            "correctness",
            "benchmark",
            "optimization",
        },
        {"dimensions", "costModel"},
        "task",
    )
    if payload["schemaVersion"] != 2:
        raise ValueError("only task schemaVersion 2 is supported here")
    if payload["language"] != "triton":
        raise ValueError("this tutorial implements only the triton language")

    name = _non_empty_string(payload["name"], "name")
    description = _non_empty_string(payload["description"], "description")
    entrypoint = _identifier(payload["entrypoint"], "entrypoint")

    shape_payload = payload["shape"]
    correctness_payload = payload["correctness"]
    input_payload = payload["input"]
    benchmark_payload = payload["benchmark"]
    optimization_payload = payload["optimization"]
    reference_payload = payload["reference"]
    for value, object_name in (
        (shape_payload, "shape"),
        (correctness_payload, "correctness"),
        (input_payload, "input"),
        (benchmark_payload, "benchmark"),
        (optimization_payload, "optimization"),
        (reference_payload, "reference"),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{object_name} must be a JSON object")

    _require_keys(shape_payload, {"rows", "cols"}, "shape")
    shape = Shape(
        rows=_positive_int(shape_payload["rows"], "shape.rows"),
        cols=_positive_int(shape_payload["cols"], "shape.cols"),
    )
    if shape.rows > MAX_DIMENSION_EXTENT or shape.cols > MAX_DIMENSION_EXTENT:
        raise ValueError(
            f"shape exceeds the per-dimension safety cap {MAX_DIMENSION_EXTENT}"
        )
    dimensions = _parse_dimensions(payload.get("dimensions"), shape)

    _require_allowed_keys(
        input_payload,
        {"distribution", "mean", "standardDeviation"},
        {"stressValues"},
        "input",
    )
    if input_payload["distribution"] != "normal":
        raise ValueError("input.distribution must be 'normal'")
    stress_values = (0.0, 0.0, 0.0)
    if "stressValues" in input_payload:
        raw_stress = input_payload["stressValues"]
        if not isinstance(raw_stress, list) or len(raw_stress) != 3:
            raise ValueError("input.stressValues must contain exactly three numbers")
        stress_values = tuple(
            _finite_float(value, f"input.stressValues[{index}]")
            for index, value in enumerate(raw_stress)
        )

    tensors_payload = payload["tensors"]
    if not isinstance(tensors_payload, list) or not tensors_payload:
        raise ValueError("tensors must be a non-empty array")
    if len(tensors_payload) > MAX_TENSORS:
        raise ValueError(
            f"tensors exceeds the safety cap ({len(tensors_payload)} > {MAX_TENSORS})"
        )
    tensors: list[TensorSpec] = []
    seen_names: set[str] = set()
    active_dimensions = "dimensions" in payload
    declared_dimension_names = (
        set(payload.get("dimensions", {})) | {"rows", "cols"}
        if active_dimensions
        else set()
    )
    used_dimensions: set[str] = set()
    for index, tensor_payload in enumerate(tensors_payload):
        where = f"tensors[{index}]"
        if not isinstance(tensor_payload, dict):
            raise ValueError(f"{where} must be a JSON object")
        if "kind" not in tensor_payload:
            raise ValueError(f"{where}: missing=['kind'], extra=[]")
        kind = tensor_payload["kind"]
        if kind not in _VALID_TENSOR_KINDS:
            raise ValueError(
                f"{where}.kind must be one of {sorted(_VALID_TENSOR_KINDS)}"
            )
        if kind == "scalar":
            _require_allowed_keys(
                tensor_payload,
                {"name", "kind"},
                {"dtype", "value", "expression"},
                where,
            )
        else:
            _require_allowed_keys(
                tensor_payload,
                {"name", "kind"},
                {"dtype", "shape", "initializer"},
                where,
            )

        tensor_name = _identifier(tensor_payload["name"], f"{where}.name")
        if tensor_name in seen_names:
            raise ValueError(f"{where}.name duplicate: {tensor_name!r}")
        seen_names.add(tensor_name)

        if kind == "scalar":
            has_value = "value" in tensor_payload
            has_expression = "expression" in tensor_payload
            if has_value == has_expression:
                raise ValueError(
                    f"{where} scalar must define exactly one of value or expression"
                )
            raw_dtype = tensor_payload.get("dtype")
            if raw_dtype is None:
                if has_expression:
                    dtype = "int64"
                elif isinstance(tensor_payload["value"], bool):
                    dtype = "bool"
                elif isinstance(tensor_payload["value"], int):
                    dtype = "int64"
                else:
                    dtype = "float64"
            elif isinstance(raw_dtype, str) and raw_dtype in _VALID_SCALAR_DTYPES:
                dtype = raw_dtype
            else:
                raise ValueError(
                    f"{where}.dtype must be one of {sorted(_VALID_SCALAR_DTYPES)}"
                )
            if has_expression and dtype not in {"int32", "int64"}:
                raise ValueError(f"{where}.dtype must be int32 or int64 for expression")
            expression = (
                _parse_scalar_expression(tensor_payload["expression"], f"{where}.expression")
                if has_expression
                else None
            )
            if expression is not None and expression.op == "dimension":
                assert expression.name is not None
                used_dimensions.add(expression.name)
            value = (
                _parse_scalar_literal(tensor_payload["value"], dtype, f"{where}.value")
                if has_value
                else None
            )
            tensors.append(
                TensorSpec(
                    name=tensor_name,
                    kind=kind,
                    dtype=dtype,
                    value=value,
                    expression=expression,
                )
            )
            continue

        raw_dtype = tensor_payload.get("dtype", "float16")
        if not isinstance(raw_dtype, str) or raw_dtype not in _VALID_TENSOR_DTYPES:
            raise ValueError(
                f"{where}.dtype must be one of {sorted(_VALID_TENSOR_DTYPES)}"
            )
        raw_shape = tensor_payload.get("shape", "input")
        if active_dimensions:
            if raw_shape == "input":
                used_dimensions.update(("rows", "cols"))
            elif isinstance(raw_shape, str):
                used_dimensions.add(raw_shape)
            elif isinstance(raw_shape, list):
                if any(not isinstance(axis, str) for axis in raw_shape):
                    raise ValueError(
                        f"{where}.shape must reference named dimensions; "
                        "literal extents are not allowed"
                    )
                used_dimensions.update(raw_shape)
        tensor_shape = _parse_tensor_shape(raw_shape, dimensions, f"{where}.shape")
        initializer: TensorInitializer | None = None
        if "initializer" in tensor_payload:
            if kind != "input":
                raise ValueError(f"{where}.initializer is allowed only for input tensors")
            initializer, initializer_dimensions = _parse_tensor_initializer(
                tensor_payload["initializer"],
                dtype=raw_dtype,
                dimensions=dimensions,
                where=f"{where}.initializer",
            )
            used_dimensions.update(initializer_dimensions)
        elif active_dimensions and kind == "input" and raw_dtype not in _FLOAT_TENSOR_DTYPES:
            raise ValueError(
                f"{where} with dtype={raw_dtype} requires an explicit initializer"
            )
        tensors.append(
            TensorSpec(
                name=tensor_name,
                kind=kind,
                dtype=raw_dtype,
                shape=tensor_shape,
                initializer=initializer,
            )
        )

    if active_dimensions:
        unused_dimensions = sorted(declared_dimension_names - used_dimensions)
        if unused_dimensions:
            raise ValueError(
                "every declared dimension must drive a tensor shape, input initializer, "
                "or scalar binding; "
                f"unused={unused_dimensions}"
            )

    tensors_by_name = {tensor.name: tensor for tensor in tensors}
    declared_bytes = 0
    logical_io_bytes = 0
    for index, tensor in enumerate(tensors):
        if tensor.kind == "scalar":
            continue
        assert tensor.shape is not None and tensor.dtype is not None
        if len(tensor.shape) > MAX_TENSOR_RANK:
            raise ValueError(
                f"tensors[{index}].shape rank exceeds {MAX_TENSOR_RANK}"
            )
        tensor_bytes = math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
        if tensor_bytes > MAX_TENSOR_BYTES:
            raise ValueError(
                f"tensors[{index}] exceeds the per-tensor allocation cap"
            )
        declared_bytes += tensor_bytes
        if tensor.kind in {"input", "output", "return"}:
            logical_io_bytes += tensor_bytes
    if declared_bytes > MAX_DECLARED_BYTES:
        raise ValueError(
            "tensors exceed the total declared allocation cap "
            f"({declared_bytes} > {MAX_DECLARED_BYTES} bytes)"
        )
    input_bytes = sum(
        math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
        for tensor in tensors
        if tensor.kind == "input"
        and tensor.shape is not None
        and tensor.dtype is not None
    )
    observable_bytes = sum(
        math.prod(tensor.shape) * _DTYPE_BYTES[tensor.dtype]
        for tensor in tensors
        if tensor.kind in {"output", "return"}
        and tensor.shape is not None
        and tensor.dtype is not None
    )
    evaluator_peak_bytes = 2 * (declared_bytes + input_bytes) + observable_bytes
    if evaluator_peak_bytes > MAX_EVALUATOR_PEAK_BYTES:
        raise ValueError(
            "task exceeds the evaluator peak-memory safety budget "
            f"({evaluator_peak_bytes} > {MAX_EVALUATOR_PEAK_BYTES} bytes)"
        )

    cost_model: AnalyticCostModel | None = None
    if "costModel" in payload:
        raw_cost_model = payload["costModel"]
        if not isinstance(raw_cost_model, dict):
            raise ValueError("costModel must be a JSON object")
        _require_keys(raw_cost_model, {"flops", "bytes", "label"}, "costModel")
        flops = _positive_int(raw_cost_model["flops"], "costModel.flops")
        bytes_ = _positive_int(raw_cost_model["bytes"], "costModel.bytes")
        if flops > 2**63 - 1 or bytes_ > 2**63 - 1:
            raise ValueError("costModel flops/bytes must fit in signed int64")
        label = _non_empty_string(raw_cost_model["label"], "costModel.label")
        if len(label) > 200:
            raise ValueError("costModel.label must be at most 200 characters")
        if bytes_ < logical_io_bytes:
            raise ValueError(
                "costModel.bytes cannot be smaller than declared logical "
                f"input/output traffic ({logical_io_bytes} bytes)"
            )
        cost_model = AnalyticCostModel(flops=flops, bytes=bytes_, label=label)
    if "dimensions" in payload:
        data_shapes = {
            tensor.shape
            for tensor in tensors
            if tensor.kind in {"input", "output", "return"} and tensor.shape is not None
        }
        if (shape.rows, shape.cols) not in data_shapes:
            raise ValueError(
                "active declarative task must bind shape.rows/cols to at least "
                "one input/output/return tensor"
            )
    for index, tensor in enumerate(tensors):
        expression = tensor.expression
        if expression is None:
            continue
        where = f"tensors[{index}].expression"
        if expression.op == "dimension":
            assert expression.name is not None
            if expression.name not in dimensions:
                raise ValueError(
                    f"{where} references unknown dimension {expression.name!r}"
                )
            continue
        assert expression.tensor is not None
        target = tensors_by_name.get(expression.tensor)
        if target is None:
            raise ValueError(f"{where} references unknown tensor {expression.tensor!r}")
        if target.kind in {"scalar", "return"}:
            raise ValueError(f"{where} must reference an allocated tensor")
        if expression.op == "stride":
            assert expression.axis is not None and target.shape is not None
            if expression.axis >= len(target.shape):
                raise ValueError(
                    f"{where}.axis {expression.axis} is out of range for {target.name}"
                )

    launch_arguments = payload["launchArguments"]
    if not isinstance(launch_arguments, list) or any(
        not isinstance(argument, str) for argument in launch_arguments
    ):
        raise ValueError("launchArguments must be an array of strings")
    normalized_launch_arguments = tuple(
        _identifier(argument, f"launchArguments[{index}]")
        for index, argument in enumerate(launch_arguments)
    )
    if len(normalized_launch_arguments) != len(set(normalized_launch_arguments)):
        raise ValueError("launchArguments must contain unique names")
    expected_arguments = {tensor.name for tensor in tensors if tensor.kind != "return"}
    actual_arguments = set(normalized_launch_arguments)
    if actual_arguments != expected_arguments:
        missing = sorted(expected_arguments - actual_arguments)
        extra = sorted(actual_arguments - expected_arguments)
        raise ValueError(
            f"launchArguments must exactly cover non-return tensors: missing={missing}, extra={extra}"
        )
    if not any(tensor.kind in {"output", "return"} for tensor in tensors):
        raise ValueError("tensors must contain at least one output or return tensor")

    _require_allowed_keys(
        reference_payload,
        {"filename", "entrypoint"},
        {"selfCheckPassed", "preflightEvidence"},
        "reference",
    )
    reference_filename = reference_payload["filename"]
    if (
        not isinstance(reference_filename, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", reference_filename)
        or reference_filename in {".", ".."}
    ):
        raise ValueError("reference.filename must be a simple filename")
    reference_entrypoint = _identifier(
        reference_payload["entrypoint"], "reference.entrypoint"
    )
    self_check_passed = reference_payload.get("selfCheckPassed", False)
    if not isinstance(self_check_passed, bool):
        raise ValueError("reference.selfCheckPassed must be a boolean")
    evidence = (
        _parse_preflight_evidence(reference_payload["preflightEvidence"])
        if "preflightEvidence" in reference_payload
        else None
    )
    active_declarative_task = "dimensions" in payload
    if evidence is not None and not self_check_passed:
        raise ValueError(
            "reference.preflightEvidence requires reference.selfCheckPassed=true"
        )
    if active_declarative_task and self_check_passed and evidence is None:
        raise ValueError(
            "active declarative tasks with selfCheckPassed=true require preflightEvidence"
        )

    _require_keys(correctness_payload, {"seeds", "atol", "rtol"}, "correctness")
    seeds = correctness_payload["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("correctness.seeds must be a non-empty integer array")
    _require_keys(
        benchmark_payload,
        {"warmup", "samples", "innerRepeats", "timeoutSeconds"},
        "benchmark",
    )
    _require_keys(
        optimization_payload,
        {"maxRounds", "patience", "minImprovementFraction", "maxSourceChars"},
        "optimization",
    )
    min_improvement = _non_negative_float(
        optimization_payload["minImprovementFraction"],
        "optimization.minImprovementFraction",
    )
    if min_improvement >= 1:
        raise ValueError("optimization.minImprovementFraction must be less than 1")

    return TaskSpecV2(
        schema_version=2,
        name=name,
        description=description,
        language="triton",
        entrypoint=entrypoint,
        shape=shape,
        dimensions=dimensions,
        cost_model=cost_model,
        input=InputConfig(
            distribution="normal",
            mean=_finite_float(input_payload["mean"], "input.mean"),
            standard_deviation=_positive_float(
                input_payload["standardDeviation"], "input.standardDeviation"
            ),
            stress_values=stress_values,
        ),
        tensors=tuple(tensors),
        launch_arguments=normalized_launch_arguments,
        reference=ReferenceConfig(
            filename=reference_filename,
            entrypoint=reference_entrypoint,
            self_check_passed=self_check_passed,
            preflight_evidence=evidence,
        ),
        correctness=CorrectnessConfig(
            seeds=tuple(seeds),
            atol=_non_negative_float(correctness_payload["atol"], "correctness.atol"),
            rtol=_non_negative_float(correctness_payload["rtol"], "correctness.rtol"),
        ),
        benchmark=BenchmarkConfig(
            warmup=_positive_int(benchmark_payload["warmup"], "benchmark.warmup"),
            samples=_positive_int(benchmark_payload["samples"], "benchmark.samples"),
            inner_repeats=_positive_int(
                benchmark_payload["innerRepeats"], "benchmark.innerRepeats"
            ),
            timeout_seconds=_positive_int(
                benchmark_payload["timeoutSeconds"], "benchmark.timeoutSeconds"
            ),
        ),
        optimization=OptimizationConfig(
            max_rounds=_positive_int(
                optimization_payload["maxRounds"], "optimization.maxRounds"
            ),
            patience=_positive_int(
                optimization_payload["patience"], "optimization.patience"
            ),
            min_improvement_fraction=min_improvement,
            max_source_chars=_positive_int(
                optimization_payload["maxSourceChars"],
                "optimization.maxSourceChars",
            ),
        ),
    )

