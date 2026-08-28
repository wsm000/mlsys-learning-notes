"""Structural allowlist for the tutorial's generated Triton candidate.

This deliberately small policy keeps Python host code stateless: a candidate
may define Triton JIT functions and a ``launch`` function that only computes
launch metadata and dispatches those kernels. It is a guardrail, not a security
sandbox; generated GPU code still belongs on a dedicated experiment account.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Mapping


LAUNCH_ARGUMENTS = (
    "x",
    "output",
    "exp_values",
    "row_max",
    "row_sum",
    "scale",
    "rows",
    "cols",
)
PURE_HOST_CALLS = {"float", "int", "max", "min"}
TRITON_HOST_CALLS = {"cdiv", "next_power_of_2"}
KERNEL_BUILTINS = {"float", "int", "range"}
FORBIDDEN_TRITON_CALLS = {
    "device_assert",
    "device_print",
    "inline_asm_elementwise",
    "static_print",
}
TORCH_HOST_CALLS = {
    "empty",
    "empty_like",
}
TORCH_HOST_ATTRIBUTES = {
    "bool",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
}
TENSOR_HOST_ATTRIBUTES = {"device", "dtype", "ndim", "shape"}
TENSOR_HOST_CALLS = {"numel", "stride"}
MAX_HOST_LITERAL_INT = 2**31 - 1
MAX_HOST_CONTAINER_ITEMS = 64
MAX_HOST_STRING_CHARS = 256
MAX_HOST_COMPUTED_MAGNITUDE = 2**63 - 1


def _numeric_magnitude(
    node: ast.AST,
    bounds: Mapping[str, float],
) -> float | None:
    """Conservative magnitude propagation for host-side scalar metadata."""

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool) or not math.isfinite(float(node.value)):
            return None
        return abs(float(node.value))
    if isinstance(node, ast.Name):
        return bounds.get(node.id)
    if isinstance(node, ast.UnaryOp):
        return _numeric_magnitude(node.operand, bounds)
    if isinstance(node, ast.BinOp):
        left = _numeric_magnitude(node.left, bounds)
        right = _numeric_magnitude(node.right, bounds)
        if left is None or right is None:
            return None
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            return left * max(1.0, right)
        if isinstance(node.op, ast.Mod):
            return right
        return None
    if isinstance(node, ast.IfExp):
        body = _numeric_magnitude(node.body, bounds)
        other = _numeric_magnitude(node.orelse, bounds)
        if body is None or other is None:
            return None
        return max(body, other)
    if isinstance(node, (ast.BoolOp, ast.Compare)):
        return 1.0
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            return float(MAX_HOST_LITERAL_INT)
        return None
    if isinstance(node, ast.Attribute) and node.attr == "ndim":
        return 8.0
    if isinstance(node, ast.Call):
        argument_bounds = [
            _numeric_magnitude(argument, bounds) for argument in node.args
        ]
        if any(value is None for value in argument_bounds):
            return None
        values = [float(value) for value in argument_bounds if value is not None]
        if isinstance(node.func, ast.Name) and node.func.id in PURE_HOST_CALLS:
            return max(values, default=0.0)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"numel", "stride"}:
                return float(MAX_HOST_LITERAL_INT)
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "triton"
            ):
                factor = 2.0 if node.func.attr == "next_power_of_2" else 1.0
                return factor * max(values, default=0.0)
    return None


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_attribute(node: ast.AST | None, root: str, attribute: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == root
        and node.attr == attribute
    )


def _validate_arguments(
    function: ast.FunctionDef,
    *,
    launch: bool,
    errors: list[str],
    expected_launch_args: tuple[str, ...] | None = None,
) -> None:
    arguments = function.args
    if (
        arguments.posonlyargs
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kw_defaults
        or arguments.kwarg is not None
        or arguments.defaults
    ):
        errors.append(f"{function.name} must use plain positional arguments")

    names = tuple(argument.arg for argument in arguments.args)
    if launch:
        # v1: expected_launch_args 默认是 LAUNCH_ARGUMENTS（softmax 8 参数）
        # v2: 传入 task 的 launchArguments；None 表示不检查具体名字
        expected = expected_launch_args if expected_launch_args is not None else LAUNCH_ARGUMENTS
        if names != expected:
            errors.append(
                f"{function.name} must use the expected argument ABI: "
                f"expected {list(expected)}, got {list(names)}"
            )

    for argument in arguments.args:
        if argument.arg.startswith("__"):
            errors.append(f"dunder name is not allowed: {argument.arg}")
        if launch:
            if argument.annotation is not None:
                errors.append("launch arguments must not have annotations")
        elif argument.annotation is not None and not _is_attribute(
            argument.annotation, "tl", "constexpr"
        ):
            errors.append(
                f"kernel annotation is not allowed for {function.name}.{argument.arg}"
            )
    if function.returns is not None:
        errors.append(f"return annotation is not allowed: {function.name}")


class KernelBodyPolicy(ast.NodeVisitor):
    """Validate code that Triton compiles, without executing arbitrary Python."""

    def __init__(self, function_name: str, jit_names: set[str]) -> None:
        self.function_name = function_name
        self.jit_names = jit_names
        self.errors: list[str] = []

    def _reject(self, node: ast.AST, reason: str) -> None:
        self.errors.append(f"{self.function_name}: {reason} at line {node.lineno}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self._reject(node, f"dunder name is not allowed: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self._reject(node, f"private attribute is not allowed: {node.attr}")
        elif isinstance(node.value, ast.Name) and node.value.id == "tl":
            pass
        elif node.attr == "to":
            pass
        else:
            self._reject(node, f"attribute is not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        allowed = False
        if isinstance(node.func, ast.Name):
            allowed = node.func.id in KERNEL_BUILTINS or node.func.id in self.jit_names
        elif isinstance(node.func, ast.Attribute):
            allowed = (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl"
                and not node.func.attr.startswith("_")
                and node.func.attr not in FORBIDDEN_TRITON_CALLS
            ) or node.func.attr == "to"
        if not allowed:
            self._reject(node, "call is not in the Triton kernel allowlist")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self._reject(node, "assignment target must be one local name")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._reject(node, "annotated assignment is not allowed")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            self._reject(node, "augmented assignment target must be one local name")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._reject(node, "nested function is not allowed")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._reject(node, "class is not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._reject(node, "lambda is not allowed")

    def visit_Import(self, node: ast.Import) -> None:
        self._reject(node, "function-local import is not allowed")

    visit_ImportFrom = visit_Import

    def visit_Global(self, node: ast.Global) -> None:
        self._reject(node, "global state is not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._reject(node, "nonlocal state is not allowed")

    def visit_With(self, node: ast.With) -> None:
        self._reject(node, "with statement is not allowed")

    visit_AsyncWith = visit_With

    def visit_Try(self, node: ast.Try) -> None:
        self._reject(node, "try statement is not allowed")

    visit_TryStar = visit_Try

    def visit_Raise(self, node: ast.Raise) -> None:
        self._reject(node, "raise statement is not allowed")

    def visit_Assert(self, node: ast.Assert) -> None:
        self._reject(node, "assert statement is not allowed")

    def visit_Delete(self, node: ast.Delete) -> None:
        self._reject(node, "delete statement is not allowed")

    def visit_While(self, node: ast.While) -> None:
        self._reject(node, "while loop is not allowed")

    def visit_Await(self, node: ast.Await) -> None:
        self._reject(node, "await is not allowed")

    def visit_Yield(self, node: ast.Yield) -> None:
        self._reject(node, "yield is not allowed")

    visit_YieldFrom = visit_Yield

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._reject(node, "assignment expression is not allowed")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._reject(node, "comprehension is not allowed")

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_Match(self, node: ast.Match) -> None:
        self._reject(node, "match statement is not allowed")


class LaunchBodyPolicy:
    """Validate the host entrypoint as stateless launch-metadata code."""

    def __init__(
        self,
        jit_names: set[str],
        *,
        allow_torch: bool = False,
        tensor_arguments: set[str] | None = None,
        scalar_bounds: Mapping[str, float] | None = None,
    ) -> None:
        self.jit_names = jit_names
        self.allow_torch = allow_torch
        self.tensor_names = set(tensor_arguments or ())
        self.container_names: set[str] = set()
        self.numeric_bounds = dict(scalar_bounds or {})
        self.errors: list[str] = []
        self.dispatches = 0

    def _reject(self, node: ast.AST, reason: str) -> None:
        self.errors.append(f"launch: {reason} at line {node.lineno}")

    def validate(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._statement(statement)
        if self.dispatches == 0:
            self.errors.append("launch must dispatch at least one Triton JIT kernel")

    def _statement(self, node: ast.stmt) -> None:
        if _is_docstring(node):
            return
        if isinstance(node, ast.Assign):
            # v1: 只允许单名赋值（M = ...）
            # v2 (allow_torch): 允许元组解包（M, K = a.shape）和单名
            if self.allow_torch and len(node.targets) == 1 and isinstance(node.targets[0], (ast.Tuple, ast.List)):
                if not (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"
                ):
                    self._reject(node, "tuple unpacking is limited to tensor.shape metadata")
                    return
                for elt in node.targets[0].elts:
                    if not isinstance(elt, ast.Name) or elt.id.startswith("__"):
                        self._reject(node, "assignment target must be plain names")
                        return
                    self.numeric_bounds[elt.id] = float(MAX_HOST_LITERAL_INT)
                self._expression(node.value)
                return
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                self._reject(node, "assignment target must be one local name")
                return
            if node.targets[0].id.startswith("__"):
                self._reject(node, "dunder local name is not allowed")
            self._expression(node.value)
            target_name = node.targets[0].id
            if self._is_tensor_expression(node.value):
                self.tensor_names.add(target_name)
            else:
                self.tensor_names.discard(target_name)
            if self._is_container_expression(node.value):
                self.container_names.add(target_name)
            else:
                self.container_names.discard(target_name)
            numeric_bound = _numeric_magnitude(node.value, self.numeric_bounds)
            if numeric_bound is not None:
                self.numeric_bounds[target_name] = numeric_bound
                if numeric_bound > MAX_HOST_COMPUTED_MAGNITUDE:
                    self._reject(node, "computed host metadata exceeds the numeric limit")
            else:
                self.numeric_bounds.pop(target_name, None)
            return
        if isinstance(node, ast.If):
            self._reject(node, "conditional host control flow is not allowed")
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if self._dispatch(node.value):
                self.dispatches += 1
            else:
                self._reject(node, "only Triton JIT dispatch calls may be statements")
            return
        # v2 host 入口允许 return（naive_gemm 等 host 函数会 return tensor）
        if self.allow_torch and isinstance(node, ast.Return):
            if self.dispatches == 0:
                self._reject(node, "return must occur after a Triton JIT dispatch")
            if node.value is not None:
                self._expression(node.value)
            return
        self._reject(node, f"statement is not allowed: {type(node).__name__}")

    def _dispatch(self, node: ast.Call) -> bool:
        function = node.func
        if not (
            isinstance(function, ast.Subscript)
            and isinstance(function.value, ast.Name)
            and function.value.id in self.jit_names
        ):
            return False
        self._expression(function.slice)
        for argument in node.args:
            self._expression(argument)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("__"):
                self._reject(keyword, "expanded or dunder keyword is not allowed")
            self._expression(keyword.value)
        return True

    def _expression(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                self._reject(node, f"dunder name is not allowed: {node.id}")
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool, str, type(None))):
                self._reject(node, "constant type is not allowed")
            elif isinstance(node.value, int) and not isinstance(node.value, bool) and abs(node.value) > MAX_HOST_LITERAL_INT:
                self._reject(node, "integer literal exceeds the host metadata limit")
            elif isinstance(node.value, str) and len(node.value) > MAX_HOST_STRING_CHARS:
                self._reject(node, "string literal exceeds the host metadata limit")
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            if len(node.elts) > MAX_HOST_CONTAINER_ITEMS:
                self._reject(node, "host container literal is too large")
            for element in node.elts:
                self._expression(element)
            return
        if isinstance(node, ast.Lambda):
            arguments = node.args
            if (
                len(arguments.args) != 1
                or arguments.posonlyargs
                or arguments.vararg is not None
                or arguments.kwonlyargs
                or arguments.kw_defaults
                or arguments.kwarg is not None
                or arguments.defaults
                or arguments.args[0].arg.startswith("__")
            ):
                self._reject(
                    node,
                    "grid lambda must have exactly one plain positional argument",
                )
                return
            if not isinstance(node.body, (ast.Tuple, ast.List)):
                self._reject(node, "grid lambda must return a tuple/list of launch extents")
                return
            allow_torch = self.allow_torch
            self.allow_torch = False
            try:
                self._expression(node.body)
            finally:
                self.allow_torch = allow_torch
            return
        if isinstance(node, ast.Subscript):
            self._expression(node.value)
            self._expression(node.slice)
            return
        if isinstance(node, ast.Slice):
            for value in (node.lower, node.upper, node.step):
                if value is not None:
                    self._expression(value)
            return
        if isinstance(node, ast.IfExp):
            if any(
                self._is_tensor_expression(child)
                for child in (node.test, node.body, node.orelse)
            ):
                self._reject(
                    node,
                    "conditional expressions may only select scalar launch metadata",
                )
            self._expression(node.test)
            self._expression(node.body)
            self._expression(node.orelse)
            return
        if isinstance(node, ast.UnaryOp):
            if self._is_tensor_expression(node.operand):
                self._reject(node, "tensor arithmetic is not allowed in the host entrypoint")
            self._expression(node.operand)
            return
        if isinstance(node, (ast.BinOp, ast.BoolOp, ast.Compare)):
            if isinstance(node, ast.BinOp) and (
                isinstance(node.op, (ast.Pow, ast.LShift, ast.RShift))
                or self._is_container_expression(node.left)
                or self._is_container_expression(node.right)
            ):
                self._reject(
                    node,
                    "container repetition and unbounded integer operators are not allowed",
                )
            numeric_bound = _numeric_magnitude(node, self.numeric_bounds)
            if (
                numeric_bound is not None
                and numeric_bound > MAX_HOST_COMPUTED_MAGNITUDE
            ):
                self._reject(node, "computed host metadata exceeds the numeric limit")
            if any(
                self._is_tensor_expression(child)
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.expr)
            ):
                self._reject(node, "tensor arithmetic is not allowed in the host entrypoint")
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self._expression(child)
            return
        if isinstance(node, ast.Attribute):
            if self.allow_torch:
                if node.attr.startswith("_"):
                    self._reject(node, f"private attribute is not allowed: {node.attr}")
                elif isinstance(node.value, ast.Name) and node.value.id == "torch":
                    if node.attr not in TORCH_HOST_ATTRIBUTES:
                        self._reject(node, f"torch attribute is not allowed: {node.attr}")
                elif node.attr in TENSOR_HOST_ATTRIBUTES | TENSOR_HOST_CALLS:
                    self._expression(node.value)
                else:
                    self._reject(node, f"tensor attribute is not allowed: {node.attr}")
                return
            self._reject(node, f"attribute is not allowed: {node.attr}")
            return
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in PURE_HOST_CALLS:
                pass
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "triton"
                and node.func.attr in TRITON_HOST_CALLS
            ):
                pass
            elif self.allow_torch and self._is_allowed_torch_call(node.func):
                # v2 host 入口只允许空输出分配和 shape/stride 查询。
                # zeros/ones 会把 PyTorch 计算伪装成 Triton return 输出。
                pass
            else:
                self._reject(node, "call is not in the stateless host allowlist")
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in PURE_HOST_CALLS
                and any(self._is_tensor_expression(argument) for argument in node.args)
            ):
                self._reject(node, "host scalar conversion may not consume a tensor")
            for argument in node.args:
                self._expression(argument)
            for keyword in node.keywords:
                if keyword.arg is None:
                    self._reject(keyword, "expanded keyword is not allowed")
                self._expression(keyword.value)
            return
        self._reject(node, f"expression is not allowed: {type(node).__name__}")

    def _is_tensor_expression(self, node: ast.AST) -> bool:
        """Conservatively classify expressions that carry tensor data."""

        if isinstance(node, ast.Name):
            return node.id in self.tensor_names
        if isinstance(node, (ast.Tuple, ast.List)):
            return any(self._is_tensor_expression(element) for element in node.elts)
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            ):
                return False
            if node.attr in TENSOR_HOST_ATTRIBUTES | TENSOR_HOST_CALLS:
                return False
            return self._is_tensor_expression(node.value)
        if isinstance(node, ast.Subscript):
            return self._is_tensor_expression(node.value)
        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr in TORCH_HOST_CALLS
            )
        if isinstance(node, (ast.UnaryOp, ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp)):
            return any(
                self._is_tensor_expression(child)
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.expr)
            )
        return False

    def _is_container_expression(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.container_names
        if isinstance(node, (ast.Tuple, ast.List)):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return True
        if isinstance(node, ast.IfExp):
            return self._is_container_expression(
                node.body
            ) or self._is_container_expression(node.orelse)
        return False

    def _is_allowed_torch_call(self, func: ast.expr) -> bool:
        """Return whether a v2 host call is a deterministic allocation/query."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "torch" and func.attr in TORCH_HOST_CALLS:
                return True
            if func.value.id != "torch" and func.attr in TENSOR_HOST_CALLS:
                return True
        return False


def _validate_import(node: ast.Import, errors: list[str], *, allow_torch: bool = False) -> None:
    for alias in node.names:
        allowed = (alias.name == "triton" and alias.asname is None) or (
            alias.name == "triton.language" and alias.asname == "tl"
        )
        if allow_torch and alias.name == "torch" and alias.asname is None:
            allowed = True
        if not allowed:
            rendered = alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
            errors.append(f"import is not allowed: {rendered}")


def _is_triton_jit(node: ast.AST) -> bool:
    return _is_attribute(node, "triton", "jit")


def validate_candidate(
    path: Path,
    max_source_chars: int,
    *,
    expected_launch_args: tuple[str, ...] | None = None,
    entrypoint: str | None = None,
    allow_torch: bool = False,
    tensor_arguments: tuple[str, ...] | None = None,
    scalar_bounds: Mapping[str, float] | None = None,
) -> list[str]:
    """校验候选源码。

    expected_launch_args:
        - None（默认）：launch 必须用 v1 的 8 参数 softmax ABI（LAUNCH_ARGUMENTS）
        - 传 tuple：launch/host 入口必须用这个 ABI（v2 从 task.launch_arguments 传入）
    entrypoint:
        - None（默认，v1）：入口函数必须叫 launch
        - 传字符串（v2）：入口函数叫这个名字（如 naive_gemm）
    allow_torch:
        - False（v1）：host 入口不允许 import torch / 调 torch（evaluator 自己分配张量）
        - True（v2）：host 入口只允许受 worker 预算约束的 empty/empty_like
          输出分配和 shape/stride 查询
    """
    source = path.read_text(encoding="utf-8")
    if len(source) > max_source_chars:
        return [f"candidate exceeds {max_source_chars} characters"]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [f"syntax error: {error.msg} at line {error.lineno}"]

    errors: list[str] = []
    functions: list[ast.FunctionDef] = []
    imports: list[tuple[str, str | None]] = []
    for statement in tree.body:
        if _is_docstring(statement):
            continue
        if isinstance(statement, ast.Import):
            _validate_import(statement, errors, allow_torch=allow_torch)
            imports.extend((alias.name, alias.asname) for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            errors.append(f"from-import is not allowed: {statement.module}")
        elif isinstance(statement, ast.FunctionDef):
            functions.append(statement)
        else:
            errors.append(f"top-level statement is not allowed: {type(statement).__name__}")

    names = [function.name for function in functions]
    if imports.count(("triton", None)) != 1:
        errors.append("candidate must contain exactly one 'import triton'")
    if imports.count(("triton.language", "tl")) != 1:
        errors.append("candidate must contain exactly one 'import triton.language as tl'")
    if imports.count(("torch", None)) > 1:
        errors.append("candidate may contain at most one 'import torch'")
    if len(names) != len(set(names)):
        errors.append("function names must be unique")
    if any(name.startswith("__") for name in names):
        errors.append("dunder function names are not allowed")

    # 区分 kernel 和 host 函数：
    # - v1：名字叫 launch 的是 host，其它全是 kernel
    # - v2：带 @triton.jit 装饰器的是 kernel，不带的是 host；host 入口由 entrypoint 指定
    if entrypoint is not None:
        # v2：按装饰器区分
        jit_functions = [f for f in functions if _has_triton_jit_decorator(f)]
        host_functions = [f for f in functions if not _has_triton_jit_decorator(f)]
        entry_functions = [f for f in host_functions if f.name == entrypoint]
        other_host = [f for f in host_functions if f.name != entrypoint]
    else:
        # v1：按名字区分
        entry_functions = [f for f in functions if f.name == "launch"]
        jit_functions = [f for f in functions if f.name != "launch"]
        other_host = []

    jit_names = {function.name for function in jit_functions}
    if len(entry_functions) != 1:
        errors.append(
            f"candidate must define exactly one entry function named "
            f"{entrypoint or 'launch'}"
        )
    if not jit_functions:
        errors.append("candidate must define at least one Triton JIT kernel")

    # kernel（@triton.jit）：严格检查
    for function in jit_functions:
        _validate_arguments(function, launch=False, errors=errors)
        if len(function.decorator_list) != 1 or not _is_triton_jit(
            function.decorator_list[0]
        ):
            errors.append(f"{function.name} must have exactly @triton.jit")
        body_policy = KernelBodyPolicy(function.name, jit_names)
        for statement in function.body:
            body_policy.visit(statement)
        errors.extend(body_policy.errors)

    # entry host 函数：检查 ABI + body（v2 允许 torch）
    for function in entry_functions:
        _validate_arguments(
            function, launch=True, errors=errors, expected_launch_args=expected_launch_args
        )
        if _has_triton_jit_decorator(function):
            errors.append(f"{function.name} (entry) must not have @triton.jit")
        launch_policy = LaunchBodyPolicy(
            jit_names,
            allow_torch=allow_torch,
            tensor_arguments=set(tensor_arguments or ()),
            scalar_bounds=scalar_bounds,
        )
        launch_policy.validate(function.body)
        errors.extend(launch_policy.errors)

    # Host helpers can hide arbitrary Python behind a harmless-looking call.
    # Keep the v2 host surface to one audited entrypoint.
    for function in other_host:
        errors.append(f"host helper function is not allowed: {function.name}")

    return sorted(set(errors))


def _has_triton_jit_decorator(function: ast.FunctionDef) -> bool:
    """检查函数有没有 @triton.jit 装饰器。"""
    return any(_is_triton_jit(dec) for dec in function.decorator_list)


REFERENCE_BUILTIN_CALLS = {"abs", "bool", "float", "int", "max", "min"}
REFERENCE_TORCH_CALLS = {
    "abs",
    "add",
    "amax",
    "amin",
    "arange",
    "bmm",
    "cat",
    "clamp",
    "clone",
    "div",
    "einsum",
    "empty_like",
    "exp",
    "full_like",
    "matmul",
    "maximum",
    "mean",
    "minimum",
    "mul",
    "mm",
    "ones_like",
    "relu",
    "rsqrt",
    "sigmoid",
    "softmax",
    "sqrt",
    "stack",
    "sum",
    "sub",
    "tanh",
    "tril",
    "triu",
    "where",
    "zeros_like",
}
REFERENCE_TORCH_ATTRIBUTES = {
    "bool",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
}
REFERENCE_TENSOR_CALLS = {
    "abs",
    "add",
    "amax",
    "amin",
    "clamp",
    "clamp_min",
    "clamp_max",
    "clone",
    "contiguous",
    "div",
    "exp",
    "float",
    "half",
    "masked_fill",
    "max",
    "mean",
    "min",
    "numel",
    "mul",
    "permute",
    "pow",
    "remainder",
    "reshape",
    "rsqrt",
    "softmax",
    "sqrt",
    "squeeze",
    "stride",
    "sum",
    "sub",
    "to",
    "transpose",
    "unsqueeze",
    "view",
}
REFERENCE_TENSOR_ATTRIBUTES = {
    "T",
    "device",
    "dtype",
    "indices",
    "ndim",
    "shape",
    "values",
}


class ReferenceBodyPolicy:
    """Allow pure tensor expressions while rejecting Python side effects."""

    def __init__(
        self,
        function_name: str,
        *,
        scalar_bounds: Mapping[str, float] | None = None,
    ) -> None:
        self.function_name = function_name
        self.errors: list[str] = []
        self.returns = 0
        self.container_names: set[str] = set()
        self.numeric_bounds = dict(scalar_bounds or {})

    def _reject(self, node: ast.AST, reason: str) -> None:
        self.errors.append(
            f"{self.function_name}: {reason} at line {getattr(node, 'lineno', '?')}"
        )

    @staticmethod
    def _is_dtype_expression(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and (
                (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "torch"
                    and node.attr in REFERENCE_TORCH_ATTRIBUTES
                )
                or node.attr == "dtype"
            )
        )

    def _validate_tensor_to(self, node: ast.Call) -> None:
        """Allow dtype conversion only; device transfers bypass memory accounting."""

        if len(node.args) == 1 and not node.keywords:
            if not self._is_dtype_expression(node.args[0]):
                self._reject(node, "tensor.to only permits an explicit dtype")
            return
        if (
            not node.args
            and len(node.keywords) == 1
            and node.keywords[0].arg == "dtype"
            and self._is_dtype_expression(node.keywords[0].value)
        ):
            return
        self._reject(node, "tensor.to only permits one explicit dtype argument")

    def validate(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._statement(statement)
        if self.returns == 0:
            self.errors.append(f"{self.function_name} must return the reference outputs")

    def _statement(self, node: ast.stmt) -> None:
        if _is_docstring(node):
            return
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                self._reject(node, "assignment must have one plain-name target")
                return
            target = node.targets[0]
            targets = [target]
            if target.id.startswith("__"):
                self._reject(node, "assignment target must be plain local names")
                return
            self._expression(node.value)
            if len(targets) == 1:
                target_name = targets[0].id
                if self._is_container_expression(node.value):
                    self.container_names.add(target_name)
                else:
                    self.container_names.discard(target_name)
                numeric_bound = _numeric_magnitude(
                    node.value, self.numeric_bounds
                )
                if numeric_bound is not None:
                    self.numeric_bounds[target_name] = numeric_bound
                    if numeric_bound > MAX_HOST_COMPUTED_MAGNITUDE:
                        self._reject(
                            node,
                            "computed reference metadata exceeds the numeric limit",
                        )
                else:
                    self.numeric_bounds.pop(target_name, None)
            return
        if isinstance(node, ast.If):
            self._reject(node, "Python if is not allowed; use tensor operations")
            return
        if isinstance(node, ast.Return):
            self.returns += 1
            if node.value is None:
                self._reject(node, "reference return value must not be None")
            else:
                self._expression(node.value)
            return
        self._reject(node, f"statement is not allowed: {type(node).__name__}")

    def _expression(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                self._reject(node, f"dunder name is not allowed: {node.id}")
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (bool, int, float, str, type(None))):
                self._reject(node, "constant type is not allowed")
            elif isinstance(node.value, int) and not isinstance(node.value, bool) and abs(node.value) > MAX_HOST_LITERAL_INT:
                self._reject(node, "integer literal exceeds the reference metadata limit")
            elif isinstance(node.value, str) and len(node.value) > MAX_HOST_STRING_CHARS:
                self._reject(node, "string literal exceeds the reference metadata limit")
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            if len(node.elts) > MAX_HOST_CONTAINER_ITEMS:
                self._reject(node, "reference container literal is too large")
            for element in node.elts:
                self._expression(element)
            return
        if isinstance(node, ast.Subscript):
            self._expression(node.value)
            self._expression(node.slice)
            return
        if isinstance(node, ast.Slice):
            for value in (node.lower, node.upper, node.step):
                if value is not None:
                    self._expression(value)
            return
        if isinstance(node, ast.IfExp):
            self._expression(node.test)
            self._expression(node.body)
            self._expression(node.orelse)
            return
        if isinstance(node, ast.UnaryOp):
            self._expression(node.operand)
            return
        if isinstance(node, ast.BinOp):
            if (
                isinstance(node.op, (ast.Pow, ast.LShift, ast.RShift))
                or self._is_container_expression(node.left)
                or self._is_container_expression(node.right)
            ):
                self._reject(
                    node,
                    "container repetition and unbounded integer operators are not allowed",
                )
            numeric_bound = _numeric_magnitude(node, self.numeric_bounds)
            if (
                numeric_bound is not None
                and numeric_bound > MAX_HOST_COMPUTED_MAGNITUDE
            ):
                self._reject(
                    node, "computed reference metadata exceeds the numeric limit"
                )
            self._expression(node.left)
            self._expression(node.right)
            return
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._expression(value)
            return
        if isinstance(node, ast.Compare):
            self._expression(node.left)
            for comparator in node.comparators:
                self._expression(comparator)
            return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                self._reject(node, f"private attribute is not allowed: {node.attr}")
            elif isinstance(node.value, ast.Name) and node.value.id == "torch":
                if node.attr not in REFERENCE_TORCH_ATTRIBUTES:
                    self._reject(node, f"torch attribute is not allowed: {node.attr}")
            elif node.attr in REFERENCE_TENSOR_ATTRIBUTES | REFERENCE_TENSOR_CALLS:
                self._expression(node.value)
            else:
                self._reject(node, f"tensor attribute is not allowed: {node.attr}")
            return
        if isinstance(node, ast.Call):
            allowed = False
            if isinstance(node.func, ast.Name):
                allowed = node.func.id in REFERENCE_BUILTIN_CALLS
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                    allowed = node.func.attr in REFERENCE_TORCH_CALLS
                else:
                    allowed = node.func.attr in REFERENCE_TENSOR_CALLS
                    if node.func.attr == "to":
                        self._validate_tensor_to(node)
            if not allowed:
                self._reject(node, "call is not in the reference tensor allowlist")
            for argument in node.args:
                if isinstance(argument, ast.Starred):
                    self._reject(argument, "expanded arguments are not allowed")
                else:
                    self._expression(argument)
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg.startswith("__"):
                    self._reject(keyword, "expanded or dunder keyword is not allowed")
                self._expression(keyword.value)
            return
        self._reject(node, f"expression is not allowed: {type(node).__name__}")

    def _is_container_expression(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.container_names
        if isinstance(node, (ast.Tuple, ast.List)):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return True
        if isinstance(node, ast.IfExp):
            return self._is_container_expression(
                node.body
            ) or self._is_container_expression(node.orelse)
        return False


def validate_reference(
    path: Path,
    max_source_chars: int,
    *,
    entrypoint: str,
    expected_arguments: tuple[str, ...],
    scalar_bounds: Mapping[str, float] | None = None,
) -> list[str]:
    """Validate an independent PyTorch reference before importing it."""

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        return [f"could not read reference: {error}"]
    if len(source) > max_source_chars:
        return [f"reference exceeds {max_source_chars} characters"]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [f"syntax error: {error.msg} at line {error.lineno}"]

    errors: list[str] = []
    imports = 0
    functions: list[ast.FunctionDef] = []
    for statement in tree.body:
        if _is_docstring(statement):
            continue
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "torch" and alias.asname is None:
                    imports += 1
                else:
                    rendered = (
                        alias.name
                        if alias.asname is None
                        else f"{alias.name} as {alias.asname}"
                    )
                    errors.append(f"reference import is not allowed: {rendered}")
        elif isinstance(statement, ast.ImportFrom):
            errors.append(f"reference from-import is not allowed: {statement.module}")
        elif isinstance(statement, ast.FunctionDef):
            functions.append(statement)
        else:
            errors.append(
                f"reference top-level statement is not allowed: {type(statement).__name__}"
            )
    if imports > 1:
        errors.append("reference may contain at most one 'import torch'")
    if len(functions) != 1 or functions[0].name != entrypoint:
        errors.append(
            f"reference must define exactly one function named {entrypoint}"
        )
        return sorted(set(errors))

    function = functions[0]
    if function.decorator_list:
        errors.append("reference entrypoint must not use decorators")
    _validate_arguments(
        function,
        launch=True,
        errors=errors,
        expected_launch_args=expected_arguments,
    )
    body_policy = ReferenceBodyPolicy(
        function.name,
        scalar_bounds=scalar_bounds,
    )
    body_policy.validate(function.body)
    errors.extend(body_policy.errors)
    return sorted(set(errors))

