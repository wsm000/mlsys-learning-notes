"""intake：从用户给的原始 kernel 自动建立可优化的任务工作区。

分工（LLM 负责理解，代码负责相信）：
- LLM 生成 reference.py（PyTorch 正确性裁判）+ tensor 声明；
- 代码机械构造 v2 task.json，并用 evaluator 自检 baseline；
- 自检失败就把精确报错喂回 LLM 重试（错误驱动，最多 max_attempts 次）。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from . import llm


# ---------------------------------------------------------------------------
# 静态识别：语言 + 入口函数（紧凑 AST，不依赖正则）
# ---------------------------------------------------------------------------


def detect_language(source: str) -> str:
    if "import triton" in source:
        return "triton"
    if "hipLaunchKernelGGL" in source or "__hip" in source:
        return "hip"
    if "__global__" in source or "cuda" in source.lower():
        return "cuda"
    return "unknown"


def _is_jit_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "triton" and node.attr == "jit"
    return isinstance(node, ast.Name) and node.id == "jit"


def _has_kernel_dispatch(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Subscript):
            return True  # kernel[grid](...)
    return False


def parse_entrypoint(source: str) -> tuple[str, list[str]] | None:
    """找 host 入口函数：优先叫 launch，否则第一个调用 kernel dispatch 的非 jit 函数。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    host_funcs: list[ast.FunctionDef] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(_is_jit_decorator(d) for d in node.decorator_list):
            continue
        if not node.name.startswith("_"):
            host_funcs.append(node)
    for func in host_funcs:
        if func.name == "launch":
            return func.name, _params(func)
    for func in host_funcs:
        if _has_kernel_dispatch(func):
            return func.name, _params(func)
    return None


def _params(func: ast.FunctionDef) -> list[str]:
    return [a.arg for a in func.args.posonlyargs + func.args.args]


# ---------------------------------------------------------------------------
# LLM 生成 reference + tensor 声明
# ---------------------------------------------------------------------------


_GEN_PROMPT = """\
你要为一个 GPU kernel 建立优化任务。请生成下面的内容，放进一个 JSON。

1. reference_source：一个**极简**的纯 PyTorch 函数 `reference(...)`，作为正确性裁判。
   硬性约束（evaluator 的 AST 白名单会校验，违反就失败）：
   - **第一行写 `import torch`**（reference 允许、且仅允许这一个 import）；不写 import 直接用 torch 会 NameError。
   - 函数名必须是 reference，参数名与顺序和 kernel 入口**完全一致**（含 output、标量等全部参数）。
   - 函数体尽量就是**一个 return 语句**，返回期望的输出张量；要中间变量就用「单个名字 = 表达式」的赋值。
   - 用 torch.softmax / torch.exp / torch.sum / torch.where 等白名单算子，或张量方法（.float() .to() .reshape() .softmax(dim=-1)）；返回 dtype 与 kernel 输出一致。
   - **禁止**：除 `import torch` 外的其它 import、裸表达式语句、元组解包赋值（a, b = ...）、if/for/while/try/with、class、lambda、带注解的赋值。
   - 范例（vector add，只用张量方法、可省 import）：
     def reference(x, y, output, n_elements):
         return (x.float() + y.float()).to(x.dtype)
   - 范例（逐行 softmax，用 torch.softmax，返回 (rows, cols) 不要 flatten）：
     import torch
     def reference(x, out, n_cols):
         rows = x.numel() // n_cols
         return torch.softmax(x.float().reshape(rows, n_cols), dim=-1).to(x.dtype)

2. tensors：每个入口参数的声明，列表项形如
   {{"name": "x", "kind": "input", "dtype": "float32", "shape": "input"}}。
   - kind ∈ input / output / scalar。输入/输出张量 shape 统一写 "input"，表示形状 (rows, cols) 的**二维连续**张量（numel = rows*cols）；
     reference 返回的输出形状要与之一致（即 (rows, cols)，**不要 flatten 成一维**）。
   - **标量（kind=scalar）的 dtype 只能是 int64 或 int32**（绝不能用 float16），并给 "value"（整数，通常 = rows*cols 或 cols）。
   - 输入/输出张量的 dtype 按 kernel 实际数据类型（本例为 {dtype}）。

3. costModel（尽量给）：{{"flops": 正整数, "bytes": 正整数, "label": "口径说明"}}。
   估算本 shape 下算法的浮点运算量与数据搬运字节数（供 Roofline 判断 memory/compute-bound）。
   实在不确定就**省略该字段**，不要填 0 或乱猜。

kernel 入口：{entrypoint}({params})
算子描述：{description}
目标 shape：rows={rows}, cols={cols}（共 {numel} 个元素）
数据 dtype：{dtype}

kernel 源码：
{kernel}
{feedback}
只输出 JSON：{{"reference_source": "...", "tensors": [ ... ], "costModel": {{...}}}}，不要 markdown 围栏；没有 costModel 就省略该字段。
"""


def _generate(kernel: str, description: str, entrypoint: str, params: list[str],
              rows: int, cols: int, dtype: str, feedback: str = "") -> dict[str, Any] | None:
    prompt = _GEN_PROMPT.format(
        entrypoint=entrypoint, params=", ".join(params), description=description,
        rows=rows, cols=cols, numel=rows * cols, dtype=dtype, kernel=kernel,
        feedback=(f"\n上一次自检失败的报错如下，请据此修正：\n{feedback}\n" if feedback else ""),
    )
    message = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
    return _parse_json(message.content or "")


def _parse_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("reference_source"), str):
            return data
    return None


# ---------------------------------------------------------------------------
# 构造 task.json + 自检
# ---------------------------------------------------------------------------


def build_task_json(description: str, language: str, entrypoint: str,
                    tensors: list[dict[str, Any]], launch_arguments: list[str],
                    rows: int, cols: int, cost_model: dict[str, Any] | None = None) -> dict[str, Any]:
    task: dict[str, Any] = {
        "schemaVersion": 2,
        "name": "optimized-operator",
        "description": description,
        "language": language,
        "entrypoint": entrypoint,
        "shape": {"rows": rows, "cols": cols},
        "input": {"distribution": "normal", "mean": 0, "standardDeviation": 1},
        "tensors": tensors,
        "launchArguments": launch_arguments,
        "reference": {"filename": "reference.py", "entrypoint": "reference", "selfCheckPassed": True},
        "correctness": {"seeds": [17, 29], "atol": 0.001, "rtol": 0.001},
        "benchmark": {"warmup": 10, "samples": 25, "innerRepeats": 5, "timeoutSeconds": 180},
        "optimization": {"maxRounds": 4, "patience": 3, "minImprovementFraction": 0.01, "maxSourceChars": 50000},
    }
    # costModel 只在严格合法时才写入（flops/bytes 为正整数），供 Roofline 定方向。
    if (
        isinstance(cost_model, dict)
        and isinstance(cost_model.get("flops"), int)
        and isinstance(cost_model.get("bytes"), int)
        and not isinstance(cost_model.get("flops"), bool)
        and not isinstance(cost_model.get("bytes"), bool)
        and cost_model["flops"] > 0
        and cost_model["bytes"] > 0
    ):
        label = str(cost_model.get("label", "")).strip()[:200] or "estimated"
        task["costModel"] = {"flops": cost_model["flops"], "bytes": cost_model["bytes"], "label": label}
    return task


def setup_task(workspace: Any, kernel_source: str, description: str,
               rows: int, cols: int, dtype: str = "float32", max_attempts: int = 3) -> str:
    """从原始 kernel 建立任务工作区：生成 reference + task.json，自检 baseline。"""
    entry = parse_entrypoint(kernel_source)
    if entry is None:
        return "✗ 找不到入口函数。需要一个调用 kernel 的 host 函数（通常叫 launch）。请检查代码。"
    entrypoint, params = entry
    language = detect_language(kernel_source)

    from .tools import _run_eval  # 延迟导入避免循环

    feedback = ""
    for _ in range(max_attempts):
        gen = _generate(kernel_source, description, entrypoint, params, rows, cols, dtype, feedback)
        if gen is None:
            return "✗ 无法从模型生成 reference/contract。"
        tensors = gen.get("tensors") or []
        workspace.reference_path.write_text(gen["reference_source"], encoding="utf-8")
        task = build_task_json(
            description, language, entrypoint, tensors, params, rows, cols,
            cost_model=gen.get("costModel"),
        )
        workspace.task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        workspace.best_path.write_text(
            kernel_source if kernel_source.endswith("\n") else kernel_source + "\n",
            encoding="utf-8",
        )
        evaluation = _run_eval(workspace, kernel_source, incumbent=None, correctness_only=True)
        correctness = evaluation.get("correctness") or {}
        if evaluation.get("status") == "ok" and correctness.get("passed") is True:
            return (
                f"✓ 任务已建立并通过自检：入口 {entrypoint}({', '.join(params)})，"
                f"shape {rows}×{cols}，language={language}。可以开始优化闭环了。"
            )
        feedback = (evaluation.get("details", "") + "\n" + evaluation.get("capturedOutput", ""))[-1200:]

    return f"✗ 自检 {max_attempts} 次仍未通过，最后的报错：\n{feedback[:800]}"
