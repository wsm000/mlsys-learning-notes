"""intake 测试：从原始 kernel 自动建任务（解析 + LLM 生成 + evaluator 自检）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kernel_optimize import intake
from kernel_optimize.intake import (
    _parse_json,
    build_task_json,
    detect_language,
    parse_entrypoint,
    setup_task,
)
from kernel_optimize.tools import Workspace

TRITON_VADD = '''
import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pass

def launch(x, out, n):
    _k[(1,)](x, out, n, BLOCK=64)
'''


class ParseTest(unittest.TestCase):
    def test_detect_language(self) -> None:
        self.assertEqual(detect_language("import triton\n"), "triton")
        self.assertEqual(detect_language("__global__ void k(){}"), "cuda")
        self.assertEqual(detect_language("hipLaunchKernelGGL(k,g,b,0,0)"), "hip")

    def test_parse_entrypoint_launch(self) -> None:
        entry = parse_entrypoint(TRITON_VADD)
        self.assertEqual(entry, ("launch", ["x", "out", "n"]))

    def test_parse_entrypoint_none_for_no_host(self) -> None:
        self.assertIsNone(parse_entrypoint("x = 1\n"))


class ParseJsonTest(unittest.TestCase):
    def test_plain_and_fenced(self) -> None:
        data = _parse_json('{"reference_source": "def reference(): pass", "tensors": []}')
        assert data is not None
        self.assertIn("reference_source", data)
        fenced = _parse_json('```json\n{"reference_source": "x", "tensors": []}\n```')
        assert fenced is not None
        self.assertEqual(fenced["reference_source"], "x")


class BuildTaskTest(unittest.TestCase):
    def test_structure(self) -> None:
        task = build_task_json(
            "vector add", "triton", "launch",
            [{"name": "x", "kind": "input", "dtype": "float32", "shape": "input"}],
            ["x", "out", "n"], 4096, 2048,
        )
        self.assertEqual(task["schemaVersion"], 2)
        self.assertEqual(task["shape"], {"rows": 4096, "cols": 2048})
        self.assertEqual(task["launchArguments"], ["x", "out", "n"])
        self.assertEqual(task["reference"]["filename"], "reference.py")
        self.assertGreaterEqual(task["benchmark"]["samples"], 5)


class SetupTaskTest(unittest.TestCase):
    def _mock_gen(self):
        payload = json.dumps({
            "reference_source": "def reference(x, out, n):\n    return x\n",
            "tensors": [
                {"name": "x", "kind": "input", "dtype": "float32", "shape": "input"},
                {"name": "out", "kind": "output", "dtype": "float32", "shape": "input"},
                {"name": "n", "kind": "scalar", "value": 8388608},
            ],
        })
        return mock.patch.object(intake.llm, "chat", return_value=_FakeMsg(payload))

    def test_success_writes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            ok_eval = {"status": "ok", "correctness": {"passed": True, "maxAbsError": 0.0}}
            with self._mock_gen(), mock.patch("kernel_optimize.tools._run_eval", return_value=ok_eval):
                result = setup_task(ws, TRITON_VADD, "vector add", 4096, 2048)
            self.assertIn("✓", result)
            self.assertTrue(ws.task_path.exists())
            self.assertTrue(ws.reference_path.exists())
            self.assertTrue(ws.best_path.exists())
            task = json.loads(ws.task_path.read_text())
            self.assertEqual(task["launchArguments"], ["x", "out", "n"])

    def test_retries_on_self_check_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            evals = [
                {"status": "wrong_answer", "details": "输出不对", "correctness": {"passed": False}},
                {"status": "ok", "correctness": {"passed": True}},
            ]
            calls = {"n": 0}

            def fake_eval(*a, **k):
                result = evals[min(calls["n"], len(evals) - 1)]
                calls["n"] += 1
                return result

            with self._mock_gen() as gen_mock, mock.patch("kernel_optimize.tools._run_eval", side_effect=fake_eval):
                result = setup_task(ws, TRITON_VADD, "vector add", 4096, 2048)
            self.assertIn("✓", result)
            self.assertEqual(calls["n"], 2)          # 第一次失败、第二次成功
            self.assertEqual(gen_mock.call_count, 2)  # 据此重新生成了一次

    def test_no_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(Path(tmp))
            result = setup_task(ws, "x = 1\n", "noop", 4, 8)
            self.assertIn("找不到入口函数", result)


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


if __name__ == "__main__":
    unittest.main()
