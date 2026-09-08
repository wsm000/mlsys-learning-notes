import json
import tempfile
import unittest
from pathlib import Path

from evaluate import _as_text
from source_policy import validate_candidate
from task_spec import load_task


PART_ROOT = Path(__file__).resolve().parent.parent
TASK_PATH = PART_ROOT / "chapter13" / "task.json"
BASELINE_PATH = Path(__file__).with_name("baseline.py")
FIXTURES = Path(__file__).with_name("fixtures")


class EvaluatorContractTest(unittest.TestCase):
    def test_loads_frozen_task(self) -> None:
        task = load_task(TASK_PATH)
        self.assertEqual((task.shape.rows, task.shape.cols), (4096, 2048))
        self.assertEqual(task.optimization.min_improvement_fraction, 0.01)

    def test_baseline_passes_source_policy(self) -> None:
        self.assertEqual(validate_candidate(BASELINE_PATH, 50_000), [])

    def test_fixtures_pass_source_policy(self) -> None:
        for name in ("compile_error.py", "wrong_answer.py", "fused_reference.py"):
            with self.subTest(name=name):
                self.assertEqual(validate_candidate(FIXTURES / name, 50_000), [])

    def test_rejects_forbidden_import(self) -> None:
        source = "import subprocess\n\ndef launch(*args):\n    pass\n"
        with tempfile.TemporaryDirectory() as temporary_dir:
            candidate = Path(temporary_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            errors = validate_candidate(candidate, 50_000)
        self.assertTrue(any("import is not allowed: subprocess" in error for error in errors))

    def test_rejects_from_import_alias(self) -> None:
        source = """\
from torch import softmax as f
import triton
import triton.language as tl

@triton.jit
def _kernel(output):
    tl.store(output, 0.0)

def launch(x, output, exp_values, row_max, row_sum, scale, rows, cols):
    f(x)
    _kernel[(1,)](output)
"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            candidate = Path(temporary_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            errors = validate_candidate(candidate, 50_000)
        self.assertTrue(any("from-import is not allowed: torch" in error for error in errors))

    def test_rejects_builtins_import_escape(self) -> None:
        source = """\
import triton
import triton.language as tl

@triton.jit
def _kernel(output):
    tl.store(output, 0.0)

def launch(x, output, exp_values, row_max, row_sum, scale, rows, cols):
    module = __builtins__["__import__"]("os")
    _kernel[(1,)](output)
"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            candidate = Path(temporary_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            errors = validate_candidate(candidate, 50_000)
        self.assertTrue(any("host allowlist" in error for error in errors))

    def test_rejects_host_global_state(self) -> None:
        source = """\
import triton
import triton.language as tl

calls = 0

@triton.jit
def _kernel(output):
    tl.store(output, 0.0)

def launch(x, output, exp_values, row_max, row_sum, scale, rows, cols):
    global calls
    calls += 1
    _kernel[(1,)](output)
"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            candidate = Path(temporary_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            errors = validate_candidate(candidate, 50_000)
        self.assertTrue(any("top-level statement is not allowed" in error for error in errors))

    def test_timeout_output_accepts_bytes(self) -> None:
        self.assertEqual(_as_text(b"before timeout\xff"), "before timeout\ufffd")

    def test_rejects_parent_directory_candidate_filename(self) -> None:
        payload = json.loads(TASK_PATH.read_text(encoding="utf-8"))
        payload["candidateFilename"] = ".."
        with tempfile.TemporaryDirectory() as temporary_dir:
            task_path = Path(temporary_dir) / "task.json"
            task_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "simple filename|must not"):
                load_task(task_path)


if __name__ == "__main__":
    unittest.main()

