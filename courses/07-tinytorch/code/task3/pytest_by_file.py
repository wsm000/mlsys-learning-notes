# -*- coding: utf-8 -*-
"""逐文件运行 tests/06_autograd 并生成干净的证据日志（供打卡截图使用）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPO = ROOT / "_tmp_tt_src" / "tinytorch"
OUT = ROOT / "courses" / "07-tinytorch" / "evidence" / "task3" / "task3_pytest_by_file.log"

FILES = [
    "tests/06_autograd/test_06_autograd_progressive.py",
    "tests/06_autograd/test_autograd_core.py",
    "tests/06_autograd/test_autograd_gradient_flow.py",
    "tests/06_autograd/test_batched_matmul_backward.py",
    "tests/06_autograd/test_gradient_correctness.py",
]
BANNER = ("Modules not yet exported", "  ⚠", "[tinytorch]")

lines = [
    "从Tensor到Transformer——构建 GPT 模型 | Task 3 自动微分引擎",
    "GitHub ID: wsm000  |  本地 Windows PowerShell 5.1  |  Python 3.10.2 / NumPy 2.2.6 / pytest 9.1.1",
    "$ cd " + str(REPO),
]


def run(args):
    p = subprocess.run([sys.executable, "-m", "pytest"] + args,
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    body = []
    for ln in p.stdout.splitlines():
        if any(b in ln for b in BANNER):
            continue
        if ln.strip() == "" and body and body[-1].strip() == "":
            continue
        body.append(ln.rstrip())
    while body and not body[-1].strip():
        body.pop()
    return body


for f in FILES:
    lines.append("")
    lines.append("PS> python -m pytest " + f + " -q --tb=short")
    lines += ["  " + l for l in run([f, "-q", "--tb=short"])]

lines.append("")
lines.append("PS> python -m pytest tests/06_autograd -q --tb=short          # Module 06 汇总")
lines += ["  " + l for l in run(["tests/06_autograd", "-q", "--tb=short"])]

lines.append("")
lines.append("PS> python -m pytest tests/01_tensor tests/02_activations tests/03_layers tests/04_losses tests/05_dataloader tests/06_autograd -q --tb=short          # 01-06 全量回归")
lines += ["  " + l for l in run([
    "tests/01_tensor", "tests/02_activations", "tests/03_layers",
    "tests/04_losses", "tests/05_dataloader", "tests/06_autograd", "-q", "--tb=short"])]

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written", len(lines), "lines ->", OUT)
