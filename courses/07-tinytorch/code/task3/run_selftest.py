# -*- coding: utf-8 -*-
"""TinyTorch 模块自测运行器：加载模块源码并执行其 test_module()（Task 3 版）。

用法:  python run_selftest.py 06        # 测试 Module 06 (Autograd)

说明:  Module 06 autograd 为纯 NumPy 实现，无需 GPU。
       在 tinytorch 仓库根目录执行，或从任意位置执行（自动定位仓库）。
"""
import importlib.util
import sys
from pathlib import Path

MODULES = {
    "06": "src/06_autograd/06_autograd.py",
}


def find_repo() -> Path:
    """按优先级查找 tinytorch 仓库根目录。"""
    candidates = [
        Path.cwd(),
        # 回退：本工作区解包的 tinytorch 源码仓（hello-gpu/_tmp_tt_src/tinytorch）
        Path(__file__).resolve().parents[4] / "_tmp_tt_src" / "tinytorch",
        Path(__file__).resolve().parent / "tinytorch",
    ]
    for c in candidates:
        if (c / "src/06_autograd/06_autograd.py").exists():
            return c
    raise SystemExit("找不到 tinytorch 仓库（应包含 src/06_autograd/06_autograd.py）")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in MODULES:
        print("usage: python run_selftest.py [06]")
        return 2
    repo = find_repo()
    path = repo / MODULES[sys.argv[1]]
    spec = importlib.util.spec_from_file_location(f"module_{sys.argv[1]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # 仅加载定义（__main__ 守卫不会触发）
    mod.test_module()              # 运行模块自带集成测试
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

