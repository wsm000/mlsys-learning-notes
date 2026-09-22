# -*- coding: utf-8 -*-
"""把 src/<NN>/<NN>.py (jupytext percent 格式) 转成 modules/<NN>/<name>.ipynb,
供官方 tito module complete 的导出步骤使用。"""
import sys
from pathlib import Path

import jupytext
import nbformat

REPO = Path(sys.argv[1])
MAP = [
    ("01_tensor", "tensor"),
    ("02_activations", "activations"),
    ("03_layers", "layers"),
    ("04_losses", "losses"),
    ("05_dataloader", "dataloader"),
    ("06_autograd", "autograd"),
]
for dirname, name in MAP:
    src = REPO / "src" / dirname / (dirname + ".py")
    dst = REPO / "modules" / dirname / (name + ".ipynb")
    dst.parent.mkdir(parents=True, exist_ok=True)
    nb = jupytext.read(src)
    nbformat.write(nb, dst)
    print("converted", dst.relative_to(REPO), nbformat.write.__module__ and len(nb.cells), "cells")
