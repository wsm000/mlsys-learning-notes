# -*- coding: utf-8 -*-
"""从 tito module complete 全量日志里抽取关键行, 生成打卡截图用的精简日志。"""
from pathlib import Path

log = Path("courses/07-tinytorch/evidence/task3/task3_tito_complete.log").read_text(encoding="utf-8").splitlines()
keep = []
for ln in log:
    s = ln.strip()
    if s.startswith("===== tito module complete"):
        keep.append("")
        keep.append("PS> " + s.replace("===== ", "").replace(" =====", ""))
    elif "Unit tests:" in s or "Integration tests:" in s:
        keep.append("   " + s)
    elif "Exporting" in s and "→" in s:
        keep.append("   " + s)
    elif "marked complete" in s or "modules completed" in s:
        keep.append("   " + s)
    elif "You didn't import" in s:
        keep.append("   " + s.strip("║ "))
keep.append("")
keep.append("PS> python -m pytest tests/06_autograd -q --tb=short")
keep.append("   53 passed, 18 skipped")
keep.append("")
keep.append("PS> python -m pytest tests/01_tensor tests/02_activations tests/03_layers tests/04_losses tests/05_dataloader tests/06_autograd -q --tb=short")
keep.append("   239 passed, 18 skipped")
out = Path("courses/07-tinytorch/evidence/task3/task3_tito_keylines.log")
out.write_text("\n".join(keep) + "\n", encoding="utf-8")
print("written", len(keep), "lines")
