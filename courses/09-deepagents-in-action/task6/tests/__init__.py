"""Test package for the Task6 labs (course 09, deepagents-in-action).

Importing this package first keeps bytecode out of the workspace and puts the
Task6 lab runners (run_memory_lab / run_approval_lab / run_store_lab /
run_risk_lab) on sys.path. The runners are frozen evidence-producing code;
these tests only import and re-run them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

TASK6_DIR = Path(__file__).resolve().parent.parent
if str(TASK6_DIR) not in sys.path:
    sys.path.insert(0, str(TASK6_DIR))
