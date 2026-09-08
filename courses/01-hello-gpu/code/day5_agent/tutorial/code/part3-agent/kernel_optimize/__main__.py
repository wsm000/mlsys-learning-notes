"""python -m kernel_optimize 入口。

交互：``uv run python -m kernel_optimize``
非交互：``uv run python -m kernel_optimize --batch chapter15/fixtures/vector_add``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .agent import run_agent
from .prompts import INTAKE_GOAL

_PART_ROOT = Path(__file__).resolve().parent.parent


def _auto_workspace() -> Path:
    """自动生成一个带时间戳的工作区（在 chapter16/logs 或 chapter15/logs 下）。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _PART_ROOT / "chapter15" / "logs" / "tasks" / f"task-{timestamp}"


def _banner(batch: bool = False) -> None:
    line = "═" * 56
    mode = "非交互 batch" if batch else "对话式"
    print()
    print(f"\033[36m╭{line}╮\033[0m")
    print(f"\033[36m│\033[0m  \033[1mKernel Optimize Agent · 算子优化闭环（{mode}）\033[0m")
    print(f"\033[36m╰{line}╯\033[0m")
    print()


def _on_step(step: int, action: str, observation: str | None) -> None:
    if action == "done":
        print(f"\n\033[32m[step {step}] agent 完成\033[0m")
    elif action == "nudge":
        print(f"\033[90m[step {step}] （提示 agent 改用工具，不要只用文本回复）\033[0m")
    elif action.startswith("tool:"):
        print(f"\033[36m[step {step}] 调工具 {action.split(':', 1)[1]} …\033[0m")
    elif action.startswith("result:") and observation:
        first = observation.strip().splitlines()[0] if observation.strip() else ""
        print(f"\033[90m[step {step}]   → {first[:160]}\033[0m")


def _drain_stdin() -> None:
    """退出前排空 stdin 里残留的粘贴行，避免它们漏给 shell 被当命令执行。"""
    import select

    try:
        while select.select([sys.stdin], [], [], 0.05)[0]:
            if sys.stdin.readline() == "":
                break
    except (OSError, ValueError):
        pass


def _resolve_fixture(path: str | Path) -> Path:
    fixture = Path(path)
    if not fixture.is_absolute():
        candidate = (_PART_ROOT / fixture).resolve()
        if candidate.exists():
            fixture = candidate
        else:
            fixture = fixture.resolve()
    if not fixture.is_dir():
        raise FileNotFoundError(f"fixtures 目录不存在：{fixture}")
    for name in ("task.json", "reference.py", "baseline.py"):
        if not (fixture / name).is_file():
            raise FileNotFoundError(f"fixtures 缺少 {name}：{fixture}")
    return fixture


def _seed_workspace_from_fixture(fixture: Path, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture / "task.json", workspace / "task.json")
    shutil.copy2(fixture / "reference.py", workspace / "reference.py")
    shutil.copy2(fixture / "baseline.py", workspace / "best.py")
    # 保留一份原始 baseline，便于报告对比
    shutil.copy2(fixture / "baseline.py", workspace / "baseline.py")


def _print_deterministic_summary(workspace: Path) -> None:
    """无论 LLM 报告说什么，都打印一份基于轨迹的真账。"""
    traj = workspace / "trajectory.jsonl"
    print("\n\033[36m═══ 确定性汇总（trajectory）═══\033[0m")
    if not traj.is_file():
        print("（尚无 trajectory.jsonl）")
        return
    rows: list[dict] = []
    for line in traj.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        print("（轨迹为空）")
        return
    accepted = [r for r in rows if r.get("accepted")]
    print(f"轮次：{len(rows)}；接受：{len(accepted)}")
    for i, row in enumerate(rows, 1):
        flag = "✓" if row.get("accepted") else "✗"
        change = row.get("change") or ""
        lat = row.get("latencyMs")
        imp = row.get("improvementFraction")
        reason = row.get("reason") or row.get("status") or ""
        extra = f" latency={lat}ms" if lat is not None else ""
        if imp is not None:
            extra += f" improvement={imp:.4f}"
        print(f"  [{i}] {flag} {change[:60]}{extra} · {reason}")
    print(f"best.py：{workspace / 'best.py'}")
    print(f"轨迹：{traj}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kernel Optimize Agent")
    parser.add_argument(
        "--batch",
        metavar="FIXTURE_DIR",
        help="非交互模式：从 fixtures 目录加载 task/reference/baseline 并跑优化闭环",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="最大工具步数")
    parser.add_argument(
        "--workspace",
        default=None,
        help="指定工作区目录（默认自动生成 chapter15/logs/tasks/task-时间戳）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from .llm import _model_config

    try:
        _model_config()
    except RuntimeError as error:
        print(f"\n\033[31m✗ {error}\033[0m\n")
        return 1

    batch = bool(args.batch)
    workspace = Path(args.workspace) if args.workspace else _auto_workspace()
    max_steps = args.max_steps if args.max_steps is not None else (25 if batch else 40)

    _banner(batch=batch)

    if batch:
        try:
            fixture = _resolve_fixture(args.batch)
        except FileNotFoundError as error:
            print(f"\033[31m✗ {error}\033[0m")
            return 1
        _seed_workspace_from_fixture(fixture, workspace)
        print(f"fixtures：{fixture}")
        print(f"工作区：{workspace}\n")
        goal = None  # build_goal(batch=True)
        try:
            report = run_agent(
                workspace,
                max_steps=max_steps,
                on_step=_on_step,
                batch=True,
                goal=goal,
            )
        except Exception as error:  # noqa: BLE001
            print(f"\033[31m✗ agent 运行失败：{error}\033[0m")
            _print_deterministic_summary(workspace)
            return 1
    else:
        print("粘贴你要优化的 kernel（或描述这个算子），我会问清细节并自动跑完整优化闭环。")
        print(f"工作区：{workspace}\n")
        try:
            report = run_agent(
                workspace,
                max_steps=max_steps,
                on_step=_on_step,
                goal=INTAKE_GOAL,
                batch=False,
            )
        finally:
            _drain_stdin()

    print("\n\033[36m═══ 最终报告 ═══\033[0m\n")
    print(report)
    _print_deterministic_summary(workspace)
    print(f"\n\033[90m轨迹与产物在工作区：{workspace}\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
