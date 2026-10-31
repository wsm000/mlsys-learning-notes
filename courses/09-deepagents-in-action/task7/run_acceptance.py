#!/usr/bin/env python3
"""Task7 项目验收运行器：固定输入 + 明确预期 + 证据。

做法（对齐第 13 章验收思路）：
  1. 固定输入：每个 case 绑定 fixture 的 task.json / reference.py / 候选字节，
     运行前后都算 sha256 写进 manifest——输入身份可复核。
  2. 明确预期：每个 case 的 expect 块在跑之前就写死在 cases/*.json。
  3. 证据：一律调用项目自带的权威裁判 chapter14/evaluate.py（稳定子进程边界），
     原始 JSON、完整终端输出、墙钟耗时全部落盘；裁判说什么就记什么，
     不替裁判改数、不挑数字。

用法：
  python3 run_acceptance.py --part-root <part3-agent> --out evidence/
  python3 run_acceptance.py --part-root <part3-agent> --out evidence/ --sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parent
CANDIDATES = SUITE_ROOT / "candidates"
CASES = SUITE_ROOT / "cases"
NL = chr(10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def host_environment() -> dict:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "runnerStartedAtUtc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cudaAvailable"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpuMemoryMiB"] = round(props.total_memory / (1 << 20))
    except Exception as error:  # noqa: BLE001
        info["torchImportError"] = repr(error)
    try:
        import triton  # noqa: PLC0415

        info["triton"] = triton.__version__
    except Exception as error:  # noqa: BLE001
        info["tritonImportError"] = repr(error)
    return info


def run_evaluator(part_root: Path, task: Path, candidate: Path, reference: Path,
                  incumbent: Path | None, timeout_seconds: int) -> dict:
    """调用项目权威裁判 chapter14/evaluate.py，返回原始 JSON + 终端输出 + 墙钟。"""
    evaluate_py = part_root / "chapter14" / "evaluate.py"
    if not evaluate_py.is_file():
        raise FileNotFoundError(f"evaluator not found: {evaluate_py}")
    command = [
        sys.executable, str(evaluate_py),
        "--task", str(task),
        "--candidate", str(candidate),
        "--reference", str(reference),
        "--pretty",
    ]
    if incumbent is not None:
        command.extend(["--incumbent", str(incumbent)])
    started = time.monotonic()
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_seconds, check=False,
    )
    wall_seconds = time.monotonic() - started
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        result = {"status": "runner_error", "details": f"invalid JSON from evaluator: {error}"}
    return {
        "command": command,
        "returncode": completed.returncode,
        "wallSeconds": round(wall_seconds, 3),
        "result": result,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def check_expectation(case: dict, result: dict, candidate_sha: str, incumbent_sha: str | None) -> dict:
    """把裁判输出与 case 的预期逐条对账，产出 verdict（预期是跑之前写死的）。"""
    expect = case["expect"]
    checks = []

    def add(name: str, ok: bool, actual):
        checks.append({"check": name, "passed": bool(ok), "actual": actual})

    add("status", result.get("status") == expect["status"], result.get("status"))
    correctness = result.get("correctness") or {}
    if "correctness_passed" in expect:
        add("correctness.passed", correctness.get("passed") == expect["correctness_passed"],
            correctness.get("passed"))
    if expect.get("has_benchmark"):
        add("benchmark present", bool(result.get("benchmark")), bool(result.get("benchmark")))
    if expect.get("has_comparison"):
        add("comparison present", bool(result.get("comparison")), bool(result.get("comparison")))
    if expect.get("candidate_differs_from_incumbent"):
        add("candidate != incumbent (sha256)", candidate_sha != incumbent_sha,
            {"candidate": candidate_sha[:12], "incumbent": (incumbent_sha or "")[:12]})
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run_case(part_root: Path, case_path: Path, out_dir: Path, manifest: dict) -> dict:
    case = load_json(case_path)
    candidate_path = CANDIDATES / case["candidate"]
    incumbent_path = CANDIDATES / case["incumbent"] if case.get("incumbent") else None
    task_path = CANDIDATES / "task.json"
    reference_path = CANDIDATES / "reference.py"

    candidate_sha = sha256_file(candidate_path)
    incumbent_sha = sha256_file(incumbent_path) if incumbent_path else None
    run = run_evaluator(
        part_root, task_path, candidate_path, reference_path, incumbent_path,
        timeout_seconds=manifest["evaluatorTimeoutSeconds"],
    )
    verdict = check_expectation(case, run["result"], candidate_sha, incumbent_sha)

    case_id = case["id"]
    result_path = out_dir / (case_id + ".result.json")
    terminal_path = out_dir / (case_id + ".terminal.txt")
    write_text(result_path, json.dumps(run["result"], ensure_ascii=False, indent=2) + NL)
    terminal_text = ("$ " + " ".join(run["command"]) + NL + NL
                    + "# returncode=" + str(run["returncode"])
                    + " wall=" + str(run["wallSeconds"]) + "s" + NL
                    + "----- stdout -----" + NL + run["stdout"]
                    + "----- stderr -----" + NL + run["stderr"])
    write_text(terminal_path, terminal_text)
    record = {
        "case": case,
        "candidate": {"file": case["candidate"], "sha256": candidate_sha},
        "incumbent": ({"file": case["incumbent"], "sha256": incumbent_sha}
                       if incumbent_path else None),
        "evaluatorReturncode": run["returncode"],
        "wallSeconds": run["wallSeconds"],
        "result": run["result"],
        "verdict": verdict,
        "evidenceFiles": [result_path.name, terminal_path.name],
    }
    manifest["cases"].append(record)
    status = "PASS" if verdict["passed"] else "FAIL"
    print(f"[{status}] {case_id}: status={run['result'].get('status')} "
          f"correctness={(run['result'].get('correctness') or {}).get('passed')} "
          f"wall={run['wallSeconds']}s")
    for check in verdict["checks"]:
        if not check["passed"]:
            print(f"       ! {check['check']}: expected per case json, actual={check['actual']}")
    return record


def run_sweep(part_root: Path, out_dir: Path, manifest: dict) -> dict:
    """Advanced：对「baseline 慢」这一失败事实做单变量修改（block_size），
    同一案例（同一 task.json 契约、同一 incumbent）重跑，记录实测耗时。"""
    block_sizes = [256, 512, 1024, 2048, 4096]
    baseline_source = (CANDIDATES / "baseline.py").read_text(encoding="utf-8")
    original_line = "    block_size = 256  # 故意偏小，给优化留空间（可试更大 block / 向量化）"
    sweep_dir = out_dir / "sweep_candidates"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    task_path = CANDIDATES / "task.json"
    reference_path = CANDIDATES / "reference.py"
    baseline_path = CANDIDATES / "baseline.py"
    points = []
    for block_size in block_sizes:
        source = baseline_source.replace(original_line, "    block_size = " + str(block_size))
        if block_size != 256 and source == baseline_source:
            raise RuntimeError(f"sweep substitution failed for block_size={block_size}")
        candidate_path = sweep_dir / ("vadd_block" + str(block_size) + ".py")
        write_text(candidate_path, source)
        run = run_evaluator(
            part_root, task_path, candidate_path, reference_path, baseline_path,
            timeout_seconds=manifest["evaluatorTimeoutSeconds"],
        )
        result = run["result"]
        comparison = result.get("comparison") or {}
        benchmark = result.get("benchmark") or {}
        point = {
            "blockSize": block_size,
            "candidateSha256": sha256_file(candidate_path),
            "singleVariableDiff": "block_size: 256 -> " + str(block_size),
            "status": result.get("status"),
            "correctnessPassed": (result.get("correctness") or {}).get("passed"),
            "medianMs": benchmark.get("latencyMs"),
            "incumbentMedianMs": comparison.get("incumbentLatencyMs"),
            "improvementFraction": comparison.get("improvementFraction"),
            "pairedImprovements": comparison.get("pairedImprovements"),
            "wallSeconds": run["wallSeconds"],
        }
        points.append(point)
        print(f"[sweep] block_size={block_size:>5}: status={point['status']} "
              f"median={point['medianMs']}ms improvement={point['improvementFraction']} "
              f"wall={point['wallSeconds']}s")
    sweep_record = {
        "taskSha256": sha256_file(task_path),
        "incumbent": "baseline.py",
        "variable": "block_size",
        "points": points,
    }
    write_text(out_dir / "sweep_block_size.json",
               json.dumps(sweep_record, ensure_ascii=False, indent=2) + NL)
    manifest["sweep"] = sweep_record
    return sweep_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-root", type=Path, required=True,
                        help="part3-agent 目录（含 chapter14/evaluate.py）")
    parser.add_argument("--out", type=Path, default=SUITE_ROOT / "evidence")
    parser.add_argument("--sweep", action="store_true", help="同时跑 block_size 单变量扫描")
    parser.add_argument("--freeze", action="store_true",
                        help="把当前 candidates/ 三个固定输入的 sha256 写入 frozen_inputs.json")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frozen_path = SUITE_ROOT / "frozen_inputs.json"
    if args.freeze or not frozen_path.is_file():
        frozen = {name: sha256_file(CANDIDATES / name)
                  for name in ["task.json", "reference.py", "baseline.py"]}
        write_text(frozen_path, json.dumps(frozen, indent=2) + NL)
        print(f"frozen inputs recorded in {frozen_path.name}")
    frozen = load_json(frozen_path)

    manifest = {
        "suite": "task7-acceptance",
        "purpose": "固定输入 + 明确预期 + 证据：验收 hello-gpu Day5 kernel_optimize 项目的权威裁判",
        "partRoot": str(args.part_root),
        "frozenInputs": frozen,
        "inputIdentity": {},
        "environment": host_environment(),
        "evaluatorTimeoutSeconds": 600,
        "cases": [],
    }
    # 运行前先核对固定输入身份：任何一个文件被改动，验收不再有效。
    for name, expected in frozen.items():
        actual = sha256_file(CANDIDATES / name)
        manifest["inputIdentity"][name] = {
            "expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            print(f"[FAIL] fixed input {name} changed since freeze: {expected} -> {actual}")
    if not all(entry["match"] for entry in manifest["inputIdentity"].values()):
        raise SystemExit("fixed inputs changed; refusing to run acceptance")

    failures = 0
    for case_path in sorted(CASES.glob("case*.json")):
        record = run_case(args.part_root, case_path, args.out, manifest)
        failures += 0 if record["verdict"]["passed"] else 1
    if args.sweep:
        run_sweep(args.part_root, args.out, manifest)
    manifest["summary"] = {
        "casesTotal": len(manifest["cases"]),
        "casesFailed": failures,
        "finishedAtUtc": datetime.now(timezone.utc).isoformat(),
    }
    write_text(args.out / "manifest.json",
               json.dumps(manifest, ensure_ascii=False, indent=2) + NL)
    print(NL + "manifest: " + str(args.out / "manifest.json"))
    print(f"cases: {len(manifest['cases']) - failures}/{len(manifest['cases'])} met frozen expectations")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
