#!/usr/bin/env python3
"""最小 Rubric 验收门：读证据、对 Rubric、失败就关闭（退出码 1）。

Rubric（全部满足才放行；任何一条不满足 -> exit 1，验收不通过）：
  G1 固定输入身份：manifest.inputIdentity 全部 match（输入没被改动）
  G2 caseA：裁判接受正确 baseline（status=ok 且 correctness.passed）
  G3 caseB：裁判接受单变量优化候选（status=ok、correctness.passed、
     有配对比较、候选与 incumbent 源码不同）
  G4 caseC：裁判拒绝算错的候选（status=wrong_answer 且 correctness.passed=false）
     ——门禁必须真的会拒绝，只发通过的门禁不是验收
  G5 单变量扫描（有则查）：>=3 个数据点、全部 ok+passed、同一 task 契约
  G6 证据完整性：manifest 记录的每个 case 证据文件都在、可解析，
     且原始 result.json 的 status 与 manifest 记录一致（防篡改对账）

用法：
  python3 rubric_gate.py --evidence evidence/
  python3 rubric_gate.py --evidence evidence/ --selftest-negative   # 自测：门必须关闭
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

GATES = [
    ("G1", "fixed-input identity"),
    ("G2", "caseA accepts the correct baseline"),
    ("G3", "caseB accepts the single-variable candidate with paired comparison"),
    ("G4", "caseC rejects the wrong answer"),
    ("G5", "single-variable sweep is complete and contract-stable"),
    ("G6", "evidence files exist, parse, and match the manifest"),
]

NL = chr(10)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gates(evidence: Path) -> dict:
    findings = []

    def record(gate: str, name: str, passed: bool, detail: str) -> None:
        findings.append({"gate": gate, "name": name, "passed": bool(passed), "detail": detail})

    manifest_path = evidence / "manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "findings": [
            {"gate": "G0", "name": "manifest present", "passed": False,
             "detail": "missing " + str(manifest_path)}], "gates": GATES}
    manifest = load_json(manifest_path)
    cases = {case["case"]["id"]: case for case in manifest.get("cases", [])}

    # G1 固定输入身份
    identity = manifest.get("inputIdentity", {})
    identity_ok = bool(identity) and all(entry.get("match") for entry in identity.values())
    record("G1", "fixed-input identity", identity_ok,
           json.dumps({k: v.get("match") for k, v in identity.items()}, ensure_ascii=False))

    # G2 caseA
    case_a = cases.get("caseA_accept_baseline")
    if case_a is None:
        record("G2", "caseA accepts the correct baseline", False, "caseA missing")
    else:
        result = case_a["result"]
        ok = result.get("status") == "ok" and (result.get("correctness") or {}).get("passed") is True
        record("G2", "caseA accepts the correct baseline", ok,
               "status=" + str(result.get("status"))
               + " correctness=" + str((result.get("correctness") or {}).get("passed")))

    # G3 caseB
    case_b = cases.get("caseB_accept_improved")
    if case_b is None:
        record("G3", "caseB accepts the single-variable candidate with paired comparison",
               False, "caseB missing")
    else:
        result = case_b["result"]
        comparison = result.get("comparison") or {}
        differs = ((case_b.get("candidate") or {}).get("sha256")
                   != (case_b.get("incumbent") or {}).get("sha256"))
        ok = (result.get("status") == "ok"
              and (result.get("correctness") or {}).get("passed") is True
              and bool(comparison)
              and comparison.get("improvementFraction") is not None
              and differs)
        record("G3", "caseB accepts the single-variable candidate with paired comparison", ok,
               "status=" + str(result.get("status"))
               + " improvement=" + str(comparison.get("improvementFraction"))
               + " differs=" + str(differs))

    # G4 caseC
    case_c = cases.get("caseC_reject_wrong")
    if case_c is None:
        record("G4", "caseC rejects the wrong answer", False, "caseC missing")
    else:
        result = case_c["result"]
        ok = (result.get("status") == "wrong_answer"
              and (result.get("correctness") or {}).get("passed") is False)
        record("G4", "caseC rejects the wrong answer", ok,
               "status=" + str(result.get("status"))
               + " correctness=" + str((result.get("correctness") or {}).get("passed")))

    # G5 sweep
    sweep_path = evidence / "sweep_block_size.json"
    if not sweep_path.is_file():
        record("G5", "single-variable sweep is complete and contract-stable", False,
               "sweep_block_size.json missing (run with --sweep)")
    else:
        sweep = load_json(sweep_path)
        points = sweep.get("points", [])
        task_shas = {manifest.get("frozenInputs", {}).get("task.json"),
                     sweep.get("taskSha256")}
        ok = (len(points) >= 3
              and all(point.get("status") == "ok" and point.get("correctnessPassed") is True
                      for point in points)
              and len(task_shas) == 1 and None not in task_shas
              and all(point.get("medianMs") is not None for point in points))
        record("G5", "single-variable sweep is complete and contract-stable", ok,
               "points=" + str(len(points))
               + " medians=" + str([p.get("medianMs") for p in points]))

    # G6 证据完整性：文件在、可解析、status 与 manifest 一致
    mismatches = []
    for case in manifest.get("cases", []):
        case_id = case["case"]["id"]
        for name in case.get("evidenceFiles", []):
            path = evidence / name
            if not path.is_file():
                mismatches.append(name + ": missing")
                continue
            if name.endswith(".result.json"):
                try:
                    raw = load_json(path)
                except json.JSONDecodeError as error:
                    mismatches.append(name + ": " + str(error))
                    continue
                if raw.get("status") != case["result"].get("status"):
                    mismatches.append(name + ": status " + str(raw.get("status"))
                                      + " != manifest " + str(case["result"].get("status")))
    record("G6", "evidence files exist, parse, and match the manifest", not mismatches,
           "; ".join(mismatches) if mismatches else "all evidence files consistent")

    return {"passed": all(f["passed"] for f in findings), "findings": findings, "gates": GATES}


def selftest_negative(evidence: Path) -> int:
    """负向自测：把 caseA 证据篡改成失败，门必须关闭（证明失败就关闭不是空话）。"""
    with tempfile.TemporaryDirectory(prefix="rubric-gate-selftest-") as tmp:
        tampered = Path(tmp) / "evidence"
        shutil.copytree(evidence, tampered)
        target = tampered / "caseA_accept_baseline.result.json"
        payload = load_json(target)
        payload["status"] = "wrong_answer"
        payload["correctness"] = {
            "passed": False,
            "reason": "SELFTEST: tampered by rubric_gate --selftest-negative",
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + NL,
                          encoding="utf-8")
        outcome = evaluate_gates(tampered)
        closed = not outcome["passed"]
        g2 = next(f for f in outcome["findings"] if f["gate"] == "G2")
        print("[selftest] tampered caseA -> gate passed=" + str(outcome["passed"])
              + " (must be False)")
        print("[selftest] G2 detail: " + g2["detail"])
        print("[selftest] " + ("OK: the gate closes on failure" if closed
                               else "BROKEN: gate let a failure through"))
        return 0 if closed else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--selftest-negative", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    outcome = evaluate_gates(args.evidence)
    print("=== Task7 minimal Rubric gate ===")
    for finding in outcome["findings"]:
        mark = "PASS" if finding["passed"] else "FAIL"
        print(f"[{mark}] {finding['gate']} {finding['name']}: {finding['detail']}")
    verdict = "ACCEPTED (gate open)" if outcome["passed"] else "REJECTED (gate closed)"
    print("=== verdict: " + verdict + " ===")
    if args.json_out:
        args.json_out.write_text(json.dumps({
            **outcome,
            "verdict": verdict,
            "checkedAtUtc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2) + NL, encoding="utf-8")
    if args.selftest_negative:
        code = selftest_negative(args.evidence)
        return code if code != 0 else (0 if outcome["passed"] else 1)
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
