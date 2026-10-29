#!/usr/bin/env python3
"""实验 B 扩样：在更大的陷阱集上重测 §10.2 表 10-1 的"审查是否需要新信息"。

背景：exp10-1b-adv 只有 14 题 ×2 轮，判别样本 2 个，McNemar 精确 p = 0.500，
      在设计上不可能拒绝零假设。本脚本用 35 题的新陷阱集（B01-B35，ground truth 全部来自
      本机真实执行）做一次样本量足够的重测。

设计（与 exp_b_verify_evidence.py 完全同构，确保可比）：
  G   生成者：只看题目，无执行能力
  B1  无新信息审查：只看题目 + 生成者答案与理由
  B2  有新信息审查：额外看到 harness 真实执行的 stdout/stderr/returncode
  两臂使用同一份 prompt 模板与同一模型，唯一差别是 B2 多出执行证据。

用法：
  python exp_b_extension.py                 # 跑 1 轮（35 题 × 3 阶段 = 105 次调用）
  python exp_b_extension.py --rounds 2      # 两轮，样本翻倍
  python exp_b_extension.py --rescore       # 只读已落盘原始 JSON 确定性重判，不调模型
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import agents_core as core  # noqa: E402
import exp_b_verify_evidence as eb  # noqa: E402  复用同一套 prompt 与判分规则

EXP_DIR = os.path.join(core.EVIDENCE_ROOT, "exp10-1b-ext")

TASKS_PATH = os.path.join(EXP_DIR, "tasks.json")


def load_tasks() -> Dict[str, Any]:
    return core.load_json(TASKS_PATH)


def round_paths(round_idx: int) -> Dict[str, str]:
    if round_idx == 1:
        return {
            "gen": os.path.join(EXP_DIR, "generation.json"),
            "b1": os.path.join(EXP_DIR, "review_B1.json"),
            "b2": os.path.join(EXP_DIR, "review_B2.json"),
        }
    return {
        "gen": os.path.join(EXP_DIR, "generation_r%d.json" % round_idx),
        "b1": os.path.join(EXP_DIR, "review_B1_r%d.json" % round_idx),
        "b2": os.path.join(EXP_DIR, "review_B2_r%d.json" % round_idx),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """精确双尾 McNemar：b = 仅 B2 正确，c = 仅 B1 正确。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run_round(client, tasks: List[Dict[str, Any]], round_idx: int, py_ver: str,
              stages: List[str]) -> None:
    round_tasks = tasks if round_idx == 1 else [
        dict(t, id="%s__r%d" % (t["id"], round_idx)) for t in tasks
    ]
    eb.drive(client, round_tasks, round_paths(round_idx), py_ver, stages)


def load_round(base: str) -> List[Dict[str, Any]]:
    out = []
    prefix = base[:-5]
    for fn in sorted(os.listdir(EXP_DIR)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        if stem == prefix or (stem.startswith(prefix + "_r") and stem[len(prefix) + 2:].isdigit()):
            for row in core.load_json(os.path.join(EXP_DIR, fn)):
                row["_file"] = fn
                out.append(row)
    return out


def summarize(tasks: List[Dict[str, Any]], gens: List[Dict[str, Any]],
              b1: List[Dict[str, Any]], b2: List[Dict[str, Any]]) -> Dict[str, Any]:
    return eb.summarize(tasks, gens, b1, b2)


def build_summary(rounds: int) -> Dict[str, Any]:
    tobj = load_tasks()
    tasks = tobj["tasks"]
    all_tasks = tasks + [
        dict(t, id="%s__r%d" % (t["id"], r)) for r in range(2, rounds + 1) for t in tasks
    ]
    gens = load_round("generation.json")
    b1 = load_round("review_B1.json")
    b2 = load_round("review_B2.json")

    ids = ({g["task_id"] for g in gens} & {r["task_id"] for r in b1} & {r["task_id"] for r in b2})

    # —— 调用失败 ≠ 模型答错 ——
    # 端点抖动时（HTTP 502）有的记录 answer=None，如果直接按"答案不匹配"计入，
    # 就会把"没拿到回复"当成"模型答错了"，人为抬高生成者的错误率，
    # 进而让 B1/B2 的修正率被虚高。这里把三次调用里任何一次失败的样本整体剔除，
    # 并把剔除清单写进 summary，任何人都能复核剔除了哪些。
    def call_failed(rows):
        bad = set()
        for r in rows:
            rec = r.get("record") or {}
            if r.get("call_ok") is False or rec.get("ok") is False:
                bad.add(r["task_id"])
                continue
            # 空答案也可能是模型的真实输出（例如 B31 抛异常的题），
            # 只有"答案缺失 + 调用带错误信息"两件事同时成立才算调用失败。
            if r.get("answer") is None and rec.get("error"):
                bad.add(r["task_id"])
        return bad

    excluded = sorted(call_failed(gens) | call_failed(b1) | call_failed(b2))
    ids = ids - set(excluded)
    all_tasks = [t for t in all_tasks if t["id"] in ids]
    g_s = sorted([g for g in gens if g["task_id"] in ids], key=lambda x: x["task_id"])
    b1_s = sorted([r for r in b1 if r["task_id"] in ids], key=lambda x: x["task_id"])
    b2_s = sorted([r for r in b2 if r["task_id"] in ids], key=lambda x: x["task_id"])

    s = summarize(all_tasks, g_s, b1_s, b2_s)
    s["task_set"] = {
        "path": TASKS_PATH,
        "n_distinct_tasks": len(tasks),
        "n_samples": len(all_tasks),
        "rounds": rounds,
        "ground_truth_note": tobj["note"],
        "purpose": "为 §10.2 表 10-1 的判据补足样本量（原 exp10-1b-adv 只有 14 题/2 个判别样本）",
        "excluded_call_failures": excluded,
        "n_excluded": len(excluded),
        "exclusion_rule": "G/B1/B2 任一阶段调用本身失败（record.ok == False 或带 error）的样本整体剔除；不把'没拿到回复'当成'模型答错'",
    }

    # 精确 McNemar（以 final_correct 为准），并给出达到 p<0.05 所需的最少判别样本数
    by_b1 = {r["task_id"]: r for r in b1_s}
    by_b2 = {r["task_id"]: r for r in b2_s}
    only_b2 = [tid for tid in by_b1 if by_b2[tid]["final_correct"] and not by_b1[tid]["final_correct"]]
    only_b1 = [tid for tid in by_b1 if by_b1[tid]["final_correct"] and not by_b2[tid]["final_correct"]]
    s["paired_exact"] = {
        "only_B2_correct": len(only_b2),
        "only_B1_correct": len(only_b1),
        "only_B2_ids": only_b2,
        "only_B1_ids": only_b1,
        "mcnemar_exact_p_two_sided": round(mcnemar_exact(len(only_b2), len(only_b1)), 6),
        "min_discordant_for_p_below_0.05": 6,
        "note": "6 个全向同一侧的判别样本是精确 McNemar 达到 p<0.05 的下限（2*(1/64)=0.03125）",
    }

    # 分层：只在生成者答错的样本上比较两臂（这才是判据真正作用的子集）
    wrong = [t["id"] for t in all_tasks if not next(g for g in g_s if g["task_id"] == t["id"])["correct"]]
    b1_wrong = [by_b1[t] for t in wrong]
    b2_wrong = [by_b2[t] for t in wrong]
    s["stratum_generator_wrong"] = {
        "n": len(wrong),
        "ids": wrong,
        "B1_final_correct": sum(1 for r in b1_wrong if r["final_correct"]),
        "B2_final_correct": sum(1 for r in b2_wrong if r["final_correct"]),
        "B1_fixed": sum(1 for r in b1_wrong if r["final_correct"]),
        "B2_fixed": sum(1 for r in b2_wrong if r["final_correct"]),
        "B1_broke": sum(1 for r in b1_wrong if r["generator_correct"] and not r["final_correct"]),
        "B2_broke": sum(1 for r in b2_wrong if r["generator_correct"] and not r["final_correct"]),
    }
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1, help="重复轮数，样本量 = 35 × rounds")
    ap.add_argument("--stages", default="G,B1,B2")
    ap.add_argument("--model", default=core.MODEL)
    ap.add_argument("--rescore", action="store_true", help="只重算 summary，不调模型")
    args = ap.parse_args()

    if not os.path.exists(TASKS_PATH):
        print("missing task set: %s (run .pdfwork/make_ext_tasks.py first)" % TASKS_PATH, file=sys.stderr)
        return 2
    tobj = load_tasks()
    py_ver = tobj.get("python_version", "?")
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]

    if not args.rescore:
        client = core.ResponsesClient(model=args.model)
        for r in range(1, args.rounds + 1):
            print("=== extension round %d (%d tasks) ===" % (r, tobj["n_tasks"]), flush=True)
            run_round(client, tobj["tasks"], r, py_ver, stages)
        print("[client stats]", json.dumps(client.stats(), ensure_ascii=False), flush=True)

    s = build_summary(args.rounds)
    core.write_json(os.path.join(EXP_DIR, "summary.json"), s)

    print("\n=== extension summary (n=%d) ===" % s["task_set"]["n_samples"])
    print("generator acc = %s (%d/%d)" % (s["generator"]["accuracy"],
                                          s["generator"]["n_correct"], s["generator"]["n"]))
    for k in ("arm_B1_no_new_info", "arm_B2_execution_evidence"):
        d = s[k]
        print("%s: revise=%s change_rate=%s acc=%s fixed=%d broke=%d tokens=%d" % (
            k, d["verdict_counts"].get("revise"), d["change_rate"], d["final_accuracy"],
            d["n_fixed_wrong_to_right"], d["n_broke_right_to_wrong"], d["total_tokens"]))
    print("paired_exact:", json.dumps(s["paired_exact"], ensure_ascii=False))
    print("stratum(generator wrong):", json.dumps(s["stratum_generator_wrong"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
