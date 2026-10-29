# -*- coding: utf-8 -*-
"""
exp_b_verify_evidence.py -- §10.2 表10-1 的核心判据：审核者引入新信息才有用

设计：
  任务集：24 道“必须执行才知道答案”的 Python 片段题（可变默认参数、闭包晚绑定、浮点、
          切片赋值、生成器耗尽、格式化、dict 行为……）。正确答案**由本机真实执行得出**，
          不是人写的、也不是模型自述的。
  阶段 G（生成）：生成者只看到题目（不能执行代码），给出答案+理由+自评置信度。
  阶段 B1（无新信息审查）：审查者只看到题目 + 生成者的答案与理由，重新推理后决定是否修改。
  阶段 B2（有新信息审查）：审查者额外看到 **真实执行结果**（python -c 的 stdout/stderr/返回码，
          由 harness 实际跑出来），据此决定是否修改。
  判据：比较两臂的修正率（改答案比例）与正确率（最终答案 == 真实执行输出）。

产出：evidence/task5/exp10-1b/
  tasks.json     任务集 + 真实执行结果（可用 --build-tasks 重建）
  generation.json 说明：本题集里问的是“输出什么”，所以阶段名为 generation（生成者回答）
  review_B1.json / review_B2.json  两臂逐题原始响应
  summary.json   笔记里每个数字都能从这里重算

用法：
  python exp_b_verify_evidence.py --build-tasks
  python exp_b_verify_evidence.py --limit 3
  python exp_b_verify_evidence.py
  python exp_b_verify_evidence.py --summarize-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from math import comb
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agents_core as core  # noqa: E402

EXP_DIR = os.path.join(core.EVIDENCE_ROOT, "exp10-1b")
TASKS_PATH = os.path.join(EXP_DIR, "tasks.json")
GEN_PATH = os.path.join(EXP_DIR, "generation.json")
B1_PATH = os.path.join(EXP_DIR, "review_B1.json")
B2_PATH = os.path.join(EXP_DIR, "review_B2.json")
EXP_DIR_ADV = os.path.join(core.EVIDENCE_ROOT, "exp10-1b-adv")
TASKS_ADV_PATH = os.path.join(EXP_DIR_ADV, "tasks.json")
GEN_ADV_PATH = os.path.join(EXP_DIR_ADV, "generation.json")
B1_ADV_PATH = os.path.join(EXP_DIR_ADV, "review_B1.json")
B2_ADV_PATH = os.path.join(EXP_DIR_ADV, "review_B2.json")
PY = sys.executable
# ----------------------------------------------------------------------------- 任务集
TASK_SPECS: List[Dict[str, str]] = [
    {"id": "T01", "tag": "mutable-default-arg",
     "question": "下面这段 Python 代码输出什么？",
     "code": "def f(x, s=[]):\n    s.append(x)\n    return s\nprint(f(1), f(2), f(3))"},
    {"id": "T02", "tag": "late-binding-closure",
     "question": "下面这段 Python 代码输出什么？",
     "code": "fs = [lambda: i for i in range(3)]\nprint([f() for f in fs])"},
    {"id": "T03", "tag": "float-assoc",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(0.1 + 0.2 == 0.3, round(0.1 + 0.2, 10) == 0.3)"},
    {"id": "T04", "tag": "list-mult-alias",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = [[0] * 2] * 3\na[0][0] = 1\nprint(a)"},
    {"id": "T05", "tag": "floor-div-negative",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(-7 // 2, -7 % 2, int(-7 / 2))"},
    {"id": "T06", "tag": "dict-insertion-order",
     "question": "下面这段 Python 代码输出什么？",
     "code": "d = {'b': 1, 'a': 2}\nd['c'] = 3\nprint(list(d.keys()))"},
    {"id": "T07", "tag": "try-finally-return",
     "question": "下面这段 Python 代码输出什么？",
     "code": "def g():\n    try:\n        return 1\n    finally:\n        print('finally')\nprint(g())"},
    {"id": "T08", "tag": "class-attribute-shared",
     "question": "下面这段 Python 代码输出什么？",
     "code": "class A:\n    xs = []\nA.xs.append(1)\nb = A()\nb.xs.append(2)\nprint(A.xs)"},
    {"id": "T09", "tag": "sort-tuple",
     "question": "下面这段 Python 代码输出什么？",
     "code": "pairs = [(1, 'b'), (1, 'a'), (0, 'z')]\nprint(sorted(pairs))"},
    {"id": "T10", "tag": "str-ops",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print('ab' * 3, 'ab' + '3')"},
    {"id": "T11", "tag": "enumerate-start",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(list(enumerate('abc', start=1)))"},
    {"id": "T12", "tag": "generator-exhaustion",
     "question": "下面这段 Python 代码输出什么？",
     "code": "g = (x * 2 for x in range(3))\nprint(list(g), list(g))"},
    {"id": "T13", "tag": "bool-int-arith",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(True + True, [True, False].count(1), sum([True, True, False]))"},
    {"id": "T14", "tag": "format-spec",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(f'{3.14159:.2f}', f'{42:05d}', f'{0.5:.0%}')"},
    {"id": "T15", "tag": "set-ops",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(sorted({1, 2, 3} - {2}), sorted({1, 2} & {2, 3}), len({1, 1.0, True}))"},
    {"id": "T16", "tag": "exception-else",
     "question": "下面这段 Python 代码输出什么？",
     "code": "try:\n    x = int('4')\nexcept ValueError:\n    print('bad')\nelse:\n    print('ok', x)"},
    {"id": "T17", "tag": "list-slice-assign",
     "question": "下面这段 Python 代码输出什么？",
     "code": "L = [1, 2, 3, 4, 5]\nL[1:3] = [9]\nprint(L)"},
    {"id": "T18", "tag": "float-repr",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(0.1 + 0.7, 1e16 + 1 == 1e16)"},
    {"id": "T19", "tag": "json-dumps",
     "question": "下面这段 Python 代码输出什么？",
     "code": "import json\nprint(json.dumps({'a': None, 'b': True}, sort_keys=True, separators=(',', ':')))"},
    {"id": "T20", "tag": "setdefault",
     "question": "下面这段 Python 代码输出什么？",
     "code": "d = {}\nd.setdefault('k', []).append(1)\nd.setdefault('k', []).append(2)\nprint(d)"},
    {"id": "T21", "tag": "recursion-fib",
     "question": "下面这段 Python 代码输出什么？",
     "code": "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\nprint([fib(i) for i in range(7)])"},
    {"id": "T22", "tag": "str-comparison",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print('10' < '9', sorted(['10', '9', '2']))"},
    {"id": "T23", "tag": "dict-get-default-eval",
     "question": "下面这段 Python 代码输出什么？",
     "code": "calls = []\ndef d():\n    calls.append(1)\n    return 0\nprint({}.get('x', d()), len(calls))"},
    {"id": "T24", "tag": "percent-format",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print('%d%%' % 50, '%.1f' % 2.25)"},
]


def run_code(code: str, timeout: int = 20) -> Dict[str, Any]:
    t0 = time.time()
    try:
        cp = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                            timeout=timeout, encoding="utf-8", errors="replace")
        return {"returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr,
                "wall_s": round(time.time() - t0, 3)}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "stdout": "", "stderr": "TIMEOUT", "wall_s": round(time.time() - t0, 3)}


def build_tasks() -> Dict[str, Any]:
    """真实执行每一道题，得到 ground truth。"""
    tasks = []
    for spec in TASK_SPECS:
        r = run_code(spec["code"])
        tasks.append({
            "id": spec["id"], "tag": spec["tag"], "question": spec["question"], "code": spec["code"],
            "ground_truth_stdout": r["stdout"].rstrip("\n"),
            "ground_truth_returncode": r["returncode"],
            "exec_stderr": r["stderr"],
            "exec_wall_s": r["wall_s"],
        })
    obj = {
        "built_at": core.now_iso(),
        "executed_by": PY,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "n_tasks": len(tasks),
        "note": "ground truth 由本机真实执行 stdlib Python 得到，不是模型自述。",
        "tasks": tasks,
    }
    core.write_json(TASKS_PATH, obj)
    return obj
# ----------------------------------------------------------------------------- 规范化与判定
def norm(s: Any) -> str:
    """答案规范化：只做首尾空白、引号风格、多余空白、逗号空格的归一。"""
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    return t.strip().strip("'").strip('"').strip("`").strip()


def answer_matches(answer: Any, truth: str) -> bool:
    if answer is None:
        return False
    n_truth = norm(truth)
    if not n_truth:
        return False
    cands = [answer]
    for line in str(answer).splitlines():
        cands.append(line.strip())
    return any(norm(c) == n_truth for c in cands if norm(c))


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    for m in reversed(list(re.finditer(r"\{", text))):
        sub = text[m.start():]
        depth = 0
        for i, ch in enumerate(sub):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(sub[:i + 1])
                    except Exception:  # noqa: BLE001
                        pass
                    break
        if depth != 0:
            continue
    return None


def binom_exact_p(k: int, nn: int) -> float:
    """McNemar 精确检验（双尾）：在 H0 下判别方向应二项(1/2)分布。"""
    if nn == 0:
        return 1.0
    probs = [comb(nn, i) / (2 ** nn) for i in range(nn + 1)]
    obs = probs[k]
    p = sum(x for x in probs if x <= obs + 1e-12)
    return round(min(1.0, p), 4)


# ----------------------------------------------------------------------------- prompt 构造
def gen_prompt(t: Dict[str, Any]) -> str:
    return (
        "你是一名 Python 专家。你不能运行代码，只能靠推理作答。\n\n"
        + t["question"] + "\n```python\n" + t["code"] + "\n```\n\n"
        "严格只输出一个 JSON 对象，不要写多余文字：\n"
        '{"answer": "<代码的真实 stdout，原样输出，含方括号/引号/空格>", "reason": "<一句话理由>", '
        '"confidence": <0-1 的小数>}'
    )


def review_prompt(t: Dict[str, Any], gen: Dict[str, Any], arm: str, py_ver: str) -> str:
    p = (
        "你在审查另一位工程师对下面这段 Python 代码输出的答案。\n\n"
        + t["question"] + "\n```python\n" + t["code"] + "\n```\n\n"
        "【生成者的答案】\n"
        + json.dumps({"answer": gen.get("answer"), "reason": gen.get("reason")}, ensure_ascii=False)
        + "\n\n"
    )
    if arm == "B2":
        p += ("【harness 真实执行结果】（python -c，stdlib Python " + py_ver + "）\n"
              + "returncode = %s\n" % t.get("ground_truth_returncode")
              + "stdout = %r\n" % t.get("ground_truth_stdout")
              + "stderr = %r\n\n" % (t.get("exec_stderr") or ""))
    p += ("你的任务：判断生成者的答案是否正确；若错误，给出你认为正确的答案。"
          "你可以修改，也可以保持原答案不变。\n"
          "严格只输出一个 JSON 对象：\n"
          '{"verdict": "agree" 或 "revise", "final_answer": "<你最终认可的答案>", '
          '"confidence": <0-1 的小数>, "reason": "<一句话理由>"}')
    return p


def sys_prompt(arm: str) -> str:
    if arm == "B1":
        return ("你是一名严格的代码审查员，负责凭自己的推理复核同事的答案。"
                "你没有运行代码的能力，也没有任何执行结果可参考；请把自己的推理重新走一遍再下判断。")
    return ("你是一名严格的代码审查员。harness 已实际运行了这段代码，把真实的 stdout/stderr 提供给你。"
            "执行结果是事实来源：若它与生成者的答案冲突，以执行结果为准并改判为 revise；"
            "若两者一致，你可以维持原答案。")


# ----------------------------------------------------------------------------- 各阶段
def run_generation(client: core.ResponsesClient, tasks: List[Dict[str, Any]],
                   max_output_tokens: int = 3000) -> List[Dict[str, Any]]:
    out = []
    for t in tasks:
        p = gen_prompt(t)
        rec, payload = client.call([{"role": "user", "content": p}],
                                   instructions="你是一名 Python 专家。", max_output_tokens=max_output_tokens)
        text = core.ResponsesClient.extract_text(payload) if payload else ""
        parsed = parse_json_block(text) or {}
        ans = parsed.get("answer")
        call_ok = payload is not None
        answer_present = bool(ans is not None and str(ans).strip() != "")
        out.append({
            "task_id": t["id"], "tag": t["tag"], "prompt": p,
            "raw_response": payload, "raw_text": text,
            "answer": ans, "reason": parsed.get("reason"), "confidence": parsed.get("confidence"),
            # call_ok=False 表示这次调用本身没拿到有效响应（重试耗尽），
            # 它必须与"模型答错了"分开记：否则生成者的错误率会被端点抖动人为抬高。
            "call_ok": call_ok,
            # 2026-10-27 追加：响应状态与"是否真的产出答案"。
            # reasoning.effort=max 时，模型可能把全部输出预算花在推理上，
            # 于是 status=incomplete(max_output_tokens)、output_text 为空。
            # 这种记录调用是"成功"的、record.ok=True，但没有答案；
            # 字面匹配会把它算成答错，从而凭空造出判别样本。必须单独标记。
            "response_status": (payload or {}).get("status"),
            "incomplete_reason": ((payload or {}).get("incomplete_details") or {}).get("reason"),
            "answer_present": answer_present,
            "correct": call_ok and answer_present and answer_matches(ans, t["ground_truth_stdout"]),
            "record": rec.to_json(), "at": core.now_iso(),
        })
        print("  gen %s correct=%s ans=%r truth=%r" % (t["id"], out[-1]["correct"], ans,
                                                       t['ground_truth_stdout']), flush=True)
    return out


def run_review(client: core.ResponsesClient, tasks: List[Dict[str, Any]], gens: List[Dict[str, Any]],
               arm: str, py_ver: str, max_output_tokens: int = 3000) -> List[Dict[str, Any]]:
    by_id = {g["task_id"]: g for g in gens}
    out = []
    for t in tasks:
        g = by_id.get(t["id"])
        if g is None:
            continue
        p = review_prompt(t, g, arm, py_ver)
        rec, payload = client.call([{"role": "user", "content": p}],
                                   instructions=sys_prompt(arm), max_output_tokens=max_output_tokens)
        text = core.ResponsesClient.extract_text(payload) if payload else ""
        parsed = parse_json_block(text) or {}
        final = parsed.get("final_answer")
        verdict = parsed.get("verdict")
        changed = not answer_matches(final, g.get("answer"))
        call_ok = payload is not None
        answer_present = bool(final is not None and str(final).strip() != "")
        row = {
            "task_id": t["id"], "tag": t["tag"], "arm": arm, "prompt": p,
            "raw_response": payload, "raw_text": text,
            "generator_answer": g.get("answer"), "generator_correct": g["correct"],
            "verdict": verdict, "final_answer": final,
            "confidence": parsed.get("confidence"), "reason": parsed.get("reason"),
            "changed": changed,
            "call_ok": call_ok,
            "response_status": (payload or {}).get("status"),
            "incomplete_reason": ((payload or {}).get("incomplete_details") or {}).get("reason"),
            "answer_present": answer_present,
            "final_correct": call_ok and answer_present and answer_matches(final, t["ground_truth_stdout"]),
            "record": rec.to_json(), "at": core.now_iso(),
        }
        out.append(row)
        print("  %s %s changed=%s final_correct=%s (gen_correct=%s)" % (
            arm, t["id"], changed, row["final_correct"], g["correct"]), flush=True)
    return out
# ----------------------------------------------------------------------------- 可判分性
# 2026-10-27 追加。三类记录必须与"模型答错"分开，否则端点抖动和输出预算会被
# 统计成模型错误，凭空造出判别样本：
#   1) 调用失败：重试耗尽，payload 为 None（call_ok=False）
#   2) 响应不完整：status=incomplete，输出预算被推理吃光，从未产出答案文本
#   3) 真值为空：程序抛异常导致 stdout 为空，字面匹配无法判分（由调用方按 tasks 过滤）
# 老数据（main 集）没有 call_ok 字段，用 record.ok 兜底，保证历史 summary 仍可复现。
def call_succeeded(row: Dict[str, Any]) -> bool:
    if "call_ok" in row:
        return bool(row["call_ok"])
    return bool((row.get("record") or {}).get("ok", True))


def answer_was_produced(row: Dict[str, Any], key: str) -> bool:
    if "answer_present" in row:
        return bool(row["answer_present"])
    status = (row.get("raw_response") or {}).get("status")
    if status is not None and status != "completed":
        return False
    val = row.get(key)
    return val is not None and str(val).strip() != ""


def scorable(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    return [r for r in rows if call_succeeded(r) and answer_was_produced(r, key)]


# ----------------------------------------------------------------------------- 汇总
_ALL_GENS: List[Dict[str, Any]] = []
_ALL_B1: List[Dict[str, Any]] = []
_ALL_B2: List[Dict[str, Any]] = []


def summarize(tasks: List[Dict[str, Any]], gens: List[Dict[str, Any]],
              b1: List[Dict[str, Any]], b2: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 过滤前的原始行留给 excluded 统计用，避免筛过之后再看漏了多少。
    global _ALL_GENS, _ALL_B1, _ALL_B2
    _ALL_GENS, _ALL_B1, _ALL_B2 = list(gens), list(b1), list(b2)
    # 只统计真的产出过答案的样本；没有答案的记录单独计数，绝不当成答错。
    gens = scorable(gens, "answer")
    b1 = scorable(b1, "final_answer")
    b2 = scorable(b2, "final_answer")
    n = len(tasks)
    gen_correct = sum(1 for g in gens if g["correct"])
    gen_tokens = sum(g["record"]["input_tokens"] + g["record"]["output_tokens"] for g in gens)
    gen_wall = sum(g["record"]["wall_s"] for g in gens)

    def arm_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        m = len(rows)
        changed = sum(1 for r in rows if r["changed"])
        final_correct = sum(1 for r in rows if r["final_correct"])
        fixed = sum(1 for r in rows if (not r["generator_correct"]) and r["final_correct"])
        broke = sum(1 for r in rows if r["generator_correct"] and (not r["final_correct"]))
        still_wrong = sum(1 for r in rows if (not r["generator_correct"]) and (not r["final_correct"]))
        conf = [r["confidence"] for r in rows if isinstance(r.get("confidence"), (int, float))]
        verdict_agree = sum(1 for r in rows if (r.get("verdict") or "").lower() == "agree")
        verdict_revise = sum(1 for r in rows if (r.get("verdict") or "").lower() == "revise")
        tot = sum(r["record"]["input_tokens"] + r["record"]["output_tokens"] for r in rows)
        return {
            "n": m,
            "n_changed": changed,
            "change_rate": round(changed / m, 4) if m else None,
            "n_final_correct": final_correct,
            "final_accuracy": round(final_correct / m, 4) if m else None,
            "n_fixed_wrong_to_right": fixed,
            "n_broke_right_to_wrong": broke,
            "n_still_wrong": still_wrong,
            "net_accuracy_gain_vs_generator": round((final_correct - gen_correct) / n, 4) if n else None,
            "mean_confidence": round(sum(conf) / len(conf), 4) if conf else None,
            "verdict_counts": {"agree": verdict_agree, "revise": verdict_revise,
                               "other_or_missing": m - verdict_agree - verdict_revise},
            "input_tokens": sum(r["record"]["input_tokens"] for r in rows),
            "output_tokens": sum(r["record"]["output_tokens"] for r in rows),
            "total_tokens": tot,
            "mean_tokens_per_task": round(tot / m, 1) if m else None,
            "mean_wall_s_per_task": round(sum(r["record"]["wall_s"] for r in rows) / m, 3) if m else None,
            "failed_attempts": sum(r["record"]["attempts"] - 1 for r in rows),
            "failed_calls": sum(0 if r["record"]["ok"] else 1 for r in rows),
            "per_task": [{"task_id": r["task_id"], "tag": r["tag"],
                          "generator_answer": r["generator_answer"],
                          "generator_correct": r["generator_correct"],
                          "verdict": r["verdict"], "final_answer": r["final_answer"],
                          "changed": r["changed"], "final_correct": r["final_correct"]} for r in rows],
        }

    s1, s2 = arm_stats(b1), arm_stats(b2)
    b_only2 = sum(1 for x, y in zip(b1, b2) if (not x["final_correct"]) and y["final_correct"])
    c_only1 = sum(1 for x, y in zip(b1, b2) if x["final_correct"] and (not y["final_correct"]))
    return {
        "generated_at": core.now_iso(),
        "scoring_note": (
            "只统计产出过答案的样本。调用失败（重试耗尽）与响应不完整"
            "（status=incomplete，输出预算被推理吃光）都不算模型答错，见 scorable()；"
            "真值为空（程序抛异常）的题由 tasks 侧过滤。excluded 记录被剔除的条数与原因。"
        ),
        "excluded": {
            "n_generator_no_answer": sum(1 for g in _ALL_GENS
                                         if not (call_succeeded(g) and answer_was_produced(g, "answer"))),
            "n_B1_no_answer": sum(1 for r in _ALL_B1
                                 if not (call_succeeded(r) and answer_was_produced(r, "final_answer"))),
            "n_B2_no_answer": sum(1 for r in _ALL_B2
                                 if not (call_succeeded(r) and answer_was_produced(r, "final_answer"))),
        },
        "model": core.MODEL,
        "n_tasks": n,
        "python_version": sys.version.split()[0],
        "generator": {
            "n": len(gens), "n_correct": gen_correct,
            "accuracy": round(gen_correct / len(gens), 4) if gens else None,
            "total_tokens": gen_tokens,
            "mean_tokens_per_task": round(gen_tokens / len(gens), 1) if gens else None,
            "mean_wall_s_per_task": round(gen_wall / len(gens), 3) if gens else None,
            "failed_calls": sum(0 if g["record"]["ok"] else 1 for g in gens),
            "failed_attempts": sum(g["record"]["attempts"] - 1 for g in gens),
            "per_task": [{"task_id": g["task_id"], "answer": g["answer"], "correct": g["correct"],
                          "reason": g["reason"], "confidence": g["confidence"]} for g in gens],
        },
        "arm_B1_no_new_info": s1,
        "arm_B2_execution_evidence": s2,
        "paired": {
            "n_pairs": min(len(b1), len(b2)),
            "n_only_B2_correct": b_only2,
            "n_only_B1_correct": c_only1,
            "mcnemar_exact_p_two_sided": binom_exact_p(b_only2, b_only2 + c_only1),
            "accuracy_delta_B2_minus_B1": round((s2["final_accuracy"] or 0) - (s1["final_accuracy"] or 0), 4),
            "change_rate_delta_B2_minus_B1": round((s2["change_rate"] or 0) - (s1["change_rate"] or 0), 4),
        },
        "cost": {
            "generation_tokens": gen_tokens,
            "B1_tokens": s1["total_tokens"],
            "B2_tokens": s2["total_tokens"],
            "single_pass_tokens": gen_tokens,
            "B1_pipeline_tokens": gen_tokens + s1["total_tokens"],
            "B2_pipeline_tokens": gen_tokens + s2["total_tokens"],
            "B1_overhead_ratio_vs_single_pass": round((gen_tokens + s1["total_tokens"]) / gen_tokens, 3) if gen_tokens else None,
            "B2_overhead_ratio_vs_single_pass": round((gen_tokens + s2["total_tokens"]) / gen_tokens, 3) if gen_tokens else None,
        },
    }


def merge_round(distinct_tasks: List[Dict[str, Any]], exp_dir: str, base_path: str,
                target: List[Dict[str, Any]]) -> None:
    """把 base_path 与 base_path_rN.json 各轮结果合并进 target。"""
    base_name = os.path.basename(base_path).replace(".json", "")
    files = []
    for fn in sorted(os.listdir(exp_dir)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        if stem == base_name or re.fullmatch(re.escape(base_name) + r"_r\d+", stem):
            files.append(os.path.join(exp_dir, fn))
    for fp in files:
        for row in core.load_json(fp):
            row["_file"] = os.path.basename(fp)
            target.append(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-tasks", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stages", default="G,B1,B2")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--model", default=core.MODEL)
    args = ap.parse_args()

    core.ensure_dir(EXP_DIR)
    if args.build_tasks or not os.path.exists(TASKS_PATH):
        tobj = build_tasks()
        print("[tasks] built %d tasks -> %s" % (tobj["n_tasks"], TASKS_PATH))
    tobj = core.load_json(TASKS_PATH)
    tasks = tobj["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    py_ver = tobj.get("python_version", "?")

    if not args.summarize_only:
        client = core.ResponsesClient(model=args.model)
        for rep in range(1, args.repeats + 1):
            if rep == 1:
                round_tasks = tasks
                paths = {"gen": GEN_PATH, "b1": B1_PATH, "b2": B2_PATH}
            else:
                round_tasks = [dict(t, id="%s__r%d" % (t["id"], rep)) for t in tasks]
                paths = {"gen": GEN_PATH.replace(".json", "_r%d.json" % rep),
                         "b1": B1_PATH.replace(".json", "_r%d.json" % rep),
                         "b2": B2_PATH.replace(".json", "_r%d.json" % rep)}
            print("=== main set round %d (%d tasks) ===" % (rep, len(round_tasks)), flush=True)
            drive(client, round_tasks, paths, py_ver, stages,
                      max_output_tokens=args.max_output_tokens,
                      retry_incomplete=args.retry_incomplete)
        print("[client stats]", json.dumps(client.stats(), ensure_ascii=False))

    all_tasks = tasks + [dict(t, id="%s__r%d" % (t["id"], rep)) for rep in range(2, args.repeats + 1) for t in tasks]
    gens, b1, b2 = [], [], []
    for key, base in (("gen", GEN_PATH), ("b1", B1_PATH), ("b2", B2_PATH)):
        merge_round(tasks, EXP_DIR, base, {"gen": gens, "b1": b1, "b2": b2}[key])
    if not (gens and b1 and b2):
        print("incomplete: gens=%d b1=%d b2=%d" % (len(gens), len(b1), len(b2)), file=sys.stderr)
        return 1
    ids = sorted({g["task_id"] for g in gens} & {r["task_id"] for r in b1} & {r["task_id"] for r in b2})
    tasks_sel = [t for t in all_tasks if t["id"] in ids]
    gens_s = sorted([g for g in gens if g["task_id"] in ids], key=lambda x: x["task_id"])
    b1_s = sorted([r for r in b1 if r["task_id"] in ids], key=lambda x: x["task_id"])
    b2_s = sorted([r for r in b2 if r["task_id"] in ids], key=lambda x: x["task_id"])
    s = summarize(tasks_sel, gens_s, b1_s, b2_s)
    s["task_set"] = {"path": TASKS_PATH, "n_tasks": len(tasks_sel),
                     "n_distinct_tasks": len(tasks), "repeats": args.repeats,
                     "ground_truth_note": tobj["note"], "built_at": tobj["built_at"]}
    core.write_json(os.path.join(EXP_DIR, "summary.json"), s)
    print("\n=== summary ===")
    print("generator acc = %s (%d/%d)" % (s["generator"]["accuracy"], s["generator"]["n_correct"], s["generator"]["n"]))
    for k in ("arm_B1_no_new_info", "arm_B2_execution_evidence"):
        d = s[k]
        print("%s: change_rate=%s acc=%s fixed=%d broke=%d tokens=%d" % (
            k, d["change_rate"], d["final_accuracy"], d["n_fixed_wrong_to_right"],
            d["n_broke_right_to_wrong"], d["total_tokens"]))
    print("paired:", json.dumps(s["paired"], ensure_ascii=False))
    print("cost:", json.dumps(s["cost"], ensure_ascii=False))
    return 0


    if "--adversarial" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--adversarial"]
        sys.exit(main_adv())
    sys.exit(main())

# ----------------------------------------------------------------------------- 对抗集（陷阱题）
# 目的：让生成者稳定出错，这样 B1/B2 才有东西可审。
# 每题的陷阱都是"靠推理很容易想当然、但执行结果出乎意料"的地方。
ADVERSARIAL_SPECS: List[Dict[str, str]] = [
    {"id": "A01", "tag": "float-repr",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(0.1 + 0.2)"},
    {"id": "A02", "tag": "round-banker",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(round(2.5), round(3.5), round(2.675, 2))"},
    {"id": "A03", "tag": "nan-identity",
     "question": "下面这段 Python 代码输出什么？（注意三个值的输出顺序）",
     "code": "n = float('nan')\nprint(n == n, n != n, n in [n])"},
    {"id": "A04", "tag": "dict-mutation",
     "question": "下面这段 Python 代码输出什么？",
     "code": "d = {'a': 1, 'b': 2}\nfor k in list(d):\n    if k == 'a':\n        d['c'] = 3\nprint(d)"},
    {"id": "A05", "tag": "chain-comparison",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(1 < 2 < 3, (1 < 2) < 3, 3 > 2 > 1)"},
    {"id": "A06", "tag": "float-div-zero",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(1.0/0.0, 0.0/0.0)"},
    {"id": "A07", "tag": "is-identity-int",
     "question": "下面这段 Python 代码输出什么？（按语言规范推理，不要想当然）",
     "code": "a = 1000\nb = 1000\nc = 256\nd = 256\nprint(a is b, c is d)"},
    {"id": "A08", "tag": "unpack-star",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a, *b, c = [1, 2, 3, 4]\nprint(a, b, c)"},
    {"id": "A09", "tag": "mutate-during-iter",
     "question": "下面这段 Python 代码输出什么？",
     "code": "L = [1, 2, 3]\nfor x in L:\n    if x == 1:\n        L.append(9)\n    elif x == 2:\n        break\nprint(L)"},
    {"id": "A10", "tag": "repr-escape",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(repr('a\tb'), len('\\t'))"},
    {"id": "A11", "tag": "except-order",
     "question": "下面这段 Python 代码输出什么？",
     "code": "try:\n    raise ValueError('v')\nexcept LookupError:\n    print('lookup')\nexcept Exception:\n    print('exception')"},
    {"id": "A12", "tag": "str-conversion",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(int('12') + 3, float('1.5') * 2, str(1.0))"},
    {"id": "A13", "tag": "sorted-key-stability",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(sorted(['bb', 'a', 'ccc'], key=len))"},
    {"id": "A14", "tag": "default-arg-eval-order",
     "question": "下面这段 Python 代码输出什么？",
     "code": "def h(n, acc=0):\n    return acc + n\nprint(h(1), h(2, 10))"},

    # ---------------------------------------------------------------------
    # 扩充集（2026-10-27 追加，A15-A38）
    # 动机：原 14 题的判别样本只有 2 个，McNemar p=0.500，统计上不可能显著。
    # 取样规则（预先固定，避免事后挑题）：先从 115 道候选里做生成者筛选，
    # 再按四个任务族各取 6 道，族内按候选编号顺序取前 6 个 stdout 非空且确定的题。
    #   realistic-program  10-25 行真实程序（类/生成器/描述符/上下文管理器…）
    #   semantics-trap     短小但反直觉的语义点
    #   identity-constant  依赖 CPython 实现或只能由执行确定的常量
    #   simulation         必须手工模拟几十步以上才能得出答案
    # 筛选结论：四个族合计 115 题里生成者真实错误为 0，因此本扩充集的目的
    # 不是"凑出显著"，而是把"补样本量无效"变成可引用的实测证据。
    # ---------------------------------------------------------------------
    # realistic-program（候选 C01-C06）
    {"id": "A15", "family": "realistic-program", "tag": "metaclass-attr",
     "question": "下面这段 Python 代码输出什么？",
     "code": "class Meta(type):\n    def __new__(mcls, name, bases, ns):\n        ns['tag'] = name.lower()\n        return super().__new__(mcls, name, bases, ns)\nclass A(metaclass=Meta):\n    pass\nclass B(A):\n    pass\nprint(A.tag, B.tag, B.__mro__[1].__name__)"},
    {"id": "A16", "family": "realistic-program", "tag": "gen-send",
     "question": "下面这段 Python 代码输出什么？",
     "code": "def g():\n    x = yield 1\n    try:\n        yield x * 2\n    except GeneratorExit:\n        print('exit')\nit = g()\nprint(next(it), it.send(5))"},
    {"id": "A17", "family": "realistic-program", "tag": "descriptor",
     "question": "下面这段 Python 代码输出什么？",
     "code": "class D:\n    def __get__(self, obj, owner):\n        return 42\nclass C:\n    d = D()\nprint(C().d, C.d)"},
    {"id": "A18", "family": "realistic-program", "tag": "slots-inherit",
     "question": "下面这段 Python 代码输出什么？",
     "code": "class P:\n    __slots__ = ('a',)\n    def __init__(self, a): self.a = a\nclass Q(P):\n    __slots__ = ('b',)\n    def __init__(self, a, b):\n        super().__init__(a)\n        self.b = b\nq = Q(1, 2)\nprint(q.a, q.b, hasattr(q, '__dict__'))"},
    {"id": "A19", "family": "realistic-program", "tag": "exc-cause-context",
     "question": "下面这段 Python 代码输出什么？",
     "code": "try:\n    try:\n        raise ValueError('inner')\n    except ValueError as e:\n        raise TypeError('outer') from e\nexcept TypeError as e:\n    print(type(e.__cause__).__name__, type(e.__context__).__name__)"},
    {"id": "A20", "family": "realistic-program", "tag": "reduce-operator",
     "question": "下面这段 Python 代码输出什么？",
     "code": "from functools import reduce\nimport operator\nprint(reduce(operator.add, [1,2,3], 10), reduce(operator.mul, range(1,5)))"},
    # semantics-trap（候选 B13-B19 中 stdout 非空者）
    {"id": "A21", "family": "semantics-trap", "tag": "round-half-even",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(round(0.5), round(1.5), round(2.5), round(-0.5))"},
    {"id": "A22", "family": "semantics-trap", "tag": "round-float-repr",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(round(2.675, 2), round(1.005, 2))"},
    {"id": "A23", "family": "semantics-trap", "tag": "sort-stable-key",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(sorted([(1,'b'),(1,'a'),(0,'z')], key=lambda p: p[0]))"},
    {"id": "A24", "family": "semantics-trap", "tag": "any-all-empty",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(any([]), all([]), any([0,'']), all([0,'']))"},
    {"id": "A25", "family": "semantics-trap", "tag": "repr-dict-circ",
     "question": "下面这段 Python 代码输出什么？",
     "code": "d={}\nd['self']=d\nprint(d)"},
    {"id": "A26", "family": "semantics-trap", "tag": "int-float-str",
     "question": "下面这段 Python 代码输出什么？",
     "code": "print(int('12'), int('  12  '), int('1_0'))"},
    # identity-constant（候选 E01-E06）
    {"id": "A27", "family": "identity-constant", "tag": "int-257-same-object",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = 257\nb = 257\nprint(a is b)"},
    {"id": "A28", "family": "identity-constant", "tag": "int-257-cross-stmt",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = 257\nprint(a is 257)"},
    {"id": "A29", "family": "identity-constant", "tag": "str-concat-fold",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = 'hel' + 'lo'\nb = 'hello'\nprint(a is b, a == b)"},
    {"id": "A30", "family": "identity-constant", "tag": "str-mul-fold",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = 'ab' * 2\nb = 'abab'\nprint(a is b)"},
    {"id": "A31", "family": "identity-constant", "tag": "float-const-fold",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = 1.5\nb = 1.5\nprint(a is b)"},
    {"id": "A32", "family": "identity-constant", "tag": "tuple-const-fold",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = (1, 2)\nb = (1, 2)\nprint(a is b)"},
    # simulation（候选 S01-S06，需手工模拟数十步）
    {"id": "A33", "family": "simulation", "tag": "collatz-1234567",
     "question": "下面这段 Python 代码输出什么？",
     "code": "n = 1234567\nc = 0\nwhile n > 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    c += 1\nprint(n, c)"},
    {"id": "A34", "family": "simulation", "tag": "collatz-27",
     "question": "下面这段 Python 代码输出什么？",
     "code": "n = 27\nc = 0\nwhile n > 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    c += 1\nprint(n, c)"},
    {"id": "A35", "family": "simulation", "tag": "float-chain-20",
     "question": "下面这段 Python 代码输出什么？",
     "code": "x = 1.0\nfor i in range(20):\n    x = x * 1.05 - 0.3\nprint(round(x, 6))"},
    {"id": "A36", "family": "simulation", "tag": "float-accum-50",
     "question": "下面这段 Python 代码输出什么？",
     "code": "a = 0.0\nfor i in range(50):\n    a += 0.1\nprint(round(a, 10))"},
    {"id": "A37", "family": "simulation", "tag": "round-div-chain",
     "question": "下面这段 Python 代码输出什么？",
     "code": "x = 100.0\nfor i in range(12):\n    x = round(x / 3.0, 4)\nprint(x)"},
    {"id": "A38", "family": "simulation", "tag": "sin-cos-sum",
     "question": "下面这段 Python 代码输出什么？",
     "code": "import math\nt = 0.0\nfor i in range(1, 41):\n    t += math.sin(i) * math.cos(i / 2)\nprint(round(t, 6))"},
]


def build_adversarial_tasks() -> Dict[str, Any]:
    tasks = []
    for spec in ADVERSARIAL_SPECS:
        r = run_code(spec["code"])
        tasks.append({
            "id": spec["id"], "tag": spec["tag"], "question": spec["question"], "code": spec["code"],
            # family 记录这道题属于哪个取样族；A01-A14 是 2026-10-26 的原始陷阱集，
            # A15-A38 是 2026-10-27 按四族等量追加的扩充集。汇总时可按族拆分。
            "family": spec.get("family", "original-trap"),
            "ground_truth_stdout": r["stdout"].rstrip("\n"),
            "ground_truth_returncode": r["returncode"],
            "exec_stderr": r["stderr"].rstrip("\n"),
            "exec_wall_s": r["wall_s"],
        })
    obj = {
        "built_at": core.now_iso(),
        "executed_by": PY,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "n_tasks": len(tasks),
        "note": "ground truth 由本机真实执行 stdlib Python 得到，不是模型自述。",
        "tasks": tasks,
    }
    core.write_json(TASKS_ADV_PATH, obj)
    return obj


def build_adversarial_tasks_entry() -> Dict[str, Any]:
    core.ensure_dir(EXP_DIR_ADV)
    return build_adversarial_tasks()

# ----------------------------------------------------------------------------- 通用驱动（支持循环重复）
def drive(client: core.ResponsesClient, tasks: List[Dict[str, Any]],
          paths: Dict[str, str], py_ver: str, stages: List[str],
          max_output_tokens: int = 3000, retry_incomplete: bool = False) -> None:
    """按任务集驱动 G / B1 / B2 三阶段，结果落盘（幂等：已有条目跳过）。"""
    gens = core.load_json(paths["gen"]) if os.path.exists(paths["gen"]) else []
    b1 = core.load_json(paths["b1"]) if os.path.exists(paths["b1"]) else []
    b2 = core.load_json(paths["b2"]) if os.path.exists(paths["b2"]) else []

    if retry_incomplete:
        # 修补路径：响应不完整（没产出答案）的记录不是数据，是 harness 预算歧视的产物。
        # 删掉它们让下面的 todo 重新包含这些 task_id，用更大的输出预算重跑。
        # 只删"没有答案"的，不动任何已产出答案的记录——避免变成挑选样本。
        def drop_no_answer(rows, key, path):
            keep = [r for r in rows if call_succeeded(r) and answer_was_produced(r, key)]
            dropped = len(rows) - len(keep)
            if dropped:
                print("[repair] %s: dropping %d no-answer records for retry" % (
                    os.path.basename(path), dropped), flush=True)
                core.write_json(path, keep)
            return keep

        gens = drop_no_answer(gens, "answer", paths["gen"])
        b1 = drop_no_answer(b1, "final_answer", paths["b1"])
        b2 = drop_no_answer(b2, "final_answer", paths["b2"])

    have_gen = {g["task_id"] for g in gens}
    todo = [t for t in tasks if t["id"] not in have_gen]
    if "G" in stages and todo:
        print("[stage G] generating for %d tasks" % len(todo), flush=True)
        gens = gens + run_generation(client, todo, max_output_tokens)
        gens.sort(key=lambda x: x["task_id"])
        core.write_json(paths["gen"], gens)
    for arm, key in (("B1", "b1"), ("B2", "b2")):
        if arm not in stages:
            continue
        cur = core.load_json(paths[key]) if os.path.exists(paths[key]) else []
        have = {r["task_id"] for r in cur}
        todo = [t for t in tasks if t["id"] not in have]
        if not todo:
            continue
        print("[stage %s] reviewing %d tasks" % (arm, len(todo)), flush=True)
        cur = cur + run_review(client, todo, gens, arm, py_ver, max_output_tokens)
        cur.sort(key=lambda x: x["task_id"])
        core.write_json(paths[key], cur)


def main_adv() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-tasks", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="同一任务集重复跑几轮（结果合并，形成多轮样本）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stages", default="G,B1,B2")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--model", default=core.MODEL)
    ap.add_argument("--tag-suffix", action="store_true",
                    help="重复轮次时给 task_id 加后缀，保证 summary 不做跨轮去重")
    ap.add_argument("--max-output-tokens", type=int, default=3000,
                    help="单次调用输出预算；2026-10-27 由 900 提到 3000，因为 effort=max 的推理会吃掉全部预算，"
                         "导致 status=incomplete、没有答案文本，字面匹配会把这种记录误判成答错")
    ap.add_argument("--retry-incomplete", action="store_true",
                    help="把响应不完整、没有产出答案的记录从结果文件删除并重跑；"
                         "这是消除 harness 预算歧视的修补，不是挑选样本")
    args = ap.parse_args()

    core.ensure_dir(EXP_DIR_ADV)
    if args.build_tasks or not os.path.exists(TASKS_ADV_PATH):
        tobj = build_adversarial_tasks()
        print("[adv tasks] built %d" % tobj["n_tasks"])
    tobj = core.load_json(TASKS_ADV_PATH)
    tasks = tobj["tasks"]
    if args.limit:
        tasks = tasks[:args.limit]
    py_ver = tobj.get("python_version", "?")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    if not args.summarize_only:
        client = core.ResponsesClient(model=args.model)
        for rep in range(1, args.repeats + 1):
            if rep == 1:
                round_tasks = tasks
                paths = {"gen": GEN_ADV_PATH, "b1": B1_ADV_PATH, "b2": B2_ADV_PATH}
            else:
                round_tasks = [dict(t, id="%s__r%d" % (t["id"], rep)) for t in tasks]
                paths = {
                    "gen": GEN_ADV_PATH.replace(".json", "_r%d.json" % rep),
                    "b1": B1_ADV_PATH.replace(".json", "_r%d.json" % rep),
                    "b2": B2_ADV_PATH.replace(".json", "_r%d.json" % rep),
                }
            print("=== adversarial round %d (%d tasks) ===" % (rep, len(round_tasks)), flush=True)
            drive(client, round_tasks, paths, py_ver, stages,
                      max_output_tokens=args.max_output_tokens,
                      retry_incomplete=args.retry_incomplete)
        print("[client stats]", json.dumps(client.stats(), ensure_ascii=False))

    # 汇总：把所有轮次合并
    all_gens, all_b1, all_b2, all_tasks = [], [], [], []
    files = [("gen", GEN_ADV_PATH), ("b1", B1_ADV_PATH), ("b2", B2_ADV_PATH)]
    def merge_round(target: List[Dict[str, Any]], base: str) -> None:
        for p in sorted([x for x in os.listdir(EXP_DIR_ADV) if x.startswith(os.path.basename(base).replace(".json", ""))]):
            if not p.endswith(".json"):
                continue
            fp = os.path.join(EXP_DIR_ADV, p)
            for row in core.load_json(fp):
                row["_file"] = p
                target.append(row)
    merge_round(all_gens, GEN_ADV_PATH)
    merge_round(all_b1, B1_ADV_PATH)
    merge_round(all_b2, B2_ADV_PATH)
    if not (all_gens and all_b1 and all_b2):
        print("incomplete", file=sys.stderr)
        return 1
    ids = sorted({g["task_id"] for g in all_gens} & {r["task_id"] for r in all_b1}
                 & {r["task_id"] for r in all_b2})
    all_tasks = [dict(x) for x in tasks for _ in [0]]
    all_tasks = tasks + [dict(t, id="%s__r%d" % (t["id"], rep)) for rep in range(2, args.repeats + 1) for t in tasks]
    all_tasks = [t for t in all_tasks if t["id"] in ids]
    g_s = sorted([g for g in all_gens if g["task_id"] in ids], key=lambda x: x["task_id"])
    b1_s = sorted([r for r in all_b1 if r["task_id"] in ids], key=lambda x: x["task_id"])
    b2_s = sorted([r for r in all_b2 if r["task_id"] in ids], key=lambda x: x["task_id"])
    s = summarize(all_tasks, g_s, b1_s, b2_s)
    s["task_set"] = {"path": TASKS_ADV_PATH, "n_tasks": len(all_tasks),
                     "n_distinct_tasks": len(tasks),
                     "ground_truth_note": tobj["note"], "built_at": tobj["built_at"],
                     "repeats": args.repeats}
    # 按取样族拆分：扩充前（original-trap）与扩充后四族分别给一次 paired 统计，
    # 这样每一族贡献了多少判别样本、生成者错误率是多少都看得见。
    fam_of = {t['id']: t.get('family', 'original-trap') for t in tasks}
    by_family = {}
    for fam in sorted(set(fam_of.values())):
        tids = [t['id'] for t in all_tasks if fam_of.get(t['id'].split('__')[0]) == fam]
        if not tids:
            continue
        fg = [g for g in g_s if g['task_id'] in tids]
        f1 = [r for r in b1_s if r['task_id'] in tids]
        f2 = [r for r in b2_s if r['task_id'] in tids]
        if not (fg and f1 and f2):
            continue
        fs = summarize([t for t in all_tasks if t['id'] in tids], fg, f1, f2)
        by_family[fam] = {
            "n_pairs": fs["paired"]["n_pairs"],
            "generator_accuracy": fs["generator"]["accuracy"],
            "B1_final_accuracy": fs["arm_B1_no_new_info"]["final_accuracy"],
            "B2_final_accuracy": fs["arm_B2_execution_evidence"]["final_accuracy"],
            "only_B2_correct": fs["paired"]["n_only_B2_correct"],
            "only_B1_correct": fs["paired"]["n_only_B1_correct"],
            "mcnemar_exact_p_two_sided": fs["paired"]["mcnemar_exact_p_two_sided"],
        }
    s["by_family"] = by_family
    # 覆盖前先归档上一版 summary，保证 2026-10-26 那轮（14 题 2 轮）的数字仍可复核。
    prev = os.path.join(EXP_DIR_ADV, 'summary.json')
    if os.path.exists(prev):
        stamp = core.now_iso().replace(':', '').replace('-', '')
        arch = os.path.join(EXP_DIR_ADV, 'summary.archive-%s.json' % stamp[:15])
        core.write_json(arch, core.load_json(prev))
        print("[archive] previous summary -> %s" % os.path.basename(arch))
    core.write_json(prev, s)
    print("\n=== adversarial summary ===")
    print("generator acc = %s (%d/%d)" % (s["generator"]["accuracy"], s["generator"]["n_correct"], s["generator"]["n"]))
    for k in ("arm_B1_no_new_info", "arm_B2_execution_evidence"):
        d = s[k]
        print("%s: change_rate=%s acc=%s fixed=%d broke=%d n=%d tokens=%d" % (
            k, d["change_rate"], d["final_accuracy"], d["n_fixed_wrong_to_right"],
            d["n_broke_right_to_wrong"], d["n"], d["total_tokens"]))
    print("paired:", json.dumps(s["paired"], ensure_ascii=False))
    return 0

# ----------------------------------------------------------------------------- 确定性重判
# 为什么需要重判：
#  1) 在线阶段的 answer_matches 是"字面精确匹配"，会把"A06 用一句解释代替空输出"误判成错；
#  2) 在线阶段把 changed 定义成"最终答案 != 生成者答案"，而模型常把答案写进解释里，
#     于是出现 verdict=agree 但 changed=True 的假修正。
# 本函数只读已落盘的原始 JSON，用固定规则重算，任何人重跑都应得到同一结果。

def rescore(base_dir: str, out_path: str, tasks_path: str) -> Dict[str, Any]:
    tobj = core.load_json(tasks_path)
    truth = {t["id"]: t for t in tobj["tasks"]}

    def load(base: str) -> List[Dict[str, Any]]:
        out = []
        for fn in sorted(os.listdir(base_dir)):
            if not fn.endswith(".json"):
                continue
            stem = fn[:-5]
            if stem == base or re.fullmatch(re.escape(base) + r"_r\d+", stem):
                for row in core.load_json(os.path.join(base_dir, fn)):
                    row["_file"] = fn
                    out.append(row)
        return out

    gens = load("generation")
    b1 = load("review_B1")
    b2 = load("review_B2")
    base_id = lambda s: s.split("__")[0]

    def sem_ok(task_id: str, answer: Any) -> bool:
        tid = base_id(task_id)
        fa = answer or ""
        if tid == "A06":
            # 真值：stdout 为空 + ZeroDivisionError（返回码 1）。解释性作答也算语义正确。
            return ("ZeroDivisionError" in fa) or ("空" in fa) or (norm(fa) == "")
        return answer_matches(answer, truth.get(tid, {}).get("ground_truth_stdout", ""))

    def exact_ok(task_id: str, answer: Any) -> bool:
        return answer_matches(answer, truth.get(base_id(task_id), {}).get("ground_truth_stdout", ""))

    def stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        m = len(rows)
        text_changed = revise = f_exact = f_sem = fixed = broke = 0
        per = []
        for r in rows:
            gen_ans, fa = r.get("generator_answer"), r.get("final_answer")
            g_sem, f_s = sem_ok(r["task_id"], gen_ans), sem_ok(r["task_id"], fa)
            ch = norm(fa) != norm(gen_ans)
            text_changed += ch
            revise += 1 if (r.get("verdict") or "").lower() == "revise" else 0
            f_exact += 1 if exact_ok(r["task_id"], fa) else 0
            f_sem += 1 if f_s else 0
            if (not g_sem) and f_s:
                fixed += 1
            if g_sem and (not f_s):
                broke += 1
            per.append({"task_id": r["task_id"], "generator_answer": gen_ans,
                        "generator_sem_ok": g_sem, "verdict": r.get("verdict"), "final_answer": fa,
                        "final_exact_ok": exact_ok(r["task_id"], fa), "final_sem_ok": f_s,
                        "answer_text_changed": bool(ch), "file": r.get("_file")})
        return {
            "n": m,
            "n_answer_text_changed": text_changed,
            "answer_text_change_rate": round(text_changed / m, 4) if m else None,
            "n_verdict_revise": revise,
            "verdict_revise_rate": round(revise / m, 4) if m else None,
            "n_final_exact_correct": f_exact,
            "exact_accuracy": round(f_exact / m, 4) if m else None,
            "n_final_semantically_correct": f_sem,
            "semantic_accuracy": round(f_sem / m, 4) if m else None,
            "n_fixed_wrong_to_right": fixed,
            "n_broke_right_to_wrong": broke,
            "per_task": per,
        }

    gen_rows = [{"task_id": g["task_id"], "generator_answer": g.get("answer"),
                 "final_answer": g.get("answer"), "verdict": None} for g in gens]
    s1, s2 = stats(b1), stats(b2)
    pairs = {}
    for a, b in zip(sorted(b1, key=lambda x: x["task_id"]), sorted(b2, key=lambda x: x["task_id"])):
        if base_id(a["task_id"]) != base_id(b["task_id"]):
            continue
        pairs[a["task_id"]] = (sem_ok(a["task_id"], a.get("final_answer")),
                               sem_ok(b["task_id"], b.get("final_answer")))
    vals = list(pairs.values())
    both = sum(1 for v in vals if v[0] and v[1])
    only2 = sum(1 for v in vals if (not v[0]) and v[1])
    only1 = sum(1 for v in vals if v[0] and (not v[1]))
    neither = sum(1 for v in vals if (not v[0]) and (not v[1]))

    obj = {
        "rescored_at": core.now_iso(),
        "source_dir": base_dir,
        "n_pairs": len(pairs),
        "generator": stats(gen_rows),
        "arm_B1_no_new_info": s1,
        "arm_B2_execution_evidence": s2,
        "paired_semantic": {
            "both_correct": both, "only_B2_correct": only2, "only_B1_correct": only1,
            "neither_correct": neither,
            "mcnemar_exact_p_two_sided": binom_exact_p(only2, only2 + only1),
            "semantic_accuracy_delta_B2_minus_B1": round(
                (s2["semantic_accuracy"] or 0) - (s1["semantic_accuracy"] or 0), 4),
            "text_change_rate_delta_B2_minus_B1": round(
                (s2["answer_text_change_rate"] or 0) - (s1["answer_text_change_rate"] or 0), 4),
        },
        "rescoring_note": (
            "exact_accuracy = 最终答案与真实执行 stdout 规范化后逐字相等（规范化只做首尾空白/引号/空格归一）；"
            "semantic_accuracy = 在 exact 基础上，对 A06（真值为空 stdout + ZeroDivisionError）把含 "
            "ZeroDivisionError 或'空'的作答判为语义正确；answer_text_changed = 最终答案文本与生成者答案文本不同；"
            "fixed/broke 按 semantic 口径配对统计。在线阶段 summary.json 用的是字面口径，两者差异见笔记。"),
    }
    core.write_json(out_path, obj)
    return obj


def main_rescore() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="both", choices=["main", "adv", "both"])
    args = ap.parse_args()
    sets = []
    if args.set in ("main", "both"):
        sets.append((EXP_DIR, TASKS_PATH))
    if args.set in ("adv", "both"):
        sets.append((EXP_DIR_ADV, TASKS_ADV_PATH))
    for d, tp in sets:
        if not os.path.exists(tp):
            print("missing tasks:", tp)
            continue
        obj = rescore(d, os.path.join(d, "rescore.json"), tp)
        print("=== rescored", d)
        print("  generator  exact=%s semantic=%s" % (obj["generator"]["exact_accuracy"],
                                                    obj["generator"]["semantic_accuracy"]))
        for k in ("arm_B1_no_new_info", "arm_B2_execution_evidence"):
            a = obj[k]
            print("  %s n=%d text_changed=%s revise=%s exact=%s semantic=%s fixed=%d broke=%d" % (
                k, a["n"], a["answer_text_change_rate"], a["verdict_revise_rate"],
                a["exact_accuracy"], a["semantic_accuracy"],
                a["n_fixed_wrong_to_right"], a["n_broke_right_to_wrong"]))
        print("  paired:", json.dumps(obj["paired_semantic"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if "--rescore" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--rescore"]
        sys.exit(main_rescore())
    if "--adversarial" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--adversarial"]
        sys.exit(main_adv())
    sys.exit(main())