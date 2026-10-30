#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AgenticRAG 输出字段完整性 + F1 分布 + 空值率检查，产出 JSON 报告与抽样。"""
import json
import os
import sys

EXPECTED = [
    "question", "answer", "refined_answer", "optional_answer", "llm_answer",
    "golden_doc_answer", "identifier", "candidate_tasks_str", "llm_score",
    "golden_doc_score", "F1Score",
]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_empty(v):
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def main():
    step1 = sys.argv[1] if len(sys.argv) > 1 else "./cache/agentic_rag_task4_step1.jsonl"
    final = sys.argv[2] if len(sys.argv) > 2 else "./cache/agentic_rag_task4_step2.jsonl"
    out_json = sys.argv[3] if len(sys.argv) > 3 else "./field_check.json"

    src = sys.argv[4] if len(sys.argv) > 4 else ""
    rows1 = load(step1) if os.path.exists(step1) else []
    rows2 = load(final) if os.path.exists(final) else []

    report = {
        "input_file": src,
        "step1_cache": step1,
        "step1_rows": len(rows1),
        "final_cache": final,
        "final_rows": len(rows2),
        "step1_columns": list(rows1[0].keys()) if rows1 else [],
        "final_columns": list(rows2[0].keys()) if rows2 else [],
    }

    # 字段完整性（以最终输出为准）
    cols = report["final_columns"]
    field_check = {}
    for k in EXPECTED:
        field_check[k] = "present" if k in cols else "MISSING"
    report["expected_fields"] = EXPECTED
    report["field_presence"] = field_check
    present = sum(1 for k in EXPECTED if k in cols)
    report["field_completeness"] = {
        "present": present,
        "expected": len(EXPECTED),
        "ratio": round(present / float(len(EXPECTED)), 4),
        "rate": "%.1f%%" % (100.0 * present / len(EXPECTED)),
    }

    # 空值率
    null_rate = {}
    for k in EXPECTED:
        if k not in cols:
            continue
        empties = sum(1 for r in rows2 if is_empty(r.get(k)))
        null_rate[k] = {
            "empty": empties,
            "total": len(rows2),
            "rate": round(empties / float(len(rows2)), 4) if rows2 else None,
        }
    report["empty_rate"] = null_rate

    # F1 分布
    f1s = []
    for r in rows2:
        v = r.get("F1Score")
        try:
            f1s.append(float(v))
        except (TypeError, ValueError):
            f1s.append(0.0)
    if f1s:
        s = sorted(f1s)
        n = len(s)
        report["f1_stats"] = {
            "n": n,
            "mean": round(sum(s) / n, 4),
            "min": round(s[0], 4),
            "p25": round(s[n // 4], 4),
            "median": round(s[n // 2], 4),
            "p75": round(s[3 * n // 4], 4),
            "max": round(s[-1], 4),
            "n_eq_0": sum(1 for x in s if x == 0.0),
            "n_ge_0_5": sum(1 for x in s if x >= 0.5),
            "n_eq_1": sum(1 for x in s if x >= 0.9999),
            "histogram_0.2bins": [
                sum(1 for x in s if i * 0.2 <= x < (i + 1) * 0.2) for i in range(5)
            ],
        }
        # llm_score / golden_doc_score 分布
        for k in ("llm_score", "golden_doc_score"):
            vals = []
            for r in rows2:
                try:
                    vals.append(float(r.get(k)))
                except (TypeError, ValueError):
                    pass
            if vals:
                report.setdefault("score_stats", {})[k] = {
                    "n": len(vals),
                    "mean": round(sum(vals) / len(vals), 4),
                    "min": min(vals),
                    "max": max(vals),
                }
    else:
        report["f1_stats"] = {"n": 0}

    # 字段来源映射（便于报告引用）
    report["field_types_sample"] = {
        k: type(rows2[0].get(k)).__name__ for k in cols
    } if rows2 else {}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 抽 1 条完整样例（长度截断但保留结构）
    if rows2:
        sample = rows2[0]
        trimmed = {}
        for k, v in sample.items():
            if isinstance(v, str) and len(v) > 800:
                trimmed[k] = v[:800] + "...<TRUNCATED total=%d chars>" % len(v)
            elif isinstance(v, list):
                trimmed[k] = [
                    (x[:400] + "...<TRUNCATED>" if isinstance(x, str) and len(x) > 400 else x)
                    for x in v
                ]
            else:
                trimmed[k] = v
        sample_path = os.path.splitext(out_json)[0] + "_sample.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
        print("\n[SAMPLE first row] ->", sample_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
