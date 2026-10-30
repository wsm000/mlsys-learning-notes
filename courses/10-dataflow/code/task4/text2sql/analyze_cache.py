#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 DataFlow Text2SQL pipeline 每一步 cache jsonl 的行数/字段/非空数/难度分布，
并从日志中统计 429/ERROR 次数，输出 JSON 摘要 + 最终样本。用于 Task4 证据归档。

用法:
  python analyze_cache.py --cache-dir ./cache_refine --prefix task4_refine \
      --log refine.log --out summary_refine.json --final-step 10
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

KEY_FIELDS = [
    "db_id", "SQL", "question", "evidence", "prompt", "cot_reasoning",
    "sql_variation_type", "sql_complexity_type",
    "sql_component_difficulty", "sql_execution_difficulty",
]


def analyze_steps(cache_dir, prefix, max_step=10):
    steps = []
    for step in range(1, max_step + 1):
        path = os.path.join(cache_dir, f"{prefix}_step{step}.jsonl")
        record = {"step": step, "file": os.path.basename(path), "exists": os.path.isfile(path)}
        if not record["exists"]:
            steps.append(record)
            continue
        try:
            df = pd.read_json(path, lines=True)
        except Exception as exc:  # noqa: BLE001
            record.update({"rows": 0, "read_error": str(exc)})
            steps.append(record)
            continue
        record["rows"] = int(len(df))
        record["columns"] = list(df.columns)
        record["non_null"] = {k: int(df[k].notna().sum()) for k in KEY_FIELDS if k in df.columns}
        for col in ("sql_variation_type", "sql_complexity_type",
                    "sql_component_difficulty", "sql_execution_difficulty", "question_type"):
            if col in df.columns:
                record.setdefault("value_counts", {})[col] = {
                    str(k): int(v) for k, v in df[col].fillna("<NA>").astype(str).value_counts().items()
                }
        if "db_id" in df.columns:
            record["db_id_counts"] = {str(k): int(v) for k, v in df["db_id"].value_counts().items()}
        steps.append(record)
    return steps


def analyze_log(log_path):
    summary = {"path": log_path, "exists": os.path.isfile(log_path)}
    if not summary["exists"]:
        return summary
    with open(log_path, "r", encoding="utf-8", errors="replace") as fin:
        text = fin.read()
    summary["total_lines"] = text.count("\n")
    summary["status_429_count"] = len(re.findall(r"status=429", text))
    summary["error_lines"] = len(re.findall(r"\| ERROR", text))
    summary["rate_limit_exceeded_count"] = len(re.findall(r"rate_limit", text))
    summary["token_rate_limit_count"] = len(re.findall(r"token_rate_limit", text))
    summary["concurrency_limit_count"] = len(re.findall(r"concurrency limit", text))
    summary["model_glm_count"] = len(re.findall(r"GLM-5\.3-Flash", text))
    summary["model_deepseek_count"] = len(re.findall(r"DeepSeek-V4-Flash", text))
    summary["step_markers"] = re.findall(r"\[STEP (\d+)\] rows=(\d+)", text)
    return summary


def build_samples(cache_dir, prefix, final_step, limit=2):
    path = os.path.join(cache_dir, f"{prefix}_step{final_step}.jsonl")
    if not os.path.isfile(path):
        candidates = sorted(
            glob.glob(os.path.join(cache_dir, f"{prefix}_step*.jsonl")),
            key=lambda p: int(re.search(r"_step(\d+)", p).group(1)),
        )
        if not candidates:
            return []
        path = candidates[-1]
    df = pd.read_json(path, lines=True)
    rows = df.head(limit).to_dict(orient="records")
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, list):
                row[key] = [str(v)[:2000] for v in value][:10]
            elif isinstance(value, str) and len(value) > 8000:
                row[key] = value[:8000] + "...[truncated]"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--log", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--final-step", type=int, default=10)
    parser.add_argument("--samples-out", default="")
    parser.add_argument("--samples-num", type=int, default=2)
    args = parser.parse_args()

    payload = {
        "cache_dir": os.path.abspath(args.cache_dir),
        "prefix": args.prefix,
        "steps": analyze_steps(args.cache_dir, args.prefix),
        "log": analyze_log(args.log) if args.log else None,
    }
    with open(args.out, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
    print(f"summary -> {args.out}")

    if args.samples_out:
        samples = build_samples(args.cache_dir, args.prefix, args.final_step, args.samples_num)
        with open(args.samples_out, "w", encoding="utf-8") as fout:
            for sample in samples:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"samples -> {args.samples_out} ({len(samples)} rows)")


if __name__ == "__main__":
    main()
