#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task6 — 评测数据集构建（官方 example 数据原样落盘 + 完整性核对）

数据来源：已安装的 open-dataflow 1.0.11 包内
  dataflow/example/core_text_data/unified_bench_eval_type{1..6}.jsonl
与本地仓库 DataFlow-main/dataflow/example/core_text_data/ 同源（sha256 核对一致）。

同时把官方 7 个 pipeline 原型（type1-6 + semantic 变体）拷到 data/official_pipelines/，
保证"跑的就是官方管线"可逐字对照。
"""
import hashlib
import json
import os
import shutil
import sys

import dataflow

TYPES = [
    ("key1_text_score", "unified_bench_eval_type1.jsonl", "unified_bench_eval_type1.py"),
    ("key2_qa", "unified_bench_eval_type2.jsonl", "unified_bench_eval_type2.py"),
    ("key2_q_ma", "unified_bench_eval_type3.jsonl", "unified_bench_eval_type3.py"),
    ("key3_q_choices_a", "unified_bench_eval_type4.jsonl", "unified_bench_eval_type4.py"),
    ("key3_q_choices_as", "unified_bench_eval_type5.jsonl", "unified_bench_eval_type5.py"),
    ("key3_q_a_rejected", "unified_bench_eval_type6.jsonl", "unified_bench_eval_type6.py"),
]
SEMANTIC_PIPELINE = "unified_bench_eval_type_semantic.py"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pkg_dir = os.path.dirname(dataflow.__file__)
    src_dir = os.path.join(pkg_dir, "example", "core_text_data")
    out_dir = os.environ.get("T6_DATA") or "./data"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "official_pipelines"), exist_ok=True)

    manifest = {"source_pkg": pkg_dir, "source_dir": src_dir, "datasets": {}, "pipelines": []}

    for eval_type, jsonl_name, py_name in TYPES:
        src = os.path.join(src_dir, jsonl_name)
        dst = os.path.join(out_dir, jsonl_name)
        shutil.copyfile(src, dst)
        rows = []
        with open(dst, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        fields = sorted({k for r in rows for k in r.keys()})
        manifest["datasets"][eval_type] = {
            "file": jsonl_name,
            "rows": len(rows),
            "fields": fields,
            "sha256": sha256(dst),
            "bytes": os.path.getsize(dst),
        }
        print(f"[DATA] {eval_type:20s} {jsonl_name:36s} rows={len(rows):3d} fields={fields}")

    # 官方 pipeline 原型（含 semantic 变体）
    pipe_src = os.path.join(pkg_dir, "statics", "pipelines", "gpu_pipelines", "benchmark_eval")
    for _, _, py_name in TYPES + [(None, None, SEMANTIC_PIPELINE)]:
        src = os.path.join(pipe_src, py_name)
        if not os.path.exists(src):
            print(f"[WARN] official pipeline not found: {src}", file=sys.stderr)
            continue
        dst = os.path.join(out_dir, "official_pipelines", py_name)
        shutil.copyfile(src, dst)
        manifest["pipelines"].append({"file": py_name, "sha256": sha256(dst)})
        print(f"[PIPE] {py_name}")

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[DONE] manifest -> {os.path.join(out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
