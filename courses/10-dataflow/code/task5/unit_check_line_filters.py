
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task5 — 行类过滤器算子级单元核对（补 pipeline 级证据的盲区）

背景：官方 PT 管线把 3 个 refiner 放在最前面，其中 RemoveExtraSpacesRefiner 用
" ".join(text.split()) 会把换行符一起压掉，导致后面三个"行比例"过滤器
（LineEndWithEllipsisFilter / LineStartWithBulletpointFilter / LineWithJavascriptFilter）
实际看到的是单行文本。本脚本直接在**原始（未过 refiner）文本**上逐个算子核对，
证明这些算子本身按设计工作，并量化 refiner 换行压平对它们的影响。
"""
import json
import os
import sys

from dataflow.operators.general_text import (
    LineEndWithEllipsisFilter,
    LineStartWithBulletpointFilter,
    LineWithJavascriptFilter,
    RemoveExtraSpacesRefiner,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def make_df(texts):
    import pandas as pd
    return pd.DataFrame({"raw_content": texts})


class FakeStorage:
    """最小 storage：算子只读写 dataframe。"""

    def __init__(self, df):
        self.df = df

    def read(self, output_type="dataframe"):
        return self.df

    def write(self, df):
        self.df = df
        return "mem"


def run_filter(op, df):
    st = FakeStorage(df)
    op.run(storage=st, input_key="raw_content")
    return st.df


def main():
    corpus = os.path.join(HERE, "data", "pt_input.jsonl")
    rows = [json.loads(l) for l in open(corpus, encoding="utf-8")]
    by_id = {r["id"]: r["raw_content"] for r in rows}

    cases = {
        "ellipsis_lines": "LineEndWithEllipsisFilter",
        "bullet_lines": "LineStartWithBulletpointFilter",
        "javascript_lines": "LineWithJavascriptFilter",
        "pt_clean_00": "(control)",
    }

    report = {}
    for rid, expect in cases.items():
        text = by_id[rid]
        entry = {"expect_killed_by": expect}
        for fname, op in [
            ("LineEndWithEllipsisFilter", LineEndWithEllipsisFilter(threshold=0.3)),
            ("LineStartWithBulletpointFilter", LineStartWithBulletpointFilter(threshold=0.9)),
            ("LineWithJavascriptFilter", LineWithJavascriptFilter(threshold=3)),
        ]:
            df = run_filter(op, make_df([text]))
            entry[fname] = "drop" if len(df) == 0 else "keep"
        # refiner 之后（换行被压平）再看一次
        refined = run_filter(RemoveExtraSpacesRefiner(), make_df([text]))["raw_content"][0]
        entry["after_refiner"] = {}
        for fname, op in [
            ("LineEndWithEllipsisFilter", LineEndWithEllipsisFilter(threshold=0.3)),
            ("LineStartWithBulletpointFilter", LineStartWithBulletpointFilter(threshold=0.9)),
            ("LineWithJavascriptFilter", LineWithJavascriptFilter(threshold=3)),
        ]:
            df = run_filter(op, make_df([refined]))
            entry["after_refiner"][fname] = "drop" if len(df) == 0 else "keep"
        entry["refiner_changed_newlines"] = ("\n" in text) and ("\n" not in refined)
        report[rid] = entry

    out = os.path.join(HERE, "..", "evidence", "task5", "unit_check_line_filters.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"{'id':20s} {'expect':32s} {'ellipsis':8s} {'bullet':8s} {'javascript':10s} | after_refiner: ellipsis/bullet/js")
    for rid, e in report.items():
        print(f"{rid:20s} {e['expect_killed_by']:32s} "
              f"{e['LineEndWithEllipsisFilter']:8s} {e['LineStartWithBulletpointFilter']:8s} {e['LineWithJavascriptFilter']:10s} | "
              f"{e['after_refiner']['LineEndWithEllipsisFilter']}/{e['after_refiner']['LineStartWithBulletpointFilter']}/{e['after_refiner']['LineWithJavascriptFilter']} "
              f"(newlines_flat={e['refiner_changed_newlines']})")
    print("[written]", os.path.abspath(out))


if __name__ == "__main__":
    main()
