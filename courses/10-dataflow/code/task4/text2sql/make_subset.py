#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从官方 pipeline_refine.jsonl 截取前 N 条，生成任务用子集。"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    subset = rows[args.offset: args.offset + args.num]
    with open(args.output, "w", encoding="utf-8") as fout:
        for row in subset:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"input rows={len(rows)}, subset rows={len(subset)} -> {args.output}")


if __name__ == "__main__":
    main()
