
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task5 — 核对脚本：把三条 pipeline 的 cache step 文件还原成
「过滤漏斗 + id 级去向 + 分数分布 + 字段完整性」证据，并与设计语料的缺陷清单对照。

输出 verify_task5.json + 控制台表格。不重跑任何模型，只读 cache。
"""
import argparse
import glob
import json
import os
import statistics

PT_OPERATORS = [
    "RemoveEmojiRefiner", "HtmlUrlRemoverRefiner", "RemoveExtraSpacesRefiner",
    "BlocklistFilter", "WordNumberFilter", "ColonEndFilter", "SentenceNumberFilter",
    "LineEndWithEllipsisFilter", "ContentNullFilter", "MeanWordLengthFilter",
    "SymbolWordRatioFilter", "HtmlEntityFilter", "NoPuncFilter", "SpecialCharacterFilter",
    "WatermarkFilter", "CurlyBracketFilter", "CapitalWordsFilter", "LoremIpsumFilter",
    "UniqueWordsFilter", "CharNumberFilter", "LineStartWithBulletpointFilter",
    "LineWithJavascriptFilter", "PairQualFilter", "LanguageFilter", "MinHashDeduplicateFilter",
]

SFT_OPERATORS = ["WordNumberFilter", "SuperfilteringFilter", "DeitaQualityFilter"]
SYN_OPERATORS = ["CondorGenerator", "CondorRefiner", "AlpagasusFilter"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_steps(cache, prefix):
    """返回 {step_no: rows}；step0 是输入文件（由调用方单独给）。"""
    steps = {}
    for p in sorted(glob.glob(os.path.join(cache, f"{prefix}_step*.jsonl"))):
        name = os.path.basename(p)
        try:
            n = int(name.split("_step")[1].split(".")[0])
        except Exception:
            continue
        steps[n] = read_jsonl(p)
    return steps


def ids_of(rows):
    return [r.get("id") for r in rows]


def funnel(steps, operators, input_rows):
    """逐算子行数漏斗 + id 级 removed_by 归属。"""
    counts = [("input", len(input_rows))]
    killed_by = {}
    prev_ids = set(ids_of(input_rows))
    for n in sorted(steps):
        op = operators[n - 1] if n - 1 < len(operators) else f"step{n}"
        cur_ids = set(ids_of(steps[n]))
        counts.append((op, len(cur_ids)))
        for i in prev_ids - cur_ids:
            if i not in killed_by:
                killed_by[i] = op
        prev_ids = cur_ids
    return counts, killed_by, prev_ids


def col_stats(rows, col):
    vals = [r[col] for r in rows if col in r and isinstance(r[col], (int, float))]
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "mean": round(statistics.fmean(vals), 4),
        "median": round(statistics.median(vals), 4),
    }


def verify_pt(args, report):
    cache, prefix = args.pt_cache, args.pt_prefix
    if not os.path.isdir(cache):
        report["pt"] = {"status": "skipped (no cache)"}
        return
    input_rows = read_jsonl(args.pt_input)
    steps = load_steps(cache, prefix)
    counts, killed_by, survivors = funnel(steps, PT_OPERATORS, input_rows)

    manifest = {m["id"]: m for m in json.load(open(args.manifest_pt, encoding="utf-8"))}
    id_rows = []
    for m in manifest.values():
        rid = m["id"]
        killed = killed_by.get(rid)
        actual = "keep" if rid in survivors else "drop"
        expect = m["expect"]
        ok = (expect == "keep") == (actual == "keep")
        id_rows.append({
            "id": rid, "defect": m["defect"], "expect": expect,
            "actual": actual, "killed_by": killed, "match": ok,
        })

    # PairQual 分数分布（PairQualFilter 那一步的文件带 PairQualScore 列）
    pq_idx = PT_OPERATORS.index("PairQualFilter") + 1
    pq_stats = None
    pq_rows = []
    if pq_idx in steps:
        pq_stats = col_stats(steps[pq_idx], "PairQualScore")
        pq_rows = [{"id": r.get("id"), "PairQualScore": r.get("PairQualScore")}
                   for r in steps[pq_idx]]
    # refiner 效果：emoji / url / extra spaces 三篇在 step0 与 step3 的内容差异
    refiner_effect = {}
    if 3 in steps:
        s3 = {r.get("id"): r.get("raw_content", "") for r in steps[3]}
        for rid in ["emoji_doc", "html_url_doc", "extra_spaces_doc"]:
            src = next((r for r in input_rows if r.get("id") == rid), None)
            if src:
                refiner_effect[rid] = {
                    "before_tail": src["raw_content"][-60:],
                    "after_tail": s3.get(rid, "")[-60:],
                    "changed": src["raw_content"] != s3.get(rid, ""),
                }

    report["pt"] = {
        "status": "ok",
        "input_rows": len(input_rows),
        "steps": len(steps),
        "survivors": len(survivors),
        "funnel": [{"operator": op, "rows": n} for op, n in counts],
        "killed_by": killed_by,
        "survivor_ids": sorted(survivors),
        "id_rows": id_rows,
        "manifest_match": {
            "total": len(id_rows),
            "matched": sum(1 for r in id_rows if r["match"]),
            "mismatched": [r["id"] for r in id_rows if not r["match"]],
        },
        "pairqual_score_stats": pq_stats,
        "pairqual_scores": pq_rows,
        "refiner_effect": refiner_effect,
    }


def verify_sft(args, report):
    cache, prefix = args.sft_cache, args.sft_prefix
    if not os.path.isdir(cache):
        report["sft"] = {"status": "skipped (no cache)"}
        return
    input_rows = read_jsonl(args.sft_input)
    steps = load_steps(cache, prefix)
    # 第三步可能是官方 DeitaQualityFilter，也可能是 deita 模型不可用时的 AlpagasusFilter 退化；
    # 以 step3 产物里的分数列名判断实际跑了哪一个。
    sft_ops = list(SFT_OPERATORS)
    if 3 in steps and steps[3]:
        cols = set(steps[3][0].keys())
        if "AlpagasusScore" in cols and "DeitaQualityScore" not in cols:
            sft_ops[2] = "AlpagasusFilter(fallback)"
    counts, killed_by, survivors = funnel(steps, sft_ops, input_rows)

    manifest = {m["id"]: m for m in json.load(open(args.manifest_sft, encoding="utf-8"))}
    id_rows = []
    for m in manifest.values():
        rid = m["id"]
        killed = killed_by.get(rid)
        actual = "keep" if rid in survivors else "drop"
        expect = m["expect"]
        ok = (expect == "keep") == (actual == "keep")
        id_rows.append({
            "id": rid, "defect": m["defect"], "expect": expect,
            "actual": actual, "killed_by": killed, "match": ok,
        })

    # 分数分布：step2 输出带 SuperfilteringScore，step3 输出带质量分（DeitaQualityScore 或 AlpagasusScore）
    sf_stats = col_stats(steps.get(2, []), "SuperfilteringScore") if 2 in steps else None
    sf_scores = [{"id": r.get("id"), "SuperfilteringScore": r.get("SuperfilteringScore")}
                 for r in steps.get(2, [])]
    qcol = "DeitaQualityScore" if (3 in steps and steps[3] and "DeitaQualityScore" in steps[3][0]) else "AlpagasusScore"
    q_stats = col_stats(steps.get(3, []), qcol) if 3 in steps else None
    q_scores = [{"id": r.get("id"), qcol: r.get(qcol)} for r in steps.get(3, [])]

    report["sft"] = {
        "status": "ok",
        "input_rows": len(input_rows),
        "steps": len(steps),
        "survivors": len(survivors),
        "funnel": [{"operator": op, "rows": n} for op, n in counts],
        "killed_by": killed_by,
        "survivor_ids": sorted(survivors),
        "id_rows": id_rows,
        "manifest_match": {
            "total": len(id_rows),
            "matched": sum(1 for r in id_rows if r["match"]),
            "mismatched": [r["id"] for r in id_rows if not r["match"]],
        },
        "superfiltering_score_stats": sf_stats,
        "superfiltering_scores": sf_scores,
        "quality_score_column": qcol,
        "quality_score_stats": q_stats,
        "quality_scores": q_scores,
    }


def verify_syn(args, report):
    cache, prefix = args.syn_cache, args.syn_prefix
    if not os.path.isdir(cache):
        report["syn"] = {"status": "skipped (no cache)"}
        return
    steps = load_steps(cache, prefix)
    counts = [(SYN_OPERATORS[n - 1] if n - 1 < len(SYN_OPERATORS) else f"step{n}", len(rows))
              for n, rows in sorted(steps.items())]

    step3 = steps.get(3, [])
    final = []
    for r in step3:
        final.append({
            "difficulty": r.get("difficulty"),
            "instruction": (r.get("instruction") or "")[:120],
            "output_len": len(r.get("output") or ""),
            "AlpagasusScore": r.get("AlpagasusScore"),
        })
    diff_dist = {}
    for r in final:
        diff_dist[r["difficulty"]] = diff_dist.get(r["difficulty"], 0) + 1

    api_stats = None
    api_path = os.path.join(cache, "api_calls.jsonl")
    if os.path.exists(api_path):
        calls = read_jsonl(api_path)
        lat = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
        api_stats = {
            "total_calls": len(calls),
            "ok_calls": sum(1 for c in calls if c.get("ok")),
            "latency_mean_s": round(statistics.fmean(lat), 2) if lat else None,
            "latency_max_s": round(max(lat), 2) if lat else None,
            "first_call_ts": calls[0]["ts"] if calls else None,
            "last_call_ts": calls[-1]["ts"] if calls else None,
        }

    report["syn"] = {
        "status": "ok",
        "funnel": [{"operator": op, "rows": n} for op, n in counts],
        "final_rows": final,
        "difficulty_distribution": diff_dist,
        "api_calls": api_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-cache"); ap.add_argument("--pt-prefix")
    ap.add_argument("--sft-cache"); ap.add_argument("--sft-prefix")
    ap.add_argument("--syn-cache"); ap.add_argument("--syn-prefix")
    ap.add_argument("--pt-input"); ap.add_argument("--sft-input")
    ap.add_argument("--manifest-pt"); ap.add_argument("--manifest-sft")
    ap.add_argument("--out")
    args = ap.parse_args()

    report = {}
    if args.pt_cache and args.pt_input:
        verify_pt(args, report)
    if args.sft_cache and args.sft_input:
        verify_sft(args, report)
    if args.syn_cache:
        verify_syn(args, report)

    # ---- 控制台输出 ----
    def hr(t):
        print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

    if "pt" in report and report["pt"].get("status") == "ok":
        pt = report["pt"]
        hr("PT 预训练数据过滤漏斗")
        for f in pt["funnel"]:
            print(f"  {f['operator']:32s} {f['rows']:>4d}")
        print(f"  存活 {pt['survivors']}/{pt['input_rows']}；清单吻合 {pt['manifest_match']['matched']}/{pt['manifest_match']['total']}")
        if pt["manifest_match"]["mismatched"]:
            print("  不吻合 id:", pt["manifest_match"]["mismatched"])
        print("  killed_by:", json.dumps(pt["killed_by"], ensure_ascii=False))
        if pt["pairqual_score_stats"]:
            print("  PairQualScore 分布:", pt["pairqual_score_stats"])
        for rid, e in pt["refiner_effect"].items():
            print(f"  refiner {rid}: changed={e['changed']}")

    if "sft" in report and report["sft"].get("status") == "ok":
        sft = report["sft"]
        hr("SFT 数据过滤漏斗")
        for f in sft["funnel"]:
            print(f"  {f['operator']:32s} {f['rows']:>4d}")
        print(f"  存活 {sft['survivors']}/{sft['input_rows']}；清单吻合 {sft['manifest_match']['matched']}/{sft['manifest_match']['total']}")
        if sft["manifest_match"]["mismatched"]:
            print("  不吻合 id:", sft["manifest_match"]["mismatched"])
        print("  killed_by:", json.dumps(sft["killed_by"], ensure_ascii=False))
        print("  SuperfilteringScore 分布:", sft["superfiltering_score_stats"])
        print(f"  {sft['quality_score_column']} 分布:", sft["quality_score_stats"])

    if "syn" in report and report["syn"].get("status") == "ok":
        syn = report["syn"]
        hr("SFT 数据合成")
        for f in syn["funnel"]:
            print(f"  {f['operator']:32s} {f['rows']:>4d}")
        print("  difficulty 分布:", syn["difficulty_distribution"])
        print("  api_calls:", syn["api_calls"])
        for r in syn["final_rows"]:
            print(f"  [{r['difficulty']:6s}] score={r['AlpagasusScore']} out_len={r['output_len']:5d} :: {r['instruction'][:80]}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("\n[written]", args.out)


if __name__ == "__main__":
    main()
