#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataFlow Task6 — 结果核对（不信任算子输出，逐 id 独立重算 + 聚合对账 + 跨模型对比）

核对内容（每个 run 一个 cache 目录）：
  1) 聚合对账：eval_result.jsonl 的 total/valid/accuracy(或 ppl_mean/micro_f1_mean)
     与最终 step jsonl 里 eval_score 列的独立聚合一致；
  2) 指标重算：
     - key2_qa / key2_q_ma：用 math_verify + 自实现文本包含（NFKC/下标/去尾标点/casefold）
       对 (generated_ans, golden_label/targets) 独立重算 0/1，与算子 eval_score 逐 id 比对；
     - key3_q_choices_a：检查 eval_pred 下标合法且 score == (pred == label)；
       --ll-spotcheck N 时用 transformers 独立前向重算前 N 行各选项 loglikelihood，比对 argmax；
     - key3_q_choices_as：从 eval_pred JSON 的 pred_set/gold_set 独立重算 micro-F1；
     - key3_q_a_rejected：从 eval_pred JSON 的 ll_better/ll_rejected 独立重算胜者；
     - key1_text_score：ppl 有限且 > 0，均值对账；
  3) 完整性：id 唯一、generated_ans 非空（生成类类型）、eval_error 分布；
  4) API run：api_calls.jsonl 请求数/全部 ok；
  5) 跨模型对比：同一 eval_type 下 local vs api 的指标并列。

用法：
  python verify_task6.py --runs '{"t2_local": {"cache": "cache_t2_local", "prefix": "bench_t2_local", "eval_type": "key2_qa", "model": "local"}}' --out verify_task6.json
  （run_all.sh 会自动生成 --runs 参数）
"""
import argparse
import json
import os
import re
import unicodedata

import pandas as pd

SUBSCRIPTS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}


def normalize_for_match(text):
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate(str.maketrans(SUBSCRIPTS))
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    if s.endswith((".", "。", "!", "！", "?", "？")):
        s = s[:-1].strip()
    return s.casefold()


def contains_match(pred, ref):
    p, r = normalize_for_match(pred), normalize_for_match(ref)
    if not p or not r:
        return False
    return (r in p) or (p in r)


def load_final_df(cache, prefix):
    # 最终 step：operator_step 2（step1=生成后，step2=评测后）；ppl/choice 类只有 step2
    path2 = os.path.join(cache, f"{prefix}_step2.jsonl")
    path1 = os.path.join(cache, f"{prefix}_step1.jsonl")
    path = path2 if os.path.exists(path2) else path1
    if not os.path.exists(path):
        raise FileNotFoundError(f"no step file under {cache} for prefix {prefix}")
    return pd.read_json(path, lines=True), os.path.basename(path)


def normalize_targets(raw):
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x) for x in obj if str(x).strip()]
        except Exception:
            pass
    for sep in ("||", "|", ";"):
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def try_math_verify(answer, gt):
    try:
        from math_verify import parse, verify

        return verify(parse(str(gt)), parse(str(answer))) is True
    except Exception:
        return False


def get_answer_extractor():
    """算子同款答案抽取器（AnswerExtractor）。用它复现算子的第三条判定路径；
    其余判定（归一化/包含/math_verify/聚合）均为本脚本独立实现。"""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from dataflow.utils.reasoning.AnswerExtraction import AnswerExtractor, StringCleaner, UnitTextManager

        _EXTRACTOR = AnswerExtractor(StringCleaner(UnitTextManager()))
    return _EXTRACTOR


_EXTRACTOR = None


def qa_paths(pred_raw, gt):
    """复现算子 _eval_qa_single 的三条判定路径并归因：
       A contains(pred_raw, gt)        —— 原文双向子串包含（宽松）
       B contains(extract(pred), gt)   —— 抽取答案双向子串包含（宽松）
       C math_verify(extract(pred), gt)—— 严格数学等价
    返回 (ok, path)；ok=False 时 path='none'。"""
    fa = get_answer_extractor().extract_answer(pred_raw, None)
    if try_math_verify(fa, gt):
        return True, "math_verify"
    if contains_match(fa, gt):
        return True, "text_contains(extracted)"
    if contains_match(pred_raw, gt):
        return True, "text_contains(raw)"
    return False, "none"


def recompute_qa_row(pred_raw, eval_pred, gt):
    """独立重算单参考 QA 的 0/1（复现算子文档化口径：三路径取或）"""
    ok, _ = qa_paths(pred_raw, gt)
    return ok


def set_metrics(pred_set, gold_set):
    pred_set, gold_set = set(pred_set), set(gold_set)
    inter = len(pred_set & gold_set)
    p = inter / len(pred_set) if pred_set else 0.0
    r = inter / len(gold_set) if gold_set else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return f1


def verify_run(name, spec, args):
    cache, prefix = spec["cache"], spec["prefix"]
    eval_type = spec["eval_type"]
    report = {"name": name, "eval_type": eval_type, "model": spec.get("model"), "checks": [], "ok": True}

    try:
        df, step_file = load_final_df(cache, prefix)
    except FileNotFoundError as e:
        report["checks"].append({"check": "step_files_exist", "ok": False, "error": str(e)})
        report["ok"] = False
        return report
    report["step_file"] = step_file
    report["rows"] = int(len(df))

    # ---------- 1) 聚合对账 ----------
    eval_result_path = os.path.join(cache, "eval_result", "eval_result.jsonl")
    if not os.path.exists(eval_result_path):
        report["checks"].append({"check": "eval_result_exists", "ok": False})
        report["ok"] = False
        return report
    stats = json.load(open(eval_result_path, encoding="utf-8"))
    if isinstance(stats, list):  # 算子用 pd.DataFrame([stats]).to_json(orient="records") 落盘，是单元素数组
        stats = stats[0] if stats else {}
    report["operator_stats"] = stats

    valid_mask = df["eval_valid"] == True  # noqa: E712
    n_valid = int(valid_mask.sum())
    agg_key = {"key1_text_score": "ppl_mean", "key3_q_choices_as": "micro_f1_mean"}.get(eval_type, "accuracy")
    agg_val = float(df.loc[valid_mask, "eval_score"].mean()) if n_valid else 0.0
    op_val = float(stats.get(agg_key, -1))
    agg_ok = (int(stats.get("total_samples", -1)) == len(df)) and (int(stats.get("valid_samples", -1)) == n_valid) \
        and abs(op_val - agg_val) < 1e-9
    report["checks"].append({
        "check": "aggregate_match",
        "ok": bool(agg_ok),
        "metric": agg_key,
        "operator": op_val,
        "recomputed": agg_val,
        "total": len(df), "valid": n_valid,
    })

    # ---------- 2) 指标重算 ----------
    is_semantic = name.endswith("semantic")
    mismatches = []
    match_paths = {}
    if eval_type in ("key2_qa", "key2_q_ma"):
        gt_col = "golden_label" if eval_type == "key2_qa" else "targets"
        for _, row in df.iterrows():
            if not bool(row["eval_valid"]):
                continue
            gts = [str(row[gt_col])] if eval_type == "key2_qa" else normalize_targets(row[gt_col])
            ok = any(recompute_qa_row(row.get("generated_ans"), row.get("eval_pred"), gt) for gt in gts)
            if float(ok) != float(row["eval_score"]):
                mismatches.append({"id": row.get("id"), "operator": float(row["eval_score"]), "recomputed": float(ok)})
            if ok:
                # 记录该行是靠严格命中还是仅靠子串包含得分（宽松度归因）
                path = "none"
                for gt in gts:
                    _, p = qa_paths(row.get("generated_ans"), gt)
                    if p != "none":
                        path = p
                        break
                match_paths[str(row.get("id"))] = path
        if is_semantic:
            # semantic_judge 是 LLM 裁判口径，与 math_verify 不是同一把尺子：
            # 不做同口径 pass/fail，改为记录"裁判判对但严格口径判错"的行（度量口径差异证据）
            lenient_rows = [m["id"] for m in mismatches if m["operator"] == 1.0]
            strict_miss = [m["id"] for m in mismatches if m["operator"] == 0.0]
            gap_check = {
                "check": "semantic_vs_mathverify_gap", "ok": True,
                "judge_pass_but_strict_fail": lenient_rows,
                "judge_fail_but_strict_pass": strict_miss,
            }
            if match_paths:
                from collections import Counter
                gap_check["mathverify_path_counts"] = dict(Counter(match_paths.values()))
            report["checks"].append(gap_check)
        else:
            check = {"check": "metric_recompute", "ok": not mismatches,
                     "n_compared": int(valid_mask.sum()), "mismatches": mismatches[:20]}
            if match_paths:
                from collections import Counter
                check["match_path_counts"] = dict(Counter(match_paths.values()))
            report["checks"].append(check)

    elif eval_type == "key3_q_choices_a":
        for _, row in df.iterrows():
            if not bool(row["eval_valid"]):
                continue
            pred_idx = row.get("eval_pred")
            label = row.get("label")
            choices = row.get("choices")
            n = len(choices) if isinstance(choices, list) else -1
            # 一致性：score 必须等于 (pred == label)，且 pred 下标合法
            ok = (pred_idx is not None) and (0 <= int(pred_idx) < n) \
                and (float(row["eval_score"]) == float(int(pred_idx) == int(label)))
            if not ok:
                mismatches.append({"id": row.get("id"), "pred": pred_idx, "label": label,
                                   "score": float(row["eval_score"])})
        report["checks"].append({"check": "mc_single_consistency", "ok": not mismatches,
                                 "n_compared": int(valid_mask.sum()), "mismatches": mismatches[:20]})

    elif eval_type == "key3_q_choices_as":
        for _, row in df.iterrows():
            if not bool(row["eval_valid"]):
                continue
            info = json.loads(row["eval_pred"])
            f1 = set_metrics(info["pred_set"], info["gold_set"])
            if abs(f1 - float(row["eval_score"])) > 1e-9:
                mismatches.append({"id": row.get("id"), "operator": float(row["eval_score"]), "recomputed": f1})
        report["checks"].append({"check": "micro_f1_recompute", "ok": not mismatches,
                                 "n_compared": int(valid_mask.sum()), "mismatches": mismatches[:20]})

    elif eval_type == "key3_q_a_rejected":
        for _, row in df.iterrows():
            if not bool(row["eval_valid"]):
                continue
            info = json.loads(row["eval_pred"])
            win = 1.0 if float(info["ll_better"]) > float(info["ll_rejected"]) else 0.0
            if win != float(row["eval_score"]):
                mismatches.append({"id": row.get("id"), "operator": float(row["eval_score"]), "recomputed": win})
        report["checks"].append({"check": "pairwise_recompute", "ok": not mismatches,
                                 "n_compared": int(valid_mask.sum()), "mismatches": mismatches[:20]})

    elif eval_type == "key1_text_score":
        ppl_ok = bool(((df["eval_score"] > 0) & (df["eval_score"] < 1e7)).all())
        report["checks"].append({"check": "ppl_range", "ok": ppl_ok,
                                 "ppl_min": float(df["eval_score"].min()),
                                 "ppl_max": float(df["eval_score"].max())})

    # ---------- 3) 完整性 ----------
    completeness = {}
    if "id" in df.columns:
        completeness["id_unique"] = bool(df["id"].is_unique)
        completeness["id_count"] = int(df["id"].nunique())
    if "generated_ans" in df.columns and eval_type in ("key2_qa", "key2_q_ma", "key3_q_choices_as"):
        gen = df["generated_ans"].fillna("").astype(str).str.strip()
        completeness["generated_ans_nonempty"] = int((gen != "").sum())
        completeness["generated_ans_total"] = int(len(gen))
    if "eval_error" in df.columns:
        errs = df.loc[~valid_mask, "eval_error"].fillna("").astype(str)
        completeness["error_distribution"] = {k: int(v) for k, v in errs.value_counts().items()}
    report["checks"].append({"check": "completeness", "ok": True, "detail": completeness})

    # ---------- 4) API 审计 ----------
    api_log = os.path.join(cache, "api_calls.jsonl")
    if os.path.exists(api_log):
        calls = [json.loads(l) for l in open(api_log, encoding="utf-8") if l.strip()]
        report["checks"].append({
            "check": "api_audit",
            "ok": all(c.get("ok") for c in calls) and len(calls) > 0,
            "n_calls": len(calls),
            "n_failed": int(sum(1 for c in calls if not c.get("ok"))),
            "latency_mean_s": round(sum(c["latency_s"] for c in calls) / max(1, len(calls)), 2),
        })

    report["ok"] = all(c.get("ok", True) for c in report["checks"])
    return report


def ll_spotcheck(spec, n_rows, model_path):
    """用 transformers 独立前向重算前 n_rows 行各选项 loglikelihood，与算子 argmax 比对"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache, prefix = spec["cache"], spec["prefix"]
    df, _ = load_final_df(cache, prefix)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                 dtype=torch.bfloat16).cuda().eval()

    def ll(prompt, cont):
        full_ids = tok(prompt + cont, add_special_tokens=False).input_ids
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        plen = len(p_ids) if full_ids[:len(p_ids)] == p_ids else max(0, len(full_ids) - len(tok(cont, add_special_tokens=False).input_ids))
        clen = max(0, len(full_ids) - plen)
        if clen <= 0:
            return 0.0
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([full_ids]).cuda()).logits[0]
            logprobs = torch.log_softmax(logits, dim=-1)
        start, end = plen - 1, plen + clen - 1
        return float(logprobs[start:end].gather(-1, torch.tensor(full_ids[start + 1:end + 1]).cuda().unsqueeze(-1)).sum())

    results = []
    for _, row in df.head(n_rows).iterrows():
        prompt = f"Question:\n{row['question']}\n\nChoices:\n" + "\n".join(
            f"{chr(65+i)}. {c}" for i, c in enumerate(row["choices"])) + "\n\nAnswer:"
        lls = [ll(prompt, " " + str(c)) for c in row["choices"]]
        argmax = int(max(range(len(lls)), key=lambda i: lls[i]))
        results.append({"id": row.get("id"), "lls": [round(x, 2) for x in lls],
                        "independent_argmax": argmax, "operator_pred": row.get("eval_pred"),
                        "label": row.get("label"), "agree": argmax == int(row.get("eval_pred"))})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="JSON: {name: {cache, prefix, eval_type, model}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ll-spotcheck", type=int, default=0)
    ap.add_argument("--ll-model", default=os.environ.get("T6_MODEL_PATH"))
    args = ap.parse_args()

    runs = json.loads(args.runs)
    reports = []
    for name, spec in runs.items():
        print(f"[VERIFY] {name} ({spec['eval_type']}, {spec.get('model')})", flush=True)
        rep = verify_run(name, spec, args)
        for c in rep["checks"]:
            print(f"  - {c['check']:24s} ok={c.get('ok')}", flush=True)
        if args.ll_spotcheck and spec["eval_type"] == "key3_q_choices_a" and spec.get("model") == "local":
            sc = ll_spotcheck(spec, args.ll_spotcheck, args.ll_model)
            rep["checks"].append({"check": "ll_spotcheck", "ok": all(r["agree"] for r in sc), "detail": sc})
            for r in sc:
                print(f"    spotcheck {r['id']}: indep_argmax={r['independent_argmax']} "
                      f"operator={r['operator_pred']} agree={r['agree']}", flush=True)
        rep["ok"] = all(c.get("ok", True) for c in rep["checks"])
        reports.append(rep)

    # ---------- 5) 跨模型对比 ----------
    by_type = {}
    for rep in reports:
        stats = rep.get("operator_stats", {})
        suffix = ""
        if rep.get("name", "").endswith("semantic"):
            suffix = "+semantic"
        elif "ref" in rep.get("name", ""):
            suffix = "+ref_answers"
        key = rep["eval_type"] + suffix
        by_type.setdefault(key, {})[rep.get("model", "?")] = {
            "run": rep["name"],
            **{k: stats.get(k) for k in ("accuracy", "micro_f1_mean", "ppl_mean", "total_samples", "valid_samples") if k in stats},
        }

    out = {
        "runs": reports,
        "comparison": by_type,
        "all_ok": all(r["ok"] for r in reports),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[SUMMARY] " + json.dumps(by_type, ensure_ascii=False, indent=2))
    print(f"[VERIFY] all_ok={out['all_ok']} -> {args.out}")


if __name__ == "__main__":
    main()
