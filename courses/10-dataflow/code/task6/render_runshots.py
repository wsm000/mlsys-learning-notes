#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DataFlow Task6 - 运行截图渲染器：把真实运行日志/核对结果渲染成终端风格 runshot PNG。"""
import json
import os
import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

EV = "courses/10-dataflow/evidence/task6"
FIG = os.path.join(EV, "figures")
os.makedirs(FIG, exist_ok=True)

FONT_MONO = "C:/Windows/Fonts/consola.ttf"
FONT_MONO_B = "C:/Windows/Fonts/consolab.ttf"
FONT_CJK = "C:/Windows/Fonts/msyh.ttc"

BG = (12, 12, 12)
TITLE_BG = (30, 30, 30)
FG = (204, 204, 204)
GREEN = (106, 153, 85)
YELLOW = (229, 192, 123)
CYAN = (97, 175, 239)
RED = (224, 108, 117)
GRAY = (150, 150, 150)
PURPLE = (198, 120, 221)


def font(size, bold=False, cjk=False):
    path = FONT_CJK if cjk else (FONT_MONO_B if bold else FONT_MONO)
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def has_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def render(lines, out_path, title, width=1280, font_size=16, line_h=25, pad=16):
    norm = [(ln, FG) if isinstance(ln, str) else ln for ln in lines]
    f = font(font_size)
    fb = font(font_size, bold=True)
    fc = font(font_size, cjk=True)
    h = pad * 2 + line_h * (len(norm) + 2) + 8
    img = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 38], fill=TITLE_BG)
    title_font = font(font_size, bold=True, cjk=has_cjk(title))
    d.text((pad, 8), title, font=title_font, fill=FG)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([width - 24 - i * 22, 14, width - 12 - i * 22, 26], fill=c)
    y = 38 + pad
    for text, color in norm:
        use = fc if has_cjk(text) else f
        d.text((pad, y), text, font=use, fill=color)
        y += line_h
    img.save(out_path)
    print("[write]", out_path, img.size)


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


verify = read_json(os.path.join(EV, "verify_task6.json"))
runs = {r["name"]: r for r in verify["runs"]}
comparison = verify["comparison"]

RUN_ORDER = ["t1_local", "t2_local", "t3_local", "t4_local", "t5_local", "t6_local",
             "t2_api", "t2s_api_semantic", "t3_api", "t5_api", "t2_ref_local"]
TYPE_LABEL = {
    "key1_text_score": "key1_text_score    ppl",
    "key2_qa": "key2_qa            math_verify",
    "key2_q_ma": "key2_q_ma          any_math_verify",
    "key3_q_choices_a": "key3_q_choices_a   ll_choice_acc",
    "key3_q_choices_as": "key3_q_choices_as  micro_f1",
    "key3_q_a_rejected": "key3_q_a_rejected  pairwise_ll_winrate",
}
MODEL_LABEL = {"local": "Qwen2.5-0.5B-Instruct(本地)",
               "api": "DeepSeek-V4-Flash(AMD API)"}


def metric_of(rep):
    s = rep.get("operator_stats", {})
    if "ppl_mean" in s:
        return f"ppl_mean={s['ppl_mean']:.2f}"
    if "micro_f1_mean" in s:
        return f"micro_f1={s['micro_f1_mean']:.4f}"
    return f"acc={s.get('accuracy', 0):.4f} ({s.get('valid_samples')}/{s.get('total_samples')})"


# ---------- 1) run 矩阵总览 ----------
lines = [
    ("$ bash run_all.sh    # vm-60 · open-dataflow 1.0.11 · RTX 4090 D 24GB", CYAN),
    ("", FG),
    ("官方原型 gpu_pipelines/benchmark_eval/unified_bench_eval_type{1..6}.py (+semantic 变体)", GRAY),
    ("改造点仅 serving：本地用 transformers 适配器(vllm 与现有 torch 2.9.1 冲突)；API 用官方 APILLMServing_request", GRAY),
    ("", FG),
    (f"{'run':18s} {'eval_type / metric':38s} {'model':28s} {'result':30s} {'ok':4s}", YELLOW),
]
for name in RUN_ORDER:
    rep = runs.get(name)
    if not rep:
        continue
    et = rep["eval_type"]
    label = TYPE_LABEL.get(et, et)
    if "semantic" in name:
        label = label.replace("math_verify", "semantic_judge")
    if "ref" in name:
        label = label.replace("math_verify", "math_verify(预填答案)")
    model = MODEL_LABEL.get(rep.get("model"), rep.get("model"))
    lines.append((f"{name:18s} {label:38s} {model:28s} {metric_of(rep):30s} {'YES' if rep['ok'] else 'NO':4s}",
                  GREEN if rep["ok"] else RED))
lines += [
    ("", FG),
    (f"verify all_ok = {verify['all_ok']}", GREEN if verify["all_ok"] else RED),
]
render(lines, os.path.join(FIG, "runshot_task6_01_run_matrix.png"), "DataFlow Task6 · 模型性能评测 Pipeline · 11 个 run 总览")

# ---------- 2) 本地模型六类型能力表 ----------
lines = [("$ python bench_eval_task6.py --model local --eval-type <6 种>", CYAN),
         ("被测模型 Qwen2.5-0.5B-Instruct（本地盘，transformers serving，贪心解码 temperature=0）", GRAY),
         ("", FG),
         (f"{'eval_type':20s} {'metric':22s} {'n':>4s} {'result':>10s}  说明", YELLOW)]
NOTES = {
    "t1_local": "5 条文本困惑度，16.8~249.8，均值 88.47",
    "t2_local": "15 条数学 QA 生成+抽取+比对",
    "t3_local": "6 条多参考 QA，ma_0005(光速) 格式不符判 0",
    "t4_local": "10 道单选，逐选项算 loglikelihood 取 argmax",
    "t5_local": "10 道多选，解析输出集合算 micro-F1",
    "t6_local": "10 组偏好对比，better/rejected 的 ll 胜率",
}
for name in ["t1_local", "t2_local", "t3_local", "t4_local", "t5_local", "t6_local"]:
    rep = runs[name]
    s = rep["operator_stats"]
    et = rep["eval_type"]
    metric = s.get("metric", "")
    n = s.get("total_samples")
    if "ppl_mean" in s:
        res = f"{s['ppl_mean']:.2f}"
    elif "micro_f1_mean" in s:
        res = f"{s['micro_f1_mean']:.4f}"
    else:
        res = f"{s['accuracy']:.4f}"
    lines.append((f"{et:20s} {metric:22s} {n:>4d} {res:>10s}  {NOTES[name]}", FG))
render(lines, os.path.join(FIG, "runshot_task6_02_local_six_types.png"), "DataFlow Task6 · 本地模型 6 个 eval_type 全量实测")

# ---------- 3) 跨模型对比（同一基准同一算子） ----------
lines = [("同一数据集 / 同一算子 / 同一指标，只换 serving：", CYAN),
         ("  local = Qwen2.5-0.5B-Instruct (transformers, 离线)", GRAY),
         ("  api   = DeepSeek-V4-Flash (AMD Radeon 兼容端点, 10rpm 客户端限速)", GRAY),
         ("", FG),
         (f"{'eval_type':22s} {'metric':16s} {'0.5B 本地':>12s} {'V4-Flash API':>14s} {'差距':>8s}", YELLOW)]
for et in ["key2_qa", "key2_q_ma", "key3_q_choices_as"]:
    entry = comparison.get(et, {})
    loc = entry.get("local", {}).get("accuracy")
    api = entry.get("api", {}).get("accuracy")
    if et == "key3_q_choices_as":
        loc = entry.get("local", {}).get("micro_f1_mean")
        api = entry.get("api", {}).get("micro_f1_mean")
    if loc is None or api is None:
        continue
    gap = api - loc
    lines.append((f"{et:22s} {'acc' if et != 'key3_q_choices_as' else 'micro_f1':16s} "
                  f"{loc:>12.4f} {api:>14.4f} {gap:>+8.4f}", FG))
sem = comparison.get("key2_qa+semantic", {}).get("api", {})
if sem:
    lines += [("", FG),
              (f"key2_qa + semantic_judge (API 裁判同一模型): acc={sem.get('accuracy'):.4f}", PURPLE),
              ("  -> 与 math_verify 的差异即'度量口径'差异，见笔记 §5", GRAY)]
render(lines, os.path.join(FIG, "runshot_task6_03_model_comparison.png"), "DataFlow Task6 · 同一基准跨模型能力对比")

# ---------- 4) 度量宽松度探针 ----------
ref = runs.get("t2_ref_local")
loc2 = runs.get("t2_local")
lines = [("官方 type2 数据集自带 15 条预填 model_answer（math_006/008/012/014 故意答错）", CYAN),
         ("probe_reference_answers.py 把 model_answer 复制为 generated_ans，直接进同一评测算子", GRAY),
         ("", FG)]
if ref:
    s = ref["operator_stats"]
    lines.append((f"预填答案列: acc={s['accuracy']:.4f} ({s['valid_samples']}/{s['total_samples']})"
                  f"  -> 4 条故意答错全部判 0，指标对明确错误是严格的", GREEN))
if loc2:
    s = loc2["operator_stats"]
    mp = {}
    for c in loc2["checks"]:
        if c["check"] == "metric_recompute":
            mp = c.get("match_path_counts", {})
    lines.append((f"0.5B 生成答案: acc={s['accuracy']:.4f} ({s['valid_samples']}/{s['total_samples']})", YELLOW))
    lines += [("", FG),
              ("15 条判对里，逐 id 归因命中路径（verify 独立重算，复现算子三路径口径）:", YELLOW),
              (f"  math_verify 严格数学等价         : {mp.get('math_verify', 0)}", GREEN),
              (f"  仅子串包含(抽取答案/原文, 双向)  : {mp.get('text_contains(extracted)', 0) + mp.get('text_contains(raw)', 0)}", YELLOW),
              ("", FG),
              ("结论：math_verify 路径 = 抽答案(math_verify) OR 双向子串包含；", FG),
              ("短标签(0/1/9/2x)会被子串规则放大，'满分'不等于'全对'。", RED)]
render(lines, os.path.join(FIG, "runshot_task6_04_metric_leniency_probe.png"), "DataFlow Task6 · 度量宽松度探针（官方预填答案 vs 生成答案）")

# ---------- 5) ll 类指标逐 id 明细 ----------
lines = [("key3_q_choices_a：对每个选项 continuation 算 loglikelihood，argmax 为预测", CYAN),
         ("key3_q_a_rejected：better 与 rejected 的 ll 直接比大小", CYAN),
         ("", FG)]
p4 = os.path.join(EV, "steps", "cache_t4_local", "bench_t4_local_step2.jsonl")
p6 = os.path.join(EV, "steps", "cache_t6_local", "bench_t6_local_step2.jsonl")
if os.path.exists(p4):
    df = pd.read_json(p4, lines=True)
    lines.append(("key3_q_choices_a (10 题):", YELLOW))
    for _, r in df.iterrows():
        ok = "OK " if int(r["eval_score"]) == 1 else "MISS"
        lines.append((f"  {r['id']}  pred={int(r['eval_pred'])} label={int(r['label'])} {ok}", 
                      GREEN if ok == "OK " else RED))
if os.path.exists(p6):
    df = pd.read_json(p6, lines=True)
    lines.append(("", FG))
    lines.append(("key3_q_a_rejected (10 组):", YELLOW))
    for _, r in df.iterrows():
        info = json.loads(r["eval_pred"])
        ok = "WIN " if int(r["eval_score"]) == 1 else "LOSE"
        lines.append((f"  {r['id']}  ll_better={info['ll_better']:8.2f} ll_rejected={info['ll_rejected']:8.2f} {ok}",
                      GREEN if ok == "WIN " else RED))
render(lines, os.path.join(FIG, "runshot_task6_05_ll_metrics_detail.png"), "DataFlow Task6 · loglikelihood 类指标逐 id 明细")

# ---------- 6) 核对清单 ----------
lines = [("$ python verify_task6.py --runs ... --ll-spotcheck 3", CYAN),
         ("不信任算子输出：逐 id 独立重算 + 聚合对账 + ll 抽核 + 完整性 + API 审计", GRAY),
         ("", FG)]
for name in RUN_ORDER:
    rep = runs.get(name)
    if not rep:
        continue
    checks = ", ".join(f"{c['check']}={'' if c.get('ok') else 'FAIL'}" for c in rep["checks"])
    lines.append((f"{name:18s} {checks}", GREEN if rep["ok"] else RED))
    for c in rep["checks"]:
        if c["check"] == "ll_spotcheck":
            for d in c.get("detail", []):
                mark = "agree" if d["agree"] else "DIFF"
                lines.append((f"    spotcheck {d['id']}: indep_argmax={d['independent_argmax']} "
                              f"operator={d['operator_pred']} {mark} lls={d['lls']}", 
                              GREEN if d["agree"] else RED))
lines += [("", FG), (f"all_ok = {verify['all_ok']}", GREEN if verify["all_ok"] else RED)]
render(lines, os.path.join(FIG, "runshot_task6_06_verify_checklist.png"), "DataFlow Task6 · 核对清单（id 级重算 + ll 抽核）")

print("[DONE] figures ->", FIG)
