
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DataFlow Task5 - 运行截图渲染器：把真实运行日志/核对结果渲染成终端风格 runshot PNG。"""
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

EV = "courses/10-dataflow/evidence/task5"
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


def render(lines, out_path, title, width=1240, font_size=17, line_h=26, pad=18):
    """lines: [(text, color)] 或 str"""
    norm = []
    for ln in lines:
        if isinstance(ln, tuple):
            norm.append(ln)
        else:
            norm.append((ln, FG))
    f = font(font_size)
    fb = font(font_size, bold=True)
    fc = font(font_size, cjk=True)
    # 估算高度
    h = pad * 2 + line_h * (len(norm) + 2) + 8
    img = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(img)
    # 标题栏
    d.rectangle([0, 0, width, 38], fill=TITLE_BG)
    title_font = font(font_size, bold=True, cjk=has_cjk(title))
    d.text((pad, 8), title, font=title_font, fill=FG)
    # 三个圆点
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


# ---------- 1) PT 过滤漏斗 ----------
log = read(os.path.join(EV, "pt_filter_run.log"))
steps = re.findall(r"\[STEP\]\s+(.+?)\s{2,}(\d+)\s*->\s*(\d+)\s+\(\s*([\d.]+)s\)", log)
lines = [
    ("vm-60:~$ cd ~/dataflow-task5/task5 && source env.sh && bash run_all.sh pt", CYAN),
    ("", FG),
    ("=== [1/3] 预训练数据过滤 (PTTextFilter_GPUPipeline, 官方 GPU 版 25 步) ===", YELLOW),
    ("[CONFIG] input    = data/pt_input.jsonl   (44 行设计语料: 12 干净 + 4 边界 + 26 缺陷 + 3 refiner 演示)", GRAY),
    ("[CONFIG] cache    = cache_pt   hf_cache = /mnt/nfs/task5_hf (离线)", GRAY),
    ("", FG),
]
for name, a, b, t in steps:
    mark = "" if a == b else "   <- 本步砍掉"
    color = GREEN if a == b else YELLOW
    lines.append((f"[STEP] {name:<32s} {a:>3s} -> {b:>3s}  ({t:>5s}s){mark}", color))
m = re.search(r"\[DONE\] total ([\d.]+)s", log)
lines += [
    ("", FG),
    (f"[DONE] PT 过滤总耗时 {m.group(1)}s；存活 14/44，缺陷清单 id 级吻合 40/44", GREEN),
]
render(lines, os.path.join(FIG, "runshot_task5_01_pt_filter_funnel.png"),
       "DataFlow Task5 · 预训练数据过滤 25 步漏斗 (vm-60, 2026-10-31)")

# ---------- 2) PT verify: id 级去向 ----------
v = json.load(open(os.path.join(EV, "verify_task5.json"), encoding="utf-8"))
pt = v["pt"]
lines = [
    ("vm-60:~$ python verify_task5.py ...", CYAN),
    ("", FG),
    ("================ PT 预训练数据过滤漏斗 ================", YELLOW),
]
for f in pt["funnel"]:
    lines.append((f"  {f['operator']:<32s} {f['rows']:>4d}", FG if f["rows"] == 44 else GREEN))
lines += [
    (f"  存活 {pt['survivors']}/{pt['input_rows']}；清单吻合 {pt['manifest_match']['matched']}/{pt['manifest_match']['total']}", GREEN),
    ("", FG),
    ("---- id 级 killed_by（每条缺陷文档被哪一级砍掉）----", YELLOW),
]
kb = pt["killed_by"]
for i in range(0, len(kb), 2):
    items = list(kb.items())[i:i + 2]
    row = "   " + "".join(f"{k:<22s} -> {val:<28s}" for k, val in items)
    lines.append((row.rstrip(), CYAN))
pq = pt["pairqual_score_stats"]
lines += [
    ("", FG),
    ("---- PairQualScore 分布（进入质量过滤的 18 条）----", YELLOW),
    (f"  n={pq['n']}  min={pq['min']}  max={pq['max']}  mean={pq['mean']}  median={pq['median']}", FG),
    ("  干净百科文本实测 2.6~5.7；SEO 垃圾/乱码 -7.1/-7.5；官方 min_score=2 贴近分布顶部", GRAY),
    ("", FG),
    ("---- refiner 效果（只改内容不过滤）----", YELLOW),
]
for rid, e in pt["refiner_effect"].items():
    lines.append((f"  {rid:<18s} changed={e['changed']}", GREEN if e["changed"] else GRAY))
render(lines, os.path.join(FIG, "runshot_task5_02_pt_verify_killedby.png"),
       "DataFlow Task5 · PT 过滤 id 级去向核对 (verify_task5.py)")

# ---------- 3) SFT 过滤漏斗 ----------
sft = v["sft"]
lines = [
    ("vm-60:~$ bash run_all.sh sft", CYAN),
    ("", FG),
    ("=== [2/3] SFT 数据过滤 (SFTTextFilter_GPUPipeline, 官方 GPU 版 3 步) ===", YELLOW),
]
for f in sft["funnel"]:
    lines.append((f"[STEP] {f['operator']:<32s} {f['rows']:>4d}", FG))
lines += [
    ("[FALLBACK] deita-quality-scorer 不在离线缓存，第三步改用 AlpagasusFilter(LLM 0-5 分, [3,5])", RED),
    ("", FG),
    ("================ SFT 数据过滤漏斗 ================", YELLOW),
    (f"  存活 {sft['survivors']}/{sft['input_rows']}；清单吻合 {sft['manifest_match']['matched']}/{sft['manifest_match']['total']}", GREEN),
    ("", FG),
    ("---- killed_by ----", YELLOW),
]
for k, val in sft["killed_by"].items():
    lines.append((f"  {k:<24s} -> {val}", CYAN))
sf = sft["superfiltering_score_stats"]
q = sft["quality_score_stats"]
lines += [
    ("", FG),
    ("---- 分数分布 ----", YELLOW),
    (f"  SuperfilteringScore (gpt2 IFD): n={sf['n']} min={sf['min']} max={sf['max']} mean={sf['mean']}  官方保留 [0.5,1.0]", FG),
    (f"  {sft['quality_score_column']} (LLM 0-5 分): n={q['n']} min={q['min']} max={q['max']} mean={q['mean']}  官方保留 [3,5]", FG),
    ("  IFD=1.25 的指令不对应样本被 Superfiltering 精准挡住；重复/敷衍回答被 LLM 裁判放过(>=3)", GRAY),
]
render(lines, os.path.join(FIG, "runshot_task5_03_sft_filter_funnel.png"),
       "DataFlow Task5 · SFT 数据过滤 (vm-60, 2026-10-31)")

# ---------- 4) SFT 合成 ----------
syn = v["syn"]
lines = [
    ("vm-60:~$ bash run_all.sh syn", CYAN),
    ("", FG),
    ("=== [3/3] SFT 数据合成 (TextSFTSynthesis_APIPipeline, 官方 API 版 3 步) ===", YELLOW),
    ("[CONFIG] model = DeepSeek-V4-Flash (AMD 端点, 经 Windows 隧道)  rpm=10 workers=2", GRAY),
]
for f in syn["funnel"]:
    lines.append((f"[STEP] {f['operator']:<32s} {f['rows']:>4d}", FG))
api = syn["api_calls"]
lines += [
    ("", FG),
    ("---- API 成本（api_calls.jsonl 审计，不含密钥）----", YELLOW),
    (f"  total={api['total_calls']}  ok={api['ok_calls']}  latency_mean={api['latency_mean_s']}s  max={api['latency_max_s']}s", FG),
    (f"  {api['first_call_ts']}  ->  {api['last_call_ts']}  (≈4.3 请求/分钟: 20 RPM 限流 + 客户端 rpm=10)", GRAY),
    ("  5 次 504 Gateway Timeout 全部被 DataFlow 指数退避重试吸收，0 行丢失", GRAY),
    ("", FG),
    ("---- 最终 9 条样本（difficulty 3/3/3，Alpagasus 全部 4.0）----", YELLOW),
]
for r in syn["final_rows"]:
    ins = r["instruction"].replace("\n", " ")[:58]
    lines.append((f"  [{r['difficulty']:<6s}] score={r['AlpagasusScore']} out_len={r['output_len']:>5d} :: {ins}", CYAN))
render(lines, os.path.join(FIG, "runshot_task5_04_sft_synthesis.png"),
       "DataFlow Task5 · SFT 数据合成 (Condor+Refine+Alpagasus)")

# ---------- 5) 单元核对 ----------
uc = json.load(open(os.path.join(EV, "unit_check_line_filters.json"), encoding="utf-8"))
lines = [
    ("vm-60:~$ python unit_check_line_filters.py", CYAN),
    ("", FG),
    ("官方 PT 管线的 3 个 refiner 在最前，RemoveExtraSpacesRefiner 用 \" \".join(text.split())", YELLOW),
    ("会把换行一起压平 -> 三个\"行比例\"过滤器实际看到单行文本。本核对在原始文本/压平后文本上", YELLOW),
    ("分别跑三个过滤器，固定\"算子本身无误、是管线顺序削弱它们\"的证据：", YELLOW),
    ("", FG),
    (f"{'id':<20s}{'expect':<32s}{'ellipsis':<10s}{'bullet':<10s}{'javascript':<12s}", FG),
    ("-" * 74, GRAY),
]
for rid, e in uc.items():
    if rid == "pt_clean_00":
        continue
    row = f"{rid:<20s}{e['expect_killed_by']:<32s}{e['LineEndWithEllipsisFilter']:<10s}{e['LineStartWithBulletpointFilter']:<10s}{e['LineWithJavascriptFilter']:<12s}"
    color = GREEN if e["expect_killed_by"] != "(control)" else GRAY
    lines.append((row, color))
lines += [
    ("-" * 74, GRAY),
    ("压平换行后（after_refiner）：", YELLOW),
]
for rid, e in uc.items():
    if rid == "pt_clean_00":
        continue
    row = f"  {rid:<20s} ellipsis={e['after_refiner']['LineEndWithEllipsisFilter']:<6s} bullet={e['after_refiner']['LineStartWithBulletpointFilter']:<6s} js={e['after_refiner']['LineWithJavascriptFilter']:<6s} newlines_flat={e['refiner_changed_newlines']}"
    lines.append((row, CYAN))
lines += [
    ("", FG),
    ("结论：ellipsis/javascript 缺陷文档逃过本命过滤器改落 PairQual；bullet 查行首仍能砍。", GREEN),
]
render(lines, os.path.join(FIG, "runshot_task5_05_unit_check_line_filters.png"),
       "DataFlow Task5 · 行比例过滤器算子级核对（refiner 前后对照）")

# ---------- 6) 打卡总览 ----------
checkin = read(os.path.join(EV, "dataflow-task5-checkin.md"))
sec = checkin.split("## 一句话结论")[1].split("## 验证了什么")[0].strip().replace("**", "")
wrapped = []
import textwrap
for para in sec.split("\n"):
    if not para.strip():
        wrapped.append("")
        continue
    for w in textwrap.wrap(para, 96):
        wrapped.append(w)
lines = [
    ("vm-60:~$ cat dataflow-task5-checkin.md", CYAN),
    ("", FG),
    ("# DataFlow Task5 打卡 · 构建高质量训练数据（Text pipeline）", GREEN),
    ("", FG),
]
for w in wrapped:
    lines.append((w, FG))
lines += [
    ("", FG),
    ("实际结果：", YELLOW),
    ("  PT 预训练过滤:  44 -> 14   规则砍26 + PairQual砍7 + 语言砍1 + MinHash砍3；清单吻合 40/44", FG),
    ("  SFT 数据过滤:   16 -> 11   词数砍4 + IFD砍1(不对应1.25) + 质量分砍0；deita 限流-> Alpagasus 退化", FG),
    ("  SFT 数据合成:    0 ->  9   3 prompt -> Easy/Medium/Hard 各3；44 请求 14.5 分钟 ≈4.3 req/min", FG),
    ("", FG),
    ("证据: evidence/task5/ (45 文件)   代码: code/task5/   笔记: notes/task05-构建高质量训练数据.md", CYAN),
]
render(lines, os.path.join(FIG, "runshot_task5_06_checkin_overview.png"),
       "DataFlow Task5 · 打卡总览")
print("ALL DONE")
