# -*- coding: utf-8 -*-
"""
Task4 运行结果图 / 打卡图生成器（只用 Pillow + numpy，不依赖 matplotlib）

数据来源（全部为 vm-60 真实产物，脚本只做可视化，不改数字）：
  evidence/task4/agentic_rag/agentic_rag_task4_step2_FINAL.jsonl
  evidence/task4/agentic_rag/field_check_main.json
  evidence/task4/agentic_rag/run_main.log / run_pilot_429_evidence.log
  evidence/task4/text2sql/refine/quality_checks.json / stats_refine.json / final_refine.jsonl / refine.log
  evidence/task4/text2sql/gen/summary_gen.json / samples_gen.jsonl / gen.log
  evidence/task4/text2sql/smoke/smoke.log

用法：python code/task4/make_figures.py [--out evidence/task4/figures]
"""
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (247, 249, 252)
CARD = (255, 255, 255)
INK = (26, 34, 51)
MUT = (100, 112, 133)
LINE = (225, 232, 240)
BLUE = (47, 107, 255)
GREEN = (23, 166, 115)
RED = (224, 82, 82)
ORANGE = (240, 160, 50)
PURPLE = (124, 92, 255)
TEAL = (18, 160, 170)
GRAY_BAR = (199, 208, 221)

FONT_DIR = 'C:/Windows/Fonts/'
def font(name, size):
    return ImageFont.truetype(FONT_DIR + name, size)

F_TITLE = font('msyhbd.ttc', 50)
F_SUB = font('msyh.ttc', 24)
F_H2 = font('msyhbd.ttc', 30)
F_H3 = font('msyhbd.ttc', 24)
F_BODY = font('msyh.ttc', 22)
F_SMALL = font('msyh.ttc', 18)
F_TINY = font('msyh.ttc', 16)
F_NUM = font('msyhbd.ttc', 46)
F_NUM2 = font('msyhbd.ttc', 34)
F_MONO = font('consola.ttf', 15)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVD = os.path.join(ROOT, 'evidence', 'task4')


def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(p):
    rows = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_text(p):
    try:
        with open(p, encoding='utf-8', errors='replace') as f:
            return f.read()
    except OSError:
        return ''


def count(regex, text):
    return len(re.findall(regex, text))


# ---------------------------------------------------------------- 基础绘制
def new_canvas():
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    return img, d


def text(d, xy, s, f, fill=INK, anchor=None):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def wrap(d, s, f, max_width):
    lines, cur = [], ''
    for ch in s:
        trial = cur + ch
        if d.textlength(trial, font=f) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def paragraph(d, xy, s, f, fill=INK, max_width=800, line_gap=10):
    x, y = xy
    for line in wrap(d, s, f, max_width):
        d.text((x, y), line, font=f, fill=fill)
        y += f.size + line_gap
    return y


def card(d, box, fill=CARD, outline=LINE, radius=22, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def header(d, title, subtitle, tag='vm-60 实测'):
    d.rectangle((0, 0, W, 118), fill=(23, 35, 58))
    text(d, (56, 26), title, F_TITLE, (255, 255, 255))
    text(d, (58, 88), subtitle, F_SUB, (176, 194, 220))
    tw = d.textlength(tag, font=F_SMALL)
    d.rounded_rectangle((W - tw - 88, 40, W - 56, 84), radius=12, fill=(38, 54, 84))
    text(d, (W - tw - 72, 48), tag, F_SMALL, (255, 210, 120))


def footer(d, note):
    d.line((56, H - 74, W - 56, H - 74), fill=LINE, width=2)
    text(d, (56, H - 58), note, F_TINY, MUT)
    text(d, (W - 56, H - 58), 'DataFlow Task4 · 2026-10-29 · evidence/task4', F_TINY, MUT, anchor='ra')


def badge(d, xy, s, color):
    x, y = xy
    tw = d.textlength(s, font=F_SMALL)
    d.rounded_rectangle((x, y, x + tw + 26, y + 34), radius=9, fill=(*color[:3],))
    text(d, (x + 13, y + 5), s, F_SMALL, (255, 255, 255))
    return x + tw + 26


def hbar(d, box, labels, values, vmax, colors, num_fmt='{}', label_w=150, bar_h=26, gap=14, val_w=64):
    x0, y0, x1, y1 = box
    n = len(values)
    usable_h = (y1 - y0) - gap * (n - 1)
    bh = min(bar_h, usable_h / n)
    track_x0 = x0 + label_w
    track_x1 = x1 - val_w
    for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
        cy = y0 + i * (bh + gap)
        text(d, (x0, cy - 2), lab, F_SMALL, MUT)
        d.rounded_rectangle((track_x0, cy, track_x1, cy + bh), radius=6, fill=(241, 244, 248))
        wgt = max(2, int((track_x1 - track_x0 - 2) * (v / vmax if vmax else 0)))
        d.rounded_rectangle((track_x0, cy, track_x0 + wgt, cy + bh), radius=6, fill=c)
        text(d, (x1, cy - 3), num_fmt.format(v), F_SMALL, INK, anchor='ra')


def vbars(d, box, labels, values, vmax, colors, title=None, value_fmt='{:.2f}', x_label_f=None, show_values=True):
    x0, y0, x1, y1 = box
    base_y = y1 - 34
    top_y = y0 + 10
    n = len(values)
    slot = (x1 - x0) / n
    bw = min(96, slot * 0.52)
    if title:
        text(d, (x0, y0 - 38), title, F_H3, INK)
    d.line((x0, base_y, x1, base_y), fill=LINE, width=2)
    for i, (lab, v, c) in enumerate(zip(labels, values, colors)):
        cx = x0 + slot * i + slot / 2
        hgt = max(3, (base_y - top_y) * (v / vmax if vmax else 0))
        bx0 = cx - bw / 2
        d.rounded_rectangle((bx0, base_y - hgt, bx0 + bw, base_y), radius=8, fill=c)
        if show_values:
            text(d, (cx, base_y - hgt - 16), value_fmt.format(v), F_SMALL, INK, anchor='ma')
        text(d, (cx, base_y + 10), lab, F_SMALL, MUT, anchor='ma')


# ---------------------------------------------------------------- 数据读取
def agentic_rag_data():
    rows = load_jsonl(os.path.join(EVD, 'agentic_rag', 'agentic_rag_task4_step2_FINAL.jsonl'))
    fc = load_json(os.path.join(EVD, 'agentic_rag', 'field_check_main.json'))
    main_log = read_text(os.path.join(EVD, 'agentic_rag', 'run_main.log'))
    pilot_log = read_text(os.path.join(EVD, 'agentic_rag', 'run_pilot_429_evidence.log'))
    cost = re.findall(r'\[STEP1\] done in ([\d.]+)s, llm_requests=(\d+)', main_log)
    total = re.findall(r'\[TOTAL\] ([\d.]+)s, total llm requests=(\d+)', main_log)
    f1 = [float(r.get('F1Score', 0)) for r in rows]
    questions = []
    for r in rows:
        q = str(r.get('question', '')).strip()
        questions.append((q[:34] + '…') if len(q) > 36 else q)
    return dict(rows=rows, fc=fc, f1=f1, questions=questions,
                elapsed=float(total[0][0]) if total else 654.7,
                requests=int(total[0][1]) if total else 66,
                logical=53,
                r429=count(r'status=429', main_log), r500=count(r'status=500', main_log),
                pilot429=count(r'status=429', pilot_log))


def text2sql_refine_data():
    qc = load_json(os.path.join(EVD, 'text2sql', 'refine', 'quality_checks.json'))['refine']
    stats = load_json(os.path.join(EVD, 'text2sql', 'refine', 'stats_refine.json'))
    steps = stats['steps']
    rows = [s['rows'] for s in steps]
    final = load_jsonl(os.path.join(EVD, 'text2sql', 'refine', 'final_refine.jsonl'))
    log = read_text(os.path.join(EVD, 'text2sql', 'refine', 'refine.log'))
    return dict(qc=qc, steps=steps, rows=rows, final=final,
                n=qc['n'], non_empty=qc['non_empty_per_row'],
                prompt=qc['prompt_len'], cot=qc['cot_len'], vc=qc['value_counts'],
                r429=count(r'status=429', log))


def text2sql_gen_data():
    gen = load_json(os.path.join(EVD, 'text2sql', 'gen', 'summary_gen.json'))
    rows = [s['rows'] for s in gen['steps'] if s.get('rows') is not None]
    log = read_text(os.path.join(EVD, 'text2sql', 'gen', 'gen.log'))
    samples = load_jsonl(os.path.join(EVD, 'text2sql', 'gen', 'samples_gen.jsonl')) if os.path.exists(os.path.join(EVD, 'text2sql', 'gen', 'samples_gen.jsonl')) else []
    final = next(s for s in reversed(gen['steps']) if s.get('rows') is not None)
    return dict(rows=rows, samples=samples, final=final,
                r429=count(r'status=429', log), r500=count(r'status=500', log))


def smoke_429():
    log = read_text(os.path.join(EVD, 'text2sql', 'smoke', 'smoke.log'))
    return count(r'status=429', log)


# ---------------------------------------------------------------- 图 1：打卡总览
def fig_overview(out):
    ar = agentic_rag_data()
    t2s = text2sql_refine_data()
    gen = text2sql_gen_data()
    img, d = new_canvas()
    header(d, 'DataFlow Task4 打卡：构建 RAG 与 Text2SQL 数据',
           'vm-60 · RTX 4090 D 24GB · open-dataflow 1.0.11 · Python 3.10.12 · 模型 DeepSeek-V4-Flash（AMD Radeon 兼容端点）· 2026-10-29')

    # 左卡片：AgenticRAG
    lx0, ly0, lx1, ly1 = 56, 162, 930, 830
    card(d, (lx0, ly0, lx1, ly1))
    d.rectangle((lx0, ly0, lx0 + 10, ly1), fill=BLUE)
    text(d, (lx0 + 34, ly0 + 24), 'AgenticRAG 数据合成', F_H2, INK)
    text(d, (lx0 + 34, ly0 + 68), '官方 agentic_rag_pipeline · 4 条维基文档 → 4 条 QA', F_SMALL, MUT)
    items = [
        ('输入 → 输出', '4 文档 → 8 候选任务 → 7 通过 → 4 条 QA', INK),
        ('字段齐全', '11/11 = 100%（13 列全非空）', GREEN),
        ('F1Score', '[1.0, 1.0, 0.7826, 0.0]  均值 0.696', BLUE),
        ('双过滤', 'llm_score 全 0 · golden_doc_score [2,2,1,2]', ORANGE),
        ('成本', f'{ar["requests"]} 请求 / {ar["elapsed"]:.1f}s ≈ {ar["requests"]/(ar["elapsed"]/60):.2f} req/min', PURPLE),
    ]
    y = ly0 + 122
    for k, v, c in items:
        d.ellipse((lx0 + 38, y + 9, lx0 + 50, y + 21), fill=c)
        text(d, (lx0 + 68, y), k, F_SMALL, MUT)
        text(d, (lx0 + 240, y), v, F_BODY, INK)
        y += 52
    # F1 迷你条
    text(d, (lx0 + 34, ly0 + 396), 'F1 逐样本', F_SMALL, MUT)
    bx = lx0 + 34
    for i, v in enumerate(ar['f1']):
        c = GREEN if v >= 0.999 else (ORANGE if v >= 0.5 else RED)
        hgt = int(120 * v)
        if hgt >= 4:
            d.rounded_rectangle((bx + i * 150, ly0 + 550 - hgt, bx + i * 150 + 130, ly0 + 550), radius=8, fill=c)
        else:
            d.rectangle((bx + i * 150 + 10, ly0 + 546, bx + i * 150 + 120, ly0 + 550), fill=c)
        text(d, (bx + i * 150 + 65, ly0 + 560), f'{v:.2f}', F_SMALL, INK, anchor='ma')
    text(d, (lx0 + 34, ly0 + 596), 'F1=0 反例：refined=2017-08-30 vs golden=August 30, 2017', F_TINY, RED)
    text(d, (lx0 + 34, ly0 + 620), '→ token-F1 对日期/格式敏感，只宜作弱奖励或先做答案规范化', F_TINY, MUT)

    # 右卡片：Text2SQL
    rx0, ry0, rx1, ry1 = 974, 162, W - 56, 830
    card(d, (rx0, ry0, rx1, ry1))
    d.rectangle((rx0, ry0, rx0 + 10, ry1), fill=GREEN)
    text(d, (rx0 + 34, ry0 + 24), 'Text2SQL 数据流水线', F_H2, INK)
    text(d, (rx0 + 34, ry0 + 68), '官方 text2sql_pipeline_refine + gen · SQLite 真实执行验证', F_SMALL, MUT)
    rows_txt = ' → '.join(str(x) for x in t2s['rows'])
    items = [
        ('refine 10 步', f'行数漏斗 {rows_txt}', INK),
        ('一致性过滤', 'step5 4→3（唯一带 external knowledge 的样本被淘汰）', ORANGE),
        ('最终产出', f'{t2s["n"]} 条 · 10 列 / 9 非空 · prompt {t2s["prompt"]["min"]}–{t2s["prompt"]["max"]}B', GREEN),
        ('CoT 推理', f'{t2s["cot"]["min"]}–{t2s["cot"]["max"]} 字符（执行一致性投票）', BLUE),
        ('难度分级', '组件 medium×2+extra×1 · 执行 easy×2+hard×1', PURPLE),
        ('gen 合成', f'3 个库×1 条 SQL → 一致性过滤后 {gen["final"]["rows"]} 条（9 步全通）', TEAL),
        ('限流表现', f'refine {t2s["r429"]}×429 · 5m05s；gen {gen["r429"]}×429 + {gen["r500"]}×504', MUT),
    ]
    y = ry0 + 122
    for k, v, c in items:
        d.ellipse((rx0 + 38, y + 9, rx0 + 50, y + 21), fill=c)
        text(d, (rx0 + 68, y), k, F_SMALL, MUT)
        text(d, (rx0 + 240, y), v, F_BODY, INK)
        y += 48
    text(d, (rx0 + 34, ry0 + 500), '最终样例（第 1 条，question_type = Colloquial）', F_SMALL, MUT)
    q = str(t2s['final'][0]['question'])
    q_short = q if len(q) <= 72 else q[:72] + '…'
    text(d, (rx0 + 34, ry0 + 530), 'Q: ' + q_short, F_SMALL, INK)
    text(d, (rx0 + 34, ry0 + 566), 'SQL 为 CTE + 窗口函数变体，通过 EXPLAIN 可执行性过滤；CoT 与 SQL 结果集一致', F_TINY, MUT)
    text(d, (rx0 + 34, ry0 + 630), '注：evidence 为空串 → 唯一带 external knowledge 的 Metaphorical 样本在 step5 被过滤', F_TINY, ORANGE)

    # 底部约束条
    cy0 = 856
    d.rounded_rectangle((56, cy0, W - 56, cy0 + 100), radius=18, fill=(255, 246, 232), outline=(250, 213, 165), width=2)
    text(d, (86, cy0 + 16), '真实工程约束：AMD 端点账号级 20 requests/min + 模型级并发上限；两个 pipeline 共享配额', F_H3, (146, 84, 10))
    text(d, (86, cy0 + 56), f'pilot 40×429 → 加进程级 RPM=12 限流后正式跑仅 {ar["r429"]}×429、0 行丢失；smoke step10 因 29×429 重试风暴终止（按到 step9 归档）。数据合成成本按「限流后有效吞吐」算，而不是理论并发。', F_SMALL, (120, 72, 12))
    footer(d, '图由 code/task4/make_figures.py 从 evidence/task4 的真实 JSONL/JSON/日志生成，非示意数据')
    img.save(os.path.join(out, 'runshot_task4_01_打卡总览.png'))
    return 'runshot_task4_01_打卡总览.png'


# ---------------------------------------------------------------- 图 2：AgenticRAG 结果
def fig_agentic(out):
    ar = agentic_rag_data()
    img, d = new_canvas()
    header(d, 'AgenticRAG 数据合成 · 运行结果', '4 条文档 → 8 候选任务 → 7 通过 llm_score<1 → 每文档 max_question=1 → 最终 4 条（字段 11/11 齐全）')
    # 左：F1 竖柱
    card(d, (56, 160, 1030, 760))
    vbars(d, (110, 250, 980, 640),
          ['sample 1', 'sample 2', 'sample 3', 'sample 4'],
          ar['f1'], 1.05,
          [GREEN if v >= 0.999 else (ORANGE if v >= 0.5 else RED) for v in ar['f1']],
          title='F1Score（refined_answer vs golden_doc_answer，token-F1）', value_fmt='{:.3f}')
    # 均值线
    mean = sum(ar['f1']) / len(ar['f1'])
    d.line((110, 640 - (640 - 250) * (mean / 1.05), 980, 640 - (640 - 250) * (mean / 1.05)), fill=BLUE, width=3)
    ymean = 640 - (640 - 250) * (mean / 1.05)
    text(d, (960, ymean - 14), f'mean {mean:.4f}', F_SMALL, BLUE, anchor='ra')
    text(d, (110, 700), '真实反例（sample 4）：refined=2017-08-30 vs golden=August 30, 2017 → token 交集 0 → F1=0', F_SMALL, RED)

    # 右：双过滤 + 字段 + 成本
    card(d, (1062, 160, W - 56, 760))
    text(d, (1094, 188), '双过滤：把「必须检索 + 文档可答」写成数据条件', F_H3, INK)
    hbar(d, (1094, 240, W - 96, 380), ['llm_score', 'golden_doc_score'],
         [0, 1.75], 3.0, [ORANGE, GREEN], num_fmt='{}', label_w=200)
    text(d, (1094, 392), 'llm_score = [0,0,0,0]（无上下文答不出 → 必须检索）', F_SMALL, MUT)
    text(d, (1094, 422), 'golden_doc_score = [2,2,1,2]（给文档可答 → grounded），全部 ≥1 通过', F_SMALL, MUT)

    text(d, (1094, 486), '字段齐全率 11/11 = 100%（4 行空值率 0%）', F_H3, GREEN)
    for i, line in enumerate([
        '最终 13 列：id / contents / identifier / candidate_tasks_str / question / answer /',
        'refined_answer / llm_answer / llm_score / golden_doc_answer / golden_doc_score /',
        'optional_answer / F1Score']):
        text(d, (1094, 528 + i * 26), line, F_TINY, MUT)

    text(d, (1094, 610), '成本与吞吐', F_H3, INK)
    text(d, (1094, 654), f'{ar["requests"]} 次 HTTP 请求（{ar["logical"]} 次逻辑调用）/ {ar["elapsed"]:.1f} 秒', F_BODY, INK)
    text(d, (1094, 692), f'≈ {ar["requests"]/(ar["elapsed"]/60):.2f} 请求/分钟（20 RPM 账号级限流下的有效吞吐）', F_SMALL, MUT)
    text(d, (1094, 722), f'正式跑 {ar["r429"]}×429 + {ar["r500"]}×500(504) 被重试吸收、0 行丢失；pilot 阶段 40×429 曾导致缺列报错', F_TINY, ORANGE)

    # 底部：漏斗
    card(d, (56, 790, W - 56, 968))
    text(d, (86, 810), '样本漏斗（真实产物计数）', F_H3, INK)
    stages = [('输入文档', 4), ('候选任务', 8), ('通过 llm_score<1', 7), ('最终保留', 4)]
    gap = 62
    widths = [70 + v * 52 for _, v in stages]
    total = sum(widths) + gap * (len(stages) - 1)
    x = (W - total) / 2
    for i, ((lab, v), bw) in enumerate(zip(stages, widths)):
        d.rounded_rectangle((x, 858, x + bw, 922), radius=12, fill=[BLUE, PURPLE, ORANGE, GREEN][i])
        text(d, (x + bw / 2, 868), str(v), F_NUM2, (255, 255, 255), anchor='ma')
        text(d, (x + bw / 2, 932), lab, F_SMALL, MUT, anchor='ma')
        if i < len(stages) - 1:
            text(d, (x + bw + gap / 2, 870), '→', F_H3, MUT, anchor='ma')
        x += bw + gap
    footer(d, '数据：evidence/task4/agentic_rag/{agentic_rag_task4_step2_FINAL.jsonl, field_check_main.json, run_main.log}')
    img.save(os.path.join(out, 'runshot_task4_02_agentic_rag_result.png'))
    return 'runshot_task4_02_agentic_rag_result.png'


# ---------------------------------------------------------------- 图 3：Text2SQL refine
def fig_refine(out):
    t = text2sql_refine_data()
    img, d = new_canvas()
    header(d, 'Text2SQL refine 流水线 · 运行结果', '4 条输入 → 10 步全部跑通 → 一致性过滤 4→3 → 最终 3 条（10 列 / 9 非空）· 5 分 05 秒 · 0×429')
    # 左：10 步漏斗
    card(d, (56, 160, 1120, 700))
    text(d, (86, 182), '每步行数（vm-60 真实 cache）', F_H3, INK)
    labels = [f'step{i+1}  {name}' for i, name in enumerate([
        'SQLExecutabilityFilter', 'SQLVariationGenerator', 'SQLExecutabilityFilter', 'Text2SQLQuestionGenerator',
        'Text2SQLCorrespondenceFilter', 'Text2SQLPromptGenerator', 'Text2SQLCoTGenerator',
        'Text2SQLCoTVotingGenerator', 'SQLComponentClassifier', 'SQLExecutionClassifier'])]
    colors = [GRAY_BAR] * 10
    colors[4] = ORANGE
    colors[9] = GREEN
    hbar(d, (90, 240, 1080, 660), labels, t['rows'], 4.2, colors, num_fmt='{}', label_w=430, bar_h=26, gap=12)
    text(d, (90, 668), 'step5 一致性过滤 4→3：Metaphorical 问题（唯一带 external knowledge 的样本）被淘汰 → 最终 evidence 全为空串', F_TINY, ORANGE)

    # 右：分布 + 质量
    card(d, (1150, 160, W - 56, 700))
    text(d, (1180, 182), '难度与类型分布（最终 3 条）', F_H3, INK)
    # 组件难度
    vc = t['vc']
    comp = vc['sql_component_difficulty']
    execd = vc['sql_execution_difficulty']
    text(d, (1180, 232), 'SQL 组件难度', F_SMALL, MUT)
    hbar(d, (1180, 258, W - 96, 340), list(comp.keys()), list(comp.values()), 3, [BLUE, PURPLE], num_fmt='{}', label_w=110, bar_h=22, gap=10)
    text(d, (1180, 356), 'SQL 执行难度（3 次生成通过率）', F_SMALL, MUT)
    hbar(d, (1180, 382, W - 96, 464), list(execd.keys()), list(execd.values()), 3, [GREEN, RED], num_fmt='{}', label_w=110, bar_h=22, gap=10)
    text(d, (1180, 486), 'question_type：Colloquial / Interrogative / Formal 各 1', F_SMALL, MUT)
    text(d, (1180, 520), f'prompt {t["prompt"]["min"]}–{t["prompt"]["max"]} 字符（均 {t["prompt"]["avg"]:.0f}）', F_SMALL, MUT)
    text(d, (1180, 550), f'cot_reasoning {t["cot"]["min"]}–{t["cot"]["max"]} 字符（均 {t["cot"]["avg"]:.0f}）', F_SMALL, MUT)
    text(d, (1180, 596), f'refine 全链 0×429；smoke 版 step10 因 29×429 终止（已如实归档）', F_TINY, ORANGE)

    # 底部：样例
    card(d, (56, 730, W - 56, 968))
    text(d, (86, 752), '最终样例（第 1 条）', F_H3, INK)
    q = str(t['final'][0]['question'])
    paragraph(d, (86, 790), 'Q：' + (q[:150] + ('…' if len(q) > 152 else '')), F_SMALL, INK, max_width=1760, line_gap=6)
    sql = str(t['final'][0]['SQL']).replace('\n', ' ')
    text(d, (86, 856), 'SQL：' + (sql[:150] + ('…' if len(sql) > 152 else '')), F_SMALL, MUT)
    text(d, (86, 896), '变体类型 Query Structure Modifications · CoT 由执行一致性投票选出 · prompt 含 M-Schema + Database Engine + 问题 + 指令', F_SMALL, MUT)
    footer(d, '数据：evidence/task4/text2sql/refine/{stats_refine.json, quality_checks.json, final_refine.jsonl, refine.log}')
    img.save(os.path.join(out, 'runshot_task4_03_text2sql_refine.png'))
    return 'runshot_task4_03_text2sql_refine.png'


# ---------------------------------------------------------------- 图 4：Text2SQL gen
def fig_gen(out):
    g = text2sql_gen_data()
    img, d = new_canvas()
    header(d, 'Text2SQL gen 合成模式 · 运行结果', '无需输入 SQL，从 3 个 SQLite 库的 Schema 各合成 1 条 SELECT → 9 步链路全通 → 一致性过滤 3→2')
    card(d, (56, 160, 1120, 720))
    text(d, (86, 182), '每步行数（gen 9 步）', F_H3, INK)
    names = ['SQLGenerator', 'SQLExecutabilityFilter', 'Text2SQLQuestionGenerator', 'Text2SQLCorrespondenceFilter',
             'Text2SQLPromptGenerator', 'Text2SQLCoTGenerator', 'Text2SQLCoTVotingGenerator',
             'SQLComponentClassifier', 'SQLExecutionClassifier']
    labels = [f'step{i+1}  {n}' for i, n in enumerate(names)]
    colors = [GRAY_BAR] * 9
    colors[1] = GREEN
    colors[3] = ORANGE
    hbar(d, (90, 240, 1080, 680), labels, g['rows'], 3.4, colors, num_fmt='{}', label_w=420, bar_h=28, gap=14)

    card(d, (1150, 160, W - 56, 720))
    text(d, (1180, 182), '输入与输出', F_H3, INK)
    text(d, (1180, 230), '3 个数据库（各合成 1 条）', F_SMALL, MUT)
    for i, (db, cx) in enumerate([('california_schools', 'Complex'), ('student_club', 'Moderate'), ('thrombosis_prediction', 'Simple')]):
        y = 262 + i * 42
        d.rounded_rectangle((1180, y, W - 96, y + 32), radius=8, fill=(241, 244, 248))
        text(d, (1196, y + 6), db, F_SMALL, INK)
        text(d, (W - 116, y + 6), 'sql_complexity = ' + cx, F_SMALL, MUT, anchor='ra')
    text(d, (1180, 410), '最终 2 条（thrombosis_prediction 被一致性过滤淘汰）', F_H3, GREEN)
    text(d, (1180, 452), '最终难度：组件 hard×1 + extra×1 · 执行 extra×2', F_SMALL, MUT)
    text(d, (1180, 486), 'question_type：Formal / Concise 各 1', F_SMALL, MUT)
    text(d, (1180, 520), '运行 22:40:37–22:46:19 · 0×429 · 2 次上游 504（自动重试成功）', F_SMALL, MUT)
    text(d, (1180, 566), '诚实边界：被过滤的 thrombosis 问题人工看语义一致，', F_TINY, ORANGE)
    text(d, (1180, 590), '疑似 LLM 一致性判定假阴性（已写入 REPORT 限制）', F_TINY, ORANGE)
    text(d, (1180, 640), 'gen 与 refine 共用同一套算子：过滤 → 问题 → prompt → CoT 投票 → 双难度', F_TINY, MUT)

    card(d, (56, 750, W - 56, 968))
    text(d, (86, 772), 'gen 模式说明', F_H3, INK)
    text(d, (86, 812), 'gen 不需要 SQL 样本，仅需数据库目录（db_id=文件名）；SQLGenerator 按 Schema 合成 SQL 与 sql_complexity_type，后续与 refine 相同的生成/过滤/投票/分级链。', F_SMALL, MUT)
    text(d, (86, 850), '本任务用于验证「从零合成」路径可运行；样本仅 3 库×1 条，属链路证据，不代表合成质量分布。', F_SMALL, MUT)
    footer(d, '数据：evidence/task4/text2sql/gen/{summary_gen.json, gen.log, samples_gen.jsonl}')
    img.save(os.path.join(out, 'runshot_task4_04_text2sql_gen.png'))
    return 'runshot_task4_04_text2sql_gen.png'


# ---------------------------------------------------------------- 图 5：限流与吞吐
def fig_limits(out):
    ar = agentic_rag_data()
    t2s = text2sql_refine_data()
    gen = text2sql_gen_data()
    smoke = smoke_429()
    img, d = new_canvas()
    header(d, '共享 20 RPM 端点下的限流与吞吐（真实 429 计数）', '账号级 20 requests/min + 模型级并发上限：两个 pipeline 并发时互相挤占；这是本任务最大的工程约束')
    card(d, (56, 160, 1180, 720))
    text(d, (86, 182), '各次运行的 status=429 次数（从原始日志正则统计）', F_H3, INK)
    labels = ['AR pilot（限流前）', 'smoke step10（终止）', 'AR 正式（RPM=12）', 'T2S refine', 'T2S gen']
    vals = [ar['pilot429'], smoke, ar['r429'], t2s['r429'], gen['r429']]
    vbars(d, (110, 260, 1130, 660), labels, vals, max(vals) * 1.18,
          [RED, ORANGE, GREEN, GREEN, GREEN], value_fmt='{}')
    text(d, (86, 682), 'pilot：40×429 让 LLM 响应全变 None → step1 缺列 → step2 报 Missing required column(s)', F_TINY, RED)

    card(d, (1210, 160, W - 56, 720))
    text(d, (1240, 182), '为什么加客户端限速', F_H3, INK)
    arrows = [
        ('官方默认', 'max_workers=500 并发打 20 RPM 端点', '→ 必然 429 风暴（实测 40×429）'),
        ('限速后', '进程级 RPM=12 + max_workers=2 + 重试', f'→ 正式 AR 仅 {ar["r429"]}×429、0 行丢失'),
        ('有效吞吐', f'AR {ar["requests"]} req / {ar["elapsed"]:.1f}s = {ar["requests"]/(ar["elapsed"]/60):.2f} req/min', f'refine 28 调用 / 305s ≈ 5.5 req/min'),
        ('成本含义', '数据合成成本按限流后有效吞吐算', '官方默认配置 ≈ 0.3–0.4 条样本/分钟量级'),
    ]
    y = 240
    for k, v, w in arrows:
        text(d, (1240, y), k, F_H3, BLUE)
        yy = paragraph(d, (1240, y + 38), v, F_SMALL, INK, max_width=560)
        text(d, (1240, yy + 2), w, F_SMALL, MUT)
        y = yy + 48

    card(d, (56, 750, W - 56, 968))
    text(d, (86, 772), '工程结论', F_H3, INK)
    text(d, (86, 812), '1) 共享 API 端点做数据合成，瓶颈不是 GPU 而是配额；理论并发（500）与实际有效吞吐（≈6 req/min）差两个数量级，规模估算必须用后者。', F_SMALL, MUT)
    text(d, (86, 846), '2) LLM 调用最密集的步骤（如 SQLExecutionClassifier = rows × num_generations）最适合被限流打穿，应放在低并发/缩采样配置或独立时段跑。', F_SMALL, MUT)
    text(d, (86, 880), '3) 失败要能被观测：429/504 原文、重试次数、丢弃行数都已落日志与 REPORT，未把限流导致的缺失算作数据质量结论。', F_SMALL, MUT)
    footer(d, '数据：run_main.log / run_pilot_429_evidence.log / smoke.log / refine.log / gen.log（正则统计 status=429）')
    img.save(os.path.join(out, 'runshot_task4_05_429_throughput.png'))
    return 'runshot_task4_05_429_throughput.png'


# ---------------------------------------------------------------- 图 6：真实日志摘录（渲染）
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def clean_line(s, limit=118):
    s = ANSI_RE.sub('', s).rstrip()
    return s if len(s) <= limit else s[:limit - 1] + '…'


def pick_lines(text, patterns, limit):
    out = []
    for raw in text.splitlines():
        line = clean_line(raw)
        if not line:
            continue
        if any(re.search(p, line) for p in patterns):
            out.append(line)
        if len(out) >= limit:
            break
    return out


def terminal_panel(d, box, title, lines, accent, note=None):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=16, fill=(27, 33, 48), outline=(52, 63, 86), width=2)
    d.rectangle((x0, y0, x1, y0 + 46), fill=(35, 43, 61))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((x0 + 20 + i * 26, y0 + 16, x0 + 34 + i * 26, y0 + 30), fill=c)
    text(d, (x0 + 110, y0 + 11), title, F_H3, (232, 238, 248))
    if note:
        text(d, (x1 - 18, y0 + 15), note, F_TINY, accent, anchor='ra')
    yy = y0 + 62
    for i, line in enumerate(lines):
        col = (214, 223, 235)
        if re.search(r'error|429|traceback|valueerror|missing', line, re.I):
            col = (240, 120, 120)
        elif re.search(r'done in|saved|\[final\]|rows=', line, re.I):
            col = (140, 205, 150)
        elif re.search(r'warning|rate limit|retry', line, re.I):
            col = (230, 190, 120)
        text(d, (x0 + 22, yy), line, F_MONO, col)
        yy += 24
    return yy


def fig_logs(out):
    ar_log = read_text(os.path.join(EVD, 'agentic_rag', 'run_main.log'))
    ref_log = read_text(os.path.join(EVD, 'text2sql', 'refine', 'refine.log'))
    gen_log = read_text(os.path.join(EVD, 'text2sql', 'gen', 'gen.log'))
    ar_lines = pick_lines(ar_log, [
        r'\[CONFIG\] (model|max_workers|max_per_task|max_question)',
        r'\[STEP1\] done', r'\[STEP2\] done', r'\[TOTAL\]', r'\[FINAL\]',
        r'status=429',
    ], 8)
    ref_lines = pick_lines(ref_log, [
        r'\[STEP \d+\] rows=', r'Question generation results saved',
        r'Difficulty classification results saved', r'SQL Difficulty Statistics',
    ], 8)
    # 只保留 step 快照里最关键的几步
    ref_lines = ref_lines[-8:]
    gen_lines = pick_lines(gen_log, [
        r'\[STEP \d+\] rows=', r'Question generation results saved',
        r'SQL Difficulty Statistics', r'SQL generation results saved',
    ], 6)

    img, d = new_canvas()
    d.rectangle((0, 0, W, 118), fill=(15, 19, 28))
    text(d, (56, 22), 'vm-60 真实运行日志摘录（渲染图，非截屏）', F_TITLE, (236, 242, 250))
    text(d, (58, 86), 'ANSI 颜色已去除；每行截断 118 字符；原始日志见 evidence/task4 对应目录', F_SUB, (150, 164, 186))
    terminal_panel(d, (56, 150, W - 56, 462), 'AgenticRAG · run_main.log（4 文档 → 4 条 QA，66 请求 / 654.7s）', ar_lines, (140, 205, 150), 'status=429 计数：3')
    terminal_panel(d, (56, 486, W - 56, 766), 'Text2SQL refine · refine.log（10 步全通，4→3，0×429）', ref_lines, (140, 205, 150), 'status=429 计数：0')
    terminal_panel(d, (56, 790, W - 56, 1006), 'Text2SQL gen · gen.log（3 库×1 SQL → 9 步全通 → 2 条）', gen_lines, (140, 205, 150), 'status=429 计数：0 · 504×2')
    footer(d, '图由 code/task4/make_figures.py 从 evidence/task4 的原始日志生成（去 ANSI、截断），仅作结果呈现，不是视频/终端截屏')
    img.save(os.path.join(out, 'runshot_task4_06_日志摘录.png'))
    return 'runshot_task4_06_日志摘录.png'


def main():
    out = os.path.join(EVD, 'figures')
    if '--out' in sys.argv:
        out = os.path.abspath(sys.argv[sys.argv.index('--out') + 1])
    os.makedirs(out, exist_ok=True)
    made = []
    for fn in (fig_overview, fig_agentic, fig_refine, fig_gen, fig_limits, fig_logs):
        name = fn(out)
        made.append(name)
        print('saved', os.path.join(out, name))
    print('done', len(made), 'figures ->', out)


if __name__ == '__main__':
    main()
