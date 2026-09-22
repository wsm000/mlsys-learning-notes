# -*- coding: utf-8 -*-
"""把实跑日志渲染成终端风格打卡截图（深色主题，样式与 Task 0/1/2 的 runshot 一致）。

用法:
  python make_runshot.py <log.txt> <out.png> [标题] [提示符cwd] [命令行]

依赖:  Pillow（.venv-task3 已装有 numpy/pytest/pillow）。
"""
import sys
from PIL import Image, ImageDraw, ImageFont

FS = 18            # 字号
LINE_H = 28        # 行高
PAD_X = 30         # 水平内边距
PAD_TOP = 18
PAD_BOT = 26
TITLE_H = 44

CONSOLE = 'C:/Windows/Fonts/consola.ttf'
CONSOLE_B = 'C:/Windows/Fonts/consolab.ttf'
EMOJI_FONT = 'C:/Windows/Fonts/seguiemj.ttf'
CJK = 'C:/Windows/Fonts/msyh.ttc'          # Consolas 无中文字形, 中文/全角字符走雅黑
CJK_B = 'C:/Windows/Fonts/msyhbd.ttc'

BG = (30, 30, 30)
TITLE_BG = (45, 45, 48)
BORDER = (60, 60, 64)
TEXT = (204, 204, 204)
GREEN = (78, 201, 176)
YELLOW = (220, 220, 170)
BLUE = (156, 220, 254)
DIM = (110, 110, 110)
WHITE = (255, 255, 255)
PROMPT_PATH = (156, 220, 254)
PROMPT_PS = (106, 153, 85)

EMOJI_CHARS = set('\u2705\U0001F9EA\U0001F389\U0001F4CA\U0001F3C3\u2728\u26A0\u274C\U0001F525\U0001F680\U0001F4A1\u2714\u2716\U0001F4DD')

font = ImageFont.truetype(CONSOLE, FS)
font_b = ImageFont.truetype(CONSOLE_B, FS)
font_emoji = ImageFont.truetype(EMOJI_FONT, FS)
font_cjk = ImageFont.truetype(CJK, FS)
font_cjk_b = ImageFont.truetype(CJK_B, FS)

_probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
CHAR_W = _probe.textlength('M', font=font)


def cell_len(text):
    n = 0
    for ch in text:
        if ch == '\uFE0F':
            continue
        if ch in EMOJI_CHARS:
            n += 2
        elif ord(ch) > 0x2E7F:      # CJK / 全角标点按两个等宽格计
            n += 2
        else:
            n += 1
    return n


def draw_run(d, x, y, text, color, bold=False):
    f = font_b if bold else font
    fc = font_cjk_b if bold else font_cjk
    cx = x
    for ch in text:
        if ch == '\uFE0F':
            continue
        if ch in EMOJI_CHARS:
            d.text((cx, y), ch, font=font_emoji, embedded_color=True)
            cx += CHAR_W * 2
        elif ord(ch) > 0x2E7F:
            d.text((cx, y), ch, font=fc, fill=color)
            cx += CHAR_W * 2
        else:
            d.text((cx, y), ch, font=f, fill=color)
            cx += _probe.textlength(ch, font=f)
    return cx


def line_style(line):
    s = line.strip()
    if s.startswith('\U0001F389') or 'ALL TESTS PASSED' in s or s.startswith('ALL DEMO'):
        return GREEN, True
    if s.startswith('\u2705') or ' PASSED' in s or 'passed' in s.lower() and 'skipped' in s.lower() or s.strip().endswith('passed in'):
        return GREEN, False
    if 'PASS ->' in s or 'max abs err' in s and 'FAIL' not in s:
        return GREEN, False
    if s.startswith('PS>') or s.startswith('$ '):
        return BLUE, True
    if s.startswith('===') or s.startswith('---'):
        return DIM, False
    if s.startswith('[') and s.rstrip().endswith(']'):
        return YELLOW, True
    if 'FAIL' in s or 'Error' in s or 'Traceback' in s:
        return (230, 120, 120), True
    return TEXT, False


def read_log(txt_path):
    """读取日志: 兼容 UTF-8 / UTF-8-BOM / UTF-16(PowerShell Tee-Object 默认输出)。"""
    raw = open(txt_path, 'rb').read()
    if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        text = raw.decode('utf-16')
    elif raw.startswith(b'\xef\xbb\xbf'):
        text = raw.decode('utf-8-sig')
    else:
        text = raw.decode('utf-8', errors='replace')
    return text.splitlines()


def render(txt_path, png_path, title, prompt_cwd, cmd, meta=''):
    lines = [l.rstrip() for l in read_log(txt_path)]
    while lines and not lines[-1].strip():
        lines.pop()

    body = [('prompt', prompt_cwd + '> '), ('cmd', cmd)] + [('out', l) for l in lines]
    max_cells = max(cell_len(t) for _, t in body) + 2
    win_w = int(PAD_X * 2 + max_cells * CHAR_W)
    if meta:
        win_w = max(win_w, int(cell_len(meta) * CHAR_W + PAD_X * 2))
    win_w = min(win_w, 2400)
    meta_h = LINE_H if meta else 0
    body_h = PAD_TOP + LINE_H * len(body) + PAD_BOT
    win_h = TITLE_H + meta_h + body_h

    img = Image.new('RGB', (win_w, win_h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, win_w, TITLE_H], fill=TITLE_BG)
    d.line([0, TITLE_H, win_w, TITLE_H], fill=BORDER, width=1)
    # 元信息行: 补齐打卡三要素(GitHub ID / 微信群昵称)与运行环境
    if meta:
        d.rectangle([0, TITLE_H, win_w, TITLE_H + LINE_H], fill=(38, 38, 42))
        d.line([0, TITLE_H + LINE_H, win_w, TITLE_H + LINE_H], fill=BORDER, width=1)
        draw_run(d, PAD_X, TITLE_H + 4, meta, DIM)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 22 + i * 22
        d.ellipse([cx - 6, TITLE_H // 2 - 6, cx + 6, TITLE_H // 2 + 6], fill=c)
    tw = cell_len(title) * CHAR_W
    draw_run(d, max((win_w - tw) / 2, 120), TITLE_H // 2 - FS * 0.65, title, (215, 215, 215), True)

    y = TITLE_H + meta_h + PAD_TOP
    for kind, text in body:
        if kind == 'prompt':
            d.text((PAD_X, y), 'PS ', font=font, fill=PROMPT_PS)
            w = _probe.textlength('PS ', font=font)
            d.text((PAD_X + w, y), text, font=font, fill=PROMPT_PATH)
        elif kind == 'cmd':
            x0 = PAD_X + _probe.textlength('PS ' + prompt_cwd + '> ', font=font)
            draw_run(d, x0, y, text, WHITE, True)
        else:
            color, bold = line_style(text)
            draw_run(d, PAD_X, y, text, color, bold)
        y += LINE_H

    d.rectangle([0, 0, win_w - 1, win_h - 1], outline=BORDER, width=1)
    img.save(png_path)
    print(png_path, img.size)


if __name__ == '__main__':
    txt, png = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else '从Tensor到Transformer——构建 GPT 模型 | Task 3 自动微分引擎'
    cwd = sys.argv[4] if len(sys.argv) > 4 else 'C:\\...\\_tmp_tt_src\\tinytorch'
    cmd = sys.argv[5] if len(sys.argv) > 5 else ''
    meta = sys.argv[6] if len(sys.argv) > 6 else ''
    render(txt, png, title, cwd, cmd, meta)
