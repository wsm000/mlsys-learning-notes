# -*- coding: utf-8 -*-
"""Task3 打卡截图公共组件：中文字体、页面头、卡片渲染。"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_CANDIDATES = [
    "/home/simin/task1_infer/fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]
FAMILY_ORDER = ["SimHei", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                 "Noto Sans CJK SC", "Noto Sans CJK JP", "AR PL UMing CN"]


def setup_fonts():
    """显式加载可用的中文字体（vm-60 上 CJK 字体多为 .ttc/.otf，自动扫描会漏掉）。"""
    for path in FONT_CANDIDATES:
        if not glob.glob(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass
    cjk = None
    for name in FAMILY_ORDER:
        if any(x.name == name for x in font_manager.fontManager.ttflist):
            cjk = name
            break
    plt.rcParams["font.family"] = [cjk or "sans-serif", "DejaVu Sans"]
    plt.rcParams["font.sans-serif"] = [cjk or "sans-serif", "DejaVu Sans"]
    plt.rcParams["font.monospace"] = [cjk or "sans-serif", "DejaVu Sans Mono"]
    plt.rcParams["axes.unicode_minus"] = False
    return cjk


CJK_FONT = setup_fonts()  # 导入即生效，避免调用方忘记初始化


def page(title, env_line, link_line, nrows=3, ncols=2, figsize=(13.5, 13.8)):
    fig = plt.figure(figsize=figsize, facecolor="#f5f7fb")
    gs = fig.add_gridspec(nrows, ncols, hspace=0.30, wspace=0.14,
                          top=0.845, bottom=0.045, left=0.035, right=0.965)
    fig.text(0.5, 0.978, title, ha="center", va="center",
             fontsize=15.5, color="#16264a")
    fig.text(0.5, 0.945, env_line, ha="center", va="center", fontsize=10.0, color="#3a4a6a")
    fig.text(0.5, 0.916, link_line, ha="center", va="center", fontsize=8.6, color="#5a6a8a")
    return fig, gs


def card(ax, title, lines, title_color="#16264a", body_size=8.0, title_size=10.6):
    if not hasattr(ax, "set_xticks"):
        fig = getattr(ax.get_gridspec(), "figure", None)
        if fig is None:
            raise RuntimeError("card() 需要 Axes 或带 figure 的 SubplotSpec")
        ax = fig.add_subplot(ax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c3cee6")
    ax.set_facecolor("white")
    ax.text(0.028, 0.945, title, transform=ax.transAxes, fontsize=title_size,
            color=title_color, va="top")
    ax.text(0.028, 0.865, "\n".join(lines), transform=ax.transAxes, fontsize=body_size,
            color="#22304f", va="top", family="monospace", linespacing=1.34)
    return ax


def save(fig, path):
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print("saved:", path)
