"""把 trajectory.jsonl 画成优化过程图。

输出：
- rounds_overview.png   延迟 / 改进比例 / 接受标记
- status_breakdown.png  状态分布
- process_timeline.png  每轮改动时间线（文字标注）
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _status_key(row: dict[str, Any]) -> str:
    if row.get("accepted"):
        return "accepted"
    status = str(row.get("status") or "unknown")
    reason = str(row.get("reason") or "")
    if status == "compile_error" or "compile_error" in reason:
        return "compile_error"
    if "below_threshold" in reason or status == "below_threshold":
        return "below_threshold"
    if status != "ok":
        return status
    return "rejected"


def _setup_cjk_font() -> None:
    """尽量启用本机中文字体，避免 timeline 中文缺字。"""
    from matplotlib import font_manager, rcParams

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return
    # 常见路径兜底
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ):
        p = Path(path)
        if p.is_file():
            font_manager.fontManager.addfont(str(p))
            # TTC 可能含多 face；优先挑 CJK SC / CN
            matched = None
            for f in font_manager.fontManager.ttflist:
                if str(p) in getattr(f, "fname", "") and (
                    "CJK SC" in f.name or "CN" in f.name or "Micro Hei" in f.name or "UMing" in f.name
                ):
                    matched = f.name
                    break
            name = matched or font_manager.FontProperties(fname=str(p)).get_name()
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return
    rcParams["axes.unicode_minus"] = False


def render_visualizations(
    workspace: Path,
    out_dir: Path | None = None,
    threshold: float = 0.01,
    title: str = "Kernel Optimize Agent · vector_add",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    _setup_cjk_font()

    traj_path = workspace / "trajectory.jsonl"
    rows = load_trajectory(traj_path)
    out_dir = out_dir or (workspace / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        empty = out_dir / "EMPTY.txt"
        empty.write_text("trajectory.jsonl 为空，无法绘图。\n", encoding="utf-8")
        return [empty]

    rounds = list(range(1, len(rows) + 1))
    latencies = [r.get("latencyMs") for r in rows]
    improvements = [r.get("improvementFraction") for r in rows]
    statuses = [_status_key(r) for r in rows]
    changes = [str(r.get("change") or "") for r in rows]

    color_map = {
        "accepted": "#2ca02c",
        "below_threshold": "#ff7f0e",
        "compile_error": "#d62728",
        "rejected": "#7f7f7f",
    }
    colors = [color_map.get(s, "#1f77b4") for s in statuses]

    saved: list[Path] = []

    # ── 1) 总览：延迟 + 改进 ──────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax0 = axes[0]
    xs_lat = [i for i, v in zip(rounds, latencies) if v is not None]
    ys_lat = [v for v in latencies if v is not None]
    cs_lat = [c for c, v in zip(colors, latencies) if v is not None]
    if ys_lat:
        ax0.plot(xs_lat, ys_lat, color="#4c78a8", alpha=0.35, linewidth=1.5, zorder=1)
        ax0.scatter(xs_lat, ys_lat, c=cs_lat, s=70, zorder=2, edgecolors="white", linewidths=0.6)
        # best-so-far（仅已接受或可比较延迟）
        best = None
        best_x, best_y = [], []
        for i, v in zip(rounds, latencies):
            if v is None:
                continue
            if best is None or v < best:
                best = v
            best_x.append(i)
            best_y.append(best)
        ax0.plot(best_x, best_y, color="#54a24b", linewidth=2, label="best-so-far latency", zorder=3)
    else:
        ax0.text(0.5, 0.5, "无有效 latencyMs", ha="center", va="center", transform=ax0.transAxes)
    ax0.set_ylabel("latency (ms)")
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    xs_imp = [i for i, v in zip(rounds, improvements) if v is not None]
    ys_imp = [float(v) * 100 for v in improvements if v is not None]
    cs_imp = [c for c, v in zip(colors, improvements) if v is not None]
    if ys_imp:
        ax1.axhline(threshold * 100, color="#e45756", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.0%}")
        ax1.axhline(0, color="#999999", linewidth=0.8)
        ax1.bar(xs_imp, ys_imp, color=cs_imp, width=0.7, alpha=0.9)
    else:
        ax1.text(0.5, 0.5, "无有效 improvementFraction", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("round")
    ax1.set_ylabel("paired improvement (%)")
    ax1.grid(True, axis="y", alpha=0.25)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["accepted"], markersize=9, label="accepted"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["below_threshold"], markersize=9, label="below_threshold"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map["compile_error"], markersize=9, label="compile_error"),
        Line2D([0], [0], color="#54a24b", linewidth=2, label="best-so-far latency"),
        Line2D([0], [0], color="#e45756", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.0%}"),
    ]
    ax0.legend(handles=legend_handles[:4], loc="upper right")
    ax1.legend(handles=legend_handles[4:], loc="upper right")

    fig.tight_layout()
    p1 = out_dir / "rounds_overview.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # ── 2) 状态分布（饼图 + 条形，中文标签）────────────────────────
    from collections import Counter

    status_labels_zh = {
        "accepted": "接受",
        "below_threshold": "未达阈值",
        "compile_error": "编译失败",
        "rejected": "拒绝",
        "unknown": "未知",
    }
    status_order = ["accepted", "below_threshold", "compile_error", "rejected", "unknown"]
    counts = Counter(statuses)
    ordered = [k for k in status_order if counts.get(k, 0) > 0]
    ordered += [k for k in counts if k not in ordered]
    labels_zh = [status_labels_zh.get(k, k) for k in ordered]
    values = [counts[k] for k in ordered]
    bar_colors = [color_map.get(k, "#1f77b4") for k in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle(f"{title}\n状态分布（n={len(rows)}）", fontsize=13, fontweight="bold")

    # 饼图
    ax_pie = axes[0]
    wedges, texts, autotexts = ax_pie.pie(
        values,
        labels=labels_zh,
        colors=bar_colors,
        autopct=lambda pct: f"{pct:.0f}%\n({int(round(pct / 100.0 * len(rows)))})",
        startangle=90,
        pctdistance=0.65,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    for t in texts:
        t.set_fontsize(10)
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("#222222")
    ax_pie.set_title("占比", fontsize=11)

    # 条形图
    ax_bar = axes[1]
    bars = ax_bar.bar(labels_zh, values, color=bar_colors, width=0.55)
    ax_bar.set_ylabel("次数")
    ax_bar.set_title("计数", fontsize=11)
    ax_bar.set_ylim(0, max(values) * 1.25 if values else 1)
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax_bar.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    p2 = out_dir / "status_breakdown.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # ── 3) 时间线（每轮改动） ────────────────────────────────────
    fig_h = max(4.5, 0.55 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 1)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"{title}\noptimization timeline", fontsize=13, fontweight="bold", pad=12)

    for idx, (round_i, status, change, imp, lat) in enumerate(
        zip(rounds, statuses, changes, improvements, latencies), start=1
    ):
        y = idx
        color = color_map.get(status, "#1f77b4")
        box = FancyBboxPatch(
            (0.3, y - 0.35),
            9.3,
            0.7,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
        )
        ax.add_patch(box)
        ax.plot(0.55, y, "o", color=color, markersize=10)
        meta = []
        if lat is not None:
            meta.append(f"{lat:.4f} ms")
        if imp is not None:
            meta.append(f"Δ {float(imp) * 100:+.2f}%")
        meta.append(status)
        wrapped = textwrap.fill(change or "(no change note)", width=70)
        ax.text(0.85, y - 0.08, f"R{round_i}  {' · '.join(meta)}", fontsize=9, fontweight="bold", va="center")
        ax.text(0.85, y + 0.22, wrapped, fontsize=8, color="#333333", va="center")

    fig.tight_layout()
    p3 = out_dir / "process_timeline.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    return saved


def print_trajectory_table(rows: list[dict[str, Any]], threshold: float = 0.01) -> None:
    print("\n═══ 优化轨迹表 ═══")
    print(f"{'#':>3}  {'acc':^3}  {'status':<16}  {'lat(ms)':>10}  {'imp%':>8}  change")
    print("-" * 100)
    for i, row in enumerate(rows, 1):
        acc = "✓" if row.get("accepted") else "✗"
        status = _status_key(row)
        lat = row.get("latencyMs")
        imp = row.get("improvementFraction")
        lat_s = f"{lat:.4f}" if lat is not None else "-"
        imp_s = f"{float(imp) * 100:+.2f}" if imp is not None else "-"
        change = (row.get("change") or "")[:48]
        print(f"{i:>3}  {acc:^3}  {status:<16}  {lat_s:>10}  {imp_s:>8}  {change}")
    accepted = sum(1 for r in rows if r.get("accepted"))
    print("-" * 100)
    print(f"轮次={len(rows)}  接受={accepted}  阈值={threshold:.0%}")

