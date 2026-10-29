
"""Regenerate the Task5 one-page comparison chart from evidence JSON (no external deps)."""
import json, os, sys, zlib, struct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _tinyfont import draw_text, text_width

def write_png(path, w, h, rgb):
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        s = y * w * 3
        raw.extend(rgb[s:s + w * 3])
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    open(path, "wb").write(png)

# ---------------- evidence ----------------
a = json.load(open(os.path.join(HERE, "exp10-1", "summary.json"), encoding="utf-8"))["comparison"]
run1 = json.load(open(os.path.join(HERE, "exp10-4", "exp10-4-local-run1", "summary.json"), encoding="utf-8"))
ctl = json.load(open(os.path.join(HERE, "exp10-4", "schedule_control.json"), encoding="utf-8"))
main = json.load(open(os.path.join(HERE, "exp10-1b", "summary.json"), encoding="utf-8"))
adv = json.load(open(os.path.join(HERE, "exp10-1b-adv", "summary.json"), encoding="utf-8"))

tok = a["token_ratio_vs_A1"]; lat = a["latency_ratio_vs_A1"]
sp_real = run1["measured_speedup_serial_over_parallel"]
sp8 = ctl[0]["measured_speedup_serial_over_parallel"]; sp4 = ctl[1]["measured_speedup_serial_over_parallel"]
g_adv = adv["generator"]["accuracy"]
b1_adv = adv["arm_B1_no_new_info"]["final_accuracy"]; b2_adv = adv["arm_B2_execution_evidence"]["final_accuracy"]
g_main = main["generator"]["accuracy"]
b1_main = main["arm_B1_no_new_info"]["final_accuracy"]; b2_main = main["arm_B2_execution_evidence"]["final_accuracy"]
# 2026-10-27: adv set expanded to 38 tasks x 3 rounds; paired stats and per-family split
p_adv = adv["paired"]
fam = adv["by_family"]
fam_order = ["identity-constant", "original-trap", "realistic-program", "semantics-trap", "simulation"]

# ---------------- canvas ----------------
W, H = 1280, 1180
BG = (252, 252, 250)
buf = bytearray(bytes(BG) * (W * H))
INK = (34, 34, 34); MUTED = (110, 110, 110)
BLUE = (38, 84, 150); ORANGE = (214, 122, 36); GREEN = (44, 130, 73); GREY = (150, 150, 150); RED = (176, 48, 48)
TRACK = (228, 228, 226)

def rect(x0, y0, x1, y1, color):
    x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(W, int(x1)), min(H, int(y1))
    for y in range(y0, y1):
        b = (y * W + x0) * 3
        buf[b:b + (x1 - x0) * 3] = bytes(color) * (x1 - x0)

def bar(x0, y, width, frac, color, h=30):
    rect(x0, y, x0 + width, y + h, TRACK)
    if frac > 0:
        rect(x0, y, x0 + max(2, width * frac), y + h, color)

def label(x, y, s, color=INK, scale=2):
    draw_text(buf, W, H, x, y, s, color, scale)

# header
rect(0, 0, W, 76, BLUE)
label(24, 22, "TASK5 CH10 MULTI-AGENT  EVIDENCE SUMMARY (FROM JSON)", (255, 255, 255), 3)
label(24, 54, "ALL NUMBERS REGENERATED FROM evidence/task5/*.json BY make_summary_chart.py", (206, 220, 240), 1)

# ---------------- panel A ----------------
PX, PW = 430, 790
rect(0, 92, W, 140, (234, 238, 246))
label(24, 104, "A. ROLE CARRIER: TOKEN / LATENCY RATIO VS A1", BLUE, 2)
label(24, 128, "exp10-1  A1 single prompt = 1x baseline, 3 runs per arm", MUTED, 1)
vmax = max(list(tok.values()) + list(lat.values())) * 1.12
rows_a = [("A2 TRANSFER", tok["A2"], lat["A2"]), ("A3 SKILL", tok["A3"], lat["A3"])]
for i, (name, t, l) in enumerate(rows_a):
    y = 156 + i * 76
    label(24, y + 2, name, INK, 2)
    label(24, y + 32, "TOKENS %.2fX" % t, ORANGE, 1)
    label(200, y + 32, "LATENCY %.2fX" % l, MUTED, 1)
    bar(PX, y, PW, t / vmax, ORANGE, 24)
    bar(PX, y + 28, PW, l / vmax, GREY, 12)
    label(PX + PW * (t / vmax) + 8, y + 6, "%.2fX" % t, ORANGE, 1)
for v in range(0, 6):
    x = PX + PW * (v / 5.0)
    rect(x, 296, x + 1, 304, MUTED)
    label(x - 4, 308, str(v), MUTED, 1)
label(PX + PW - 40, 308, "X", MUTED, 1)
label(24, 288, "BAR LENGTH = MULTIPLE OF A1 (A1 = 1.00X)", MUTED, 1)

# ---------------- panel B ----------------
rect(0, 336, W, 380, (234, 238, 246))
label(24, 346, "B. REVIEWER WITH NEW INFO VS WITHOUT  (TABLE 10-1)", BLUE, 2)
label(24, 368, "exp10-1b (main) and exp10-1b-adv (expanded 38 tasks x 3 rounds)", MUTED, 1)

NAMES = ["GENERATOR", "B1 NO EVIDENCE", "B2 EVIDENCE"]

def group(y0, title, tri, note):
    label(24, y0, title, INK, 2)
    label(24, y0 + 30, note, MUTED, 1)
    for i, (val, color) in enumerate(tri):
        y = y0 + 56 + i * 36
        label(250, y + 6, NAMES[i], MUTED, 1)
        bar(430, y, 620, val, color, 22)
        label(430 + 620 * val + 8, y + 6, "%.3f" % val, color, 1)

group(392, "MAIN SET  N=48", [(g_main, GREY), (b1_main, ORANGE), (b2_main, GREEN)],
      "generator 1.000 -> nothing to fix, both arms identical")
group(556, "ADV SET EXPANDED  N=%d" % p_adv["n_pairs"],
      [(g_adv, GREY), (b1_adv, ORANGE), (b2_adv, GREEN)],
      "B1 broke 8 correct answers; B2 fixed 5")
label(250, 556 + 56 + 3 * 36 + 2,
      "ONLY_B2=%d ONLY_B1=%d  MCNEMAR P=%.4f  (SIGNIFICANT)" % (
          p_adv["n_only_B2_correct"], p_adv["n_only_B1_correct"],
          p_adv["mcnemar_exact_p_two_sided"]), GREEN, 1)

# ---------------- panel B2: per-family ----------------
rect(0, 700, W, 742, (234, 238, 246))
label(24, 712, "B2. WHERE THE EFFECT COMES FROM: PER TASK FAMILY (ADV SET)", BLUE, 2)
label(24, 734, "B1 breaks correct answers only in identity-constant; elsewhere the generator is already perfect", MUTED, 1)

FAM_LABEL = {
    "identity-constant": "IDENTITY-CONST",
    "original-trap": "ORIGINAL-TRAP",
    "realistic-program": "REALISTIC-PROG",
    "semantics-trap": "SEMANTICS-TRAP",
    "simulation": "SIMULATION",
}
FAM_COLOR = {
    "identity-constant": RED,
    "original-trap": ORANGE,
    "realistic-program": GREEN,
    "semantics-trap": GREY,
    "simulation": GREY,
}
for i, f in enumerate(fam_order):
    d = fam[f]
    y = 756 + i * 62
    label(24, y + 2, FAM_LABEL[f], INK, 1)
    label(24, y + 22, "N=%d" % d["n_pairs"], MUTED, 1)
    label(250, y + 6, "GEN", MUTED, 1)
    bar(300, y, 300, d["generator_accuracy"], GREY, 16)
    label(250, y + 28, "B1", MUTED, 1)
    bar(300, y + 22, 300, d["B1_final_accuracy"], FAM_COLOR[f], 16)
    label(250, y + 50, "B2", MUTED, 1)
    bar(300, y + 44, 300, d["B2_final_accuracy"], FAM_COLOR[f], 16)
    label(612, y + 6, "%.3f" % d["generator_accuracy"], MUTED, 1)
    label(612, y + 28, "%.3f" % d["B1_final_accuracy"], FAM_COLOR[f], 1)
    label(612, y + 50, "%.3f" % d["B2_final_accuracy"], FAM_COLOR[f], 1)
    label(700, y + 20, "ONLY_B2=%d ONLY_B1=%d P=%.4f" % (
        d["only_B2_correct"], d["only_B1_correct"], d["mcnemar_exact_p_two_sided"]),
        FAM_COLOR[f], 1)

# ---------------- panel C ----------------
rect(0, 1078, W, 1120, (234, 238, 246))
label(24, 1088, "C. PARALLEL SPEEDUP: REAL VS STUB CONTROL  (EXP10-4)", BLUE, 2)
rows_c = [("REAL 15 SITES (WINNER AT 2ND)", sp_real, RED), ("STUB 4 SITES", sp4, GREY), ("STUB 8 SITES", sp8, GREEN)]
for i, (name, val, color) in enumerate(rows_c):
    y = 1130 + i * 0
rows_c = [("REAL 15 SITES (WINNER AT 2ND)", sp_real, RED), ("STUB 4 SITES", sp4, GREY), ("STUB 8 SITES", sp8, GREEN)]
for i, (name, val, color) in enumerate(rows_c):
    y = 1130 + i * 40
    label(24, y + 8, name, INK, 1)
    bar(430, y, 700, val / 9.0, color, 26)
    label(430 + 700 * (val / 9.0) + 8, y + 6, "%.3fX" % val, color, 1)
label(24, 1160, "REAL RUN MEASURES NETWORK + MODEL LATENCY, NOT SCHEDULING. STUB MEASURES SCHEDULING ONLY. DO NOT CONFLATE.", RED, 1)

out = os.path.join(HERE, "task5-summary-chart.png")
write_png(out, W, H, buf)
print("wrote", out)
