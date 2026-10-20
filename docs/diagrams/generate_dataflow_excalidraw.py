"""Generate docs/diagrams/glucosight-dataflow.excalidraw.

A diagram-first (not text-first) view of the current app: the two serving
modes, the live request flow through CV/NLP/glucose-source, and which
dataset actually has which modality. Icons + arrows + a small matrix table,
short labels only -- prose lives in docs/INTEGRATION_AUDIT.md and the chat
recap, not here.

Regenerate after the next architecture change:
    python docs/diagrams/generate_dataflow_excalidraw.py
"""
import json
import random
import time

OUT = "docs/diagrams/glucosight-dataflow.excalidraw"

GREEN = ("#2f9e44", "#ebfbee")
ORANGE = ("#e8590c", "#fff4e6")
BLUE = ("#1971c2", "#e7f5ff")
GRAY = ("#868e96", "#f1f3f5")
RED = ("#e03131", "#fff5f5")
DARK = "#1e1e1e"

elements = []
_id = 0


def nid():
    global _id
    _id += 1
    return f"el{_id}"


def _base(extra):
    return {
        "id": nid(), "angle": 0, "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "solid", "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None,
        "seed": random.randint(1, 2 ** 31 - 1), "version": 1,
        "versionNonce": random.randint(1, 2 ** 31 - 1), "isDeleted": False,
        "boundElements": [], "updated": int(time.time() * 1000), "link": None,
        "locked": False, **extra,
    }


def rect(x, y, w, h, stroke=DARK, bg="transparent", rounded=True):
    elements.append(_base({
        "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "strokeColor": stroke, "backgroundColor": bg,
        "roundness": {"type": 3} if rounded else None,
    }))


def diamond(x, y, w, h, stroke=DARK, bg="transparent"):
    elements.append(_base({
        "type": "diamond", "x": x, "y": y, "width": w, "height": h,
        "strokeColor": stroke, "backgroundColor": bg, "roundness": {"type": 2},
    }))


def text(x, y, body, color=DARK, size=14, align="left", w=None):
    lines = body.split("\n")
    longest = max((len(l) for l in lines), default=0)
    width = w or int(longest * size * 0.62) or 10
    height = int(len(lines) * size * 1.3)
    elements.append(_base({
        "type": "text", "x": x, "y": y, "width": width, "height": height,
        "strokeColor": color, "backgroundColor": "transparent", "roundness": None,
        "text": body, "fontSize": size, "fontFamily": 2, "textAlign": align,
        "verticalAlign": "top", "containerId": None, "originalText": body,
        "lineHeight": 1.3, "baseline": int(size * 0.85 * len(lines)),
    }))


def ctext(cx, y, body, color=DARK, size=14):
    """Text horizontally centered on cx."""
    longest = max((len(l) for l in body.split("\n")), default=0)
    width = int(longest * size * 0.62) or 10
    text(cx - width / 2, y, body, color=color, size=size, align="center", w=width)


def arrow(x1, y1, x2, y2, stroke=DARK, dashed=False):
    elements.append(_base({
        "type": "arrow", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
        "strokeColor": stroke, "backgroundColor": "transparent",
        "strokeStyle": "dashed" if dashed else "solid", "roundness": {"type": 2},
        "points": [[0, 0], [x2 - x1, y2 - y1]], "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None, "startArrowhead": None,
        "endArrowhead": "triangle",
    }))


def node(x, y, w, h, icon, label, stroke=DARK, bg="transparent", icon_size=26, label_size=11):
    """A box with a big icon line and a short label line -- the primary
    visual unit of this diagram. Icon-first, label short (<=3 words ideally)."""
    rect(x, y, w, h, stroke=stroke, bg=bg)
    ctext(x + w / 2, y + 8, icon, size=icon_size)
    ctext(x + w / 2, y + h - label_size * 1.6 * (label.count("\n") + 1) - 6,
          label, color=stroke, size=label_size)


def tag(x, y, w, body, color=GRAY[0], size=10):
    """A thin dataset/provenance footnote under a node. Small, not a box."""
    ctext(x + w / 2, y, body, color=color, size=size)


def arrow_label(x1, y1, x2, y2, label, stroke=DARK, dashed=False, size=11):
    arrow(x1, y1, x2, y2, stroke=stroke, dashed=dashed)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ctext(mx, my - 8, label, color=stroke, size=size)


# =====================================================================
# Header
# =====================================================================
text(40, 20, "GlucoSight — Current State & Live Data Flow", size=26)
text(40, 58, "Snapshot 2026-10-20 · follow the arrows, not the paragraphs", color="#868e96", size=13)

# Legend, top right
LX, LY = 1460, 20
rect(LX, LY, 220, 150, stroke=GRAY[0], bg=GRAY[1])
text(LX + 12, LY + 8, "LEGEND", size=12, color=GRAY[0])
legend_rows = [
    (GREEN, "default — real, unaffected"),
    (ORANGE, "experimental — ungated"),
    (BLUE, "plumbing"),
    (GRAY, "ignored in this mode"),
]
for i, (col, lbl) in enumerate(legend_rows):
    ry = LY + 34 + i * 28
    rect(LX + 12, ry, 16, 16, stroke=col[0], bg=col[1])
    text(LX + 36, ry - 1, lbl, size=11)

# =====================================================================
# Section 1: two serving modes, compact
# =====================================================================
S1_Y = 90

# baseline_demo
rect(40, S1_Y, 660, 26, stroke=GREEN[0], bg=GREEN[1])
ctext(40 + 330, S1_Y + 5, "baseline_demo — THE REAL DEFAULT", color=GREEN[0], size=13)
node(60, S1_Y + 40, 150, 68, "\U0001FA78\U0001F4C8", "g0 only", stroke=GREEN[0])
arrow(210, S1_Y + 74, 250, S1_Y + 74, stroke=GREEN[0])
node(250, S1_Y + 40, 150, 68, "➕", "cohort mean", stroke=GREEN[0])
arrow(400, S1_Y + 74, 440, S1_Y + 74, stroke=GREEN[0])
node(440, S1_Y + 40, 150, 68, "\U0001F3AF", "T+60 / T+120", stroke=GREEN[0])
ctext(370, S1_Y + 120, "\U0001F4F7 CV   \U0001F4DD NLP   \U0001F493 PPG  —  processed but IGNORED", color=GRAY[0], size=12)

# experimental_fusion
rect(740, S1_Y, 700, 26, stroke=ORANGE[0], bg=ORANGE[1])
ctext(740 + 350, S1_Y + 5, "experimental_fusion — OPT-IN ONLY", color=ORANGE[0], size=13)
node(760, S1_Y + 40, 150, 68, "\U0001FA78\U0001F4C8\U0001F493", "g0 priority", stroke=ORANGE[0])
arrow(910, S1_Y + 74, 950, S1_Y + 74, stroke=ORANGE[0])
node(950, S1_Y + 40, 150, 68, "\U0001F4F7 ➕", "+ CV adj.", stroke=ORANGE[0])
arrow(1100, S1_Y + 74, 1140, S1_Y + 74, stroke=ORANGE[0])
node(1140, S1_Y + 40, 150, 68, "\U0001F4DD ➕", "+ NLP adj.", stroke=ORANGE[0])
arrow(1290, S1_Y + 74, 1330, S1_Y + 74, stroke=ORANGE[0])
node(1330, S1_Y + 40, 110, 68, "\U0001F3AF", "T+60", stroke=ORANGE[0])
ctext(1090, S1_Y + 120, "⚠ bypasses the evidence gate — not validated", color=ORANGE[0], size=12)

# =====================================================================
# Section 2: detailed live flow (experimental_fusion internals)
# =====================================================================
S2_Y = 270
text(40, S2_Y - 30, "experimental_fusion, step by step", size=16, color=DARK)

# input icons
node(60, S2_Y, 170, 60, "\U0001F4F7", "Meal photo", stroke=DARK, icon_size=24)
node(420, S2_Y, 170, 60, "\U0001F4DD", "Bahasa note", stroke=DARK, icon_size=24)
node(780, S2_Y, 170, 60, "\U0001FA78\U0001F4C8", "Fingerstick / CGM", stroke=DARK, icon_size=24)
node(1180, S2_Y, 170, 60, "\U0001F493", "PPG capture", stroke=DARK, icon_size=24)

LANE_Y = S2_Y + 100

# --- CV lane ---
arrow(145, S2_Y + 60, 145, LANE_Y, stroke=ORANGE[0])
node(60, LANE_Y, 300, 60, "\U0001F37D", "Dish identification", stroke=ORANGE[0])
arrow(210, LANE_Y + 60, 210, LANE_Y + 90, stroke=ORANGE[0])
node(60, LANE_Y + 90, 300, 60, "\U0001F4CB", "Dish table → carbs/macros", stroke=ORANGE[0])
arrow(210, LANE_Y + 150, 210, LANE_Y + 180, stroke=ORANGE[0])
node(60, LANE_Y + 180, 300, 70, "\U0001F4D0", "CGMacros ridge", stroke=ORANGE[0])
tag(60, LANE_Y + 255, 300, "\U0001F5C4 CGMacros · 269/405 rows · 15 ppts")
arrow(210, LANE_Y + 250, 210, LANE_Y + 300, stroke=ORANGE[0])
node(60, LANE_Y + 300, 300, 46, "Δ CV", "× cv_confidence weight", stroke=ORANGE[0], bg=ORANGE[1])

# --- NLP lane ---
arrow(505, S2_Y + 60, 505, LANE_Y, stroke=ORANGE[0])
node(420, LANE_Y, 300, 60, "\U0001F524", "Bahasa lexicon", stroke=ORANGE[0])
arrow(570, LANE_Y + 60, 570, LANE_Y + 90, stroke=ORANGE[0])
node(420, LANE_Y + 90, 300, 70, "\U0001F4D0", "BIG IDEAs ridge", stroke=ORANGE[0])
tag(420, LANE_Y + 165, 300, "\U0001F5C4 BIG IDEAs · 666 meals · 15 ppts")
arrow(570, LANE_Y + 160, 570, LANE_Y + 300, stroke=ORANGE[0])
node(420, LANE_Y + 300, 300, 46, "Δ NLP", "× nlp_confidence weight", stroke=ORANGE[0], bg=ORANGE[1])

# --- glucose / PPG decision lane ---
arrow(865, S2_Y + 60, 865, LANE_Y, stroke=BLUE[0])
arrow(1265, S2_Y + 60, 1265, LANE_Y, stroke=BLUE[0])
diamond(780, LANE_Y, 220, 70, stroke=BLUE[0])
ctext(890, LANE_Y + 25, "direct measurement\npresent?", size=11)
arrow_label(1000, LANE_Y + 35, 1080, LANE_Y + 35, "yes", stroke=GREEN[0])
node(1080, LANE_Y, 260, 60, "\U0001FA78\U0001F4C8", "g0 = fingerstick/CGM\ntrust 0.95–1.0", stroke=GREEN[0], icon_size=22, label_size=10)
arrow_label(890, LANE_Y + 70, 890, LANE_Y + 130, "no", stroke=ORANGE[0])
diamond(780, LANE_Y + 130, 220, 70, stroke=ORANGE[0])
ctext(890, LANE_Y + 155, "PPG capture\npresent?", size=11)
arrow_label(1000, LANE_Y + 165, 1080, LANE_Y + 165, "yes", stroke=ORANGE[0])
node(1080, LANE_Y + 130, 260, 80, "\U0001F4D0", "rppg ridge → g0 estimate\ntrust 0.30", stroke=ORANGE[0], icon_size=22, label_size=10)
tag(1080, LANE_Y + 215, 260, "\U0001F5C4 rppg/PPG_Dataset · 13/67 recs · 8 subj")
arrow_label(890, LANE_Y + 200, 890, LANE_Y + 280, "no", stroke=GRAY[0])
node(780, LANE_Y + 280, 260, 60, "\U0001F30D", "g0 = population fallback\n120 mg/dL, trust 0.10", stroke=GRAY[0], icon_size=20, label_size=10)

MERGE_G0_Y = LANE_Y + 370
arrow(1210, LANE_Y + 60, 950, MERGE_G0_Y, stroke=GREEN[0])
arrow(1210, LANE_Y + 210, 950, MERGE_G0_Y, stroke=ORANGE[0])
arrow(910, LANE_Y + 340, 950, MERGE_G0_Y, stroke=GRAY[0])
node(820, MERGE_G0_Y, 260, 56, "\U0001FA78", "g0 (resolved)", stroke=BLUE[0], bg=BLUE[1])

# --- merge everything into the forecast ---
OUT_Y = MERGE_G0_Y + 110
arrow(210, LANE_Y + 346, 480, OUT_Y + 20, stroke=ORANGE[0])
arrow(570, LANE_Y + 346, 650, OUT_Y + 20, stroke=ORANGE[0])
arrow(950, MERGE_G0_Y + 56, 820, OUT_Y + 20, stroke=BLUE[0])
node(480, OUT_Y, 400, 70, "➕", "g0 + ΔCV + ΔNLP\n(each weighted)", stroke=DARK, label_size=12)
arrow(680, OUT_Y + 70, 680, OUT_Y + 110, stroke=DARK)
node(480, OUT_Y + 110, 400, 70, "\U0001F3AF", "T+60 forecast (mg/dL)", stroke=ORANGE[0], bg=ORANGE[1], label_size=12)
arrow_label(950, MERGE_G0_Y + 56, 1100, OUT_Y + 110, "skips CV/NLP", stroke=GRAY[0], dashed=True, size=10)
node(940, OUT_Y + 110, 320, 70, "\U0001F3AF", "T+120 = g0 + fixed excursion\nno trained arm, either mode", stroke=GRAY[0], icon_size=20, label_size=10)

# =====================================================================
# Section 3: dataset matrix
# =====================================================================
M_Y = OUT_Y + 230
text(40, M_Y - 30, "\U0001F5C4 Which dataset has which modality?", size=16)

cols = ["", "\U0001F4F7 Photo", "\U0001F4DD Text", "\U0001F493 PPG", "\U0001FA78\U0001F4C8 Glucose"]
col_x = [40, 300, 460, 620, 780]
col_w = [260, 160, 160, 160, 170]
for cx, cw, label in zip(col_x, col_w, cols):
    if label:
        ctext(cx + cw / 2, M_Y, label, size=12, color=DARK)

rows = [
    ("CGMacros", ["✅", "❌", "❌", "✅"]),
    ("BIG IDEAs", ["❌", "✅", "✅*", "✅"]),
    ("ShanghaiT2DM", ["❌", "✅", "❌", "✅"]),
    ("rppg/PPG_Dataset", ["❌", "❌", "✅", "✅†"]),
]
ROW_H = 42
for r, (name, cells) in enumerate(rows):
    ry = M_Y + 30 + r * ROW_H
    rect(col_x[0], ry, sum(col_w), ROW_H, stroke=GRAY[0],
         bg="#ffffff" if r % 2 == 0 else GRAY[1], rounded=False)
    text(col_x[0] + 12, ry + 11, name, size=12)
    for cx, cw, mark in zip(col_x[1:], col_w[1:], cells):
        ctext(cx + cw / 2, ry + 8, mark, size=16)

text(col_x[0], M_Y + 30 + len(rows) * ROW_H + 6,
     "* raw BVP exists, unused (4.7 GB fetch too slow)   † single glucometer reading, not a meal response",
     size=10, color=GRAY[0])

rect(col_x[0], M_Y + 30 + len(rows) * ROW_H + 30, sum(col_w), 40, stroke=BLUE[0], bg=BLUE[1])
ctext(col_x[0] + sum(col_w) / 2, M_Y + 30 + len(rows) * ROW_H + 42,
      "No dataset has ✅✅✅ together → can't train ONE joint model (ADR 0004)",
      size=12, color=BLUE[0])

# =====================================================================
# Footer
# =====================================================================
FOOT_Y = M_Y + 30 + len(rows) * ROW_H + 90
rect(40, FOOT_Y, sum(col_w) + 760, 44, stroke=RED[0], bg=RED[1])
ctext(40 + (sum(col_w) + 760) / 2, FOOT_Y + 12,
      "⚠ None of CV / NLP / PPG has beaten its baseline yet. Wired in anyway, by explicit decision (2026-10-20) — not a validated result.",
      size=13, color=RED[0])

# =====================================================================
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
print(f"wrote {len(elements)} elements to {OUT}")
