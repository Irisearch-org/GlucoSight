"""Generate docs/diagrams/glucosight-complete.excalidraw.

ONE complete explainer: the original pitch, the reality check that forced
a pivot, what each of the four tracks actually does today (goal -> pivot ->
today -> dataset -> result), how the live app composes them, a scoreboard,
and the blockers. Icon + arrow + short-label first; prose stays in
docs/INTEGRATION_AUDIT.md, not here.

Regenerate after the next architecture change:
    python docs/diagrams/generate_complete_excalidraw.py
"""
import json
import random
import time

OUT = "docs/diagrams/glucosight-complete.excalidraw"

GREEN = ("#2f9e44", "#ebfbee")
ORANGE = ("#e8590c", "#fff4e6")
BLUE = ("#1971c2", "#e7f5ff")
GRAY = ("#868e96", "#f1f3f5")
RED = ("#e03131", "#fff5f5")
PURPLE = ("#7048e8", "#f3f0ff")
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


def rect(x, y, w, h, stroke=DARK, bg="transparent", rounded=True, dashed=False, width=2):
    elements.append(_base({
        "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "strokeColor": stroke, "backgroundColor": bg, "strokeWidth": width,
        "strokeStyle": "dashed" if dashed else "solid",
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


def arrow_label(x1, y1, x2, y2, label, stroke=DARK, dashed=False, size=11):
    arrow(x1, y1, x2, y2, stroke=stroke, dashed=dashed)
    ctext((x1 + x2) / 2, (y1 + y2) / 2 - 8, label, color=stroke, size=size)


def node(x, y, w, h, icon, label, stroke=DARK, bg="transparent", icon_size=24, label_size=11, dashed=False):
    rect(x, y, w, h, stroke=stroke, bg=bg, dashed=dashed)
    n_lines = icon.count("\n") + 1
    ctext(x + w / 2, y + 7, icon, size=icon_size)
    ctext(x + w / 2, y + h - label_size * 1.5 * (label.count("\n") + 1) - 6, label, color=stroke, size=label_size)


def tag(x, w, y, body, color=GRAY[0], size=10):
    ctext(x + w / 2, y, body, color=color, size=size)


def section_title(x, y, body, color=DARK, size=18):
    text(x, y, body, color=color, size=size)


# =====================================================================
# ZONE A — the pitch
# =====================================================================
text(40, 20, "GlucoSight — How It Works", size=28)
text(40, 60, "A complete walkthrough: the pitch, the reality check, the four tracks, today's live app, and where it stands.",
     color="#868e96", size=13)

rect(40, 100, 1640, 90, stroke=GREEN[0], bg=GREEN[1])
ctext(860, 115, "THE PITCH", color=GREEN[0], size=12)
ctext(860, 140,
      "\U0001F4F7 Meal photo  +  \U0001F4DD Text note  +  \U0001F493 Finger scan   →   \U0001F3AF Glucose at T+60 & T+120, no CGM needed",
      size=16, color=DARK)

# =====================================================================
# ZONE B — reality check
# =====================================================================
RY = 210
arrow_label(860, 190, 860, RY, "checked the real datasets...", stroke=GRAY[0], size=12)

rect(40, RY, 780, 230, stroke=GRAY[0], bg="#ffffff")
ctext(430, RY + 10, "\U0001F5C4 WHICH DATASET HAS WHICH MODALITY?", size=12, color=GRAY[0])
cols = ["", "\U0001F4F7", "\U0001F4DD", "\U0001F493", "\U0001FA78\U0001F4C8"]
col_x = [60, 330, 440, 550, 660]
col_w = [270, 110, 110, 110, 140]
for cx, cw, label in zip(col_x, col_w, cols):
    if label:
        ctext(cx + cw / 2, RY + 35, label, size=16)
rows = [
    ("CGMacros", ["✅", "❌", "❌", "✅"]),
    ("BIG IDEAs", ["❌", "✅", "✅*", "✅"]),
    ("ShanghaiT2DM", ["❌", "✅", "❌", "✅"]),
    ("rppg/PPG_Dataset", ["❌", "❌", "✅", "✅†"]),
]
ROW_H = 36
for r, (name, cells) in enumerate(rows):
    ry = RY + 60 + r * ROW_H
    if r % 2 == 1:
        rect(60, ry, 740, ROW_H, stroke="transparent", bg=GRAY[1], rounded=False)
    text(70, ry + 9, name, size=11)
    for cx, cw, mark in zip(col_x[1:], col_w[1:], cells):
        ctext(cx + cw / 2, ry + 6, mark, size=14)
text(60, RY + 60 + len(rows) * ROW_H + 6,
     "* raw PPG exists, unused — 4.7 GB fetch too slow     † single reading, not a meal response",
     size=9, color=GRAY[0])

arrow(820, RY + 115, 870, RY + 115, stroke=RED[0])
rect(870, RY + 30, 400, 170, stroke=RED[0], bg=RED[1])
ctext(1070, RY + 45, "⚠ NO DATASET HAS ✅✅✅ TOGETHER", color=RED[0], size=13)
ctext(1070, RY + 80, "Can't train ONE joint\ntri-modal model", size=12)
ctext(1070, RY + 130, "(ADR 0004)", color=GRAY[0], size=11)
ctext(1070, RY + 155, "→ each track pivoted\nindependently instead", size=11, color=RED[0])

rect(1290, RY, 390, 230, stroke=PURPLE[0], bg=PURPLE[1])
ctext(1485, RY + 15, "PROJECT GOAL, REVISED", color=PURPLE[0], size=12)
ctext(1485, RY + 45,
      "Prove each track's signal\nseparately, on the one real\ndataset that supports it\n— not one fused claim.",
      size=13, color=DARK)
ctext(1485, RY + 170, "(DATA_STRATEGY.md)", color=GRAY[0], size=11)

# =====================================================================
# ZONE C — the four tracks
# =====================================================================
CY = RY + 270
section_title(40, CY, "The four tracks — goal → pivot → today → dataset → result", size=18)

TRACK_Y = CY + 40
COL_W = 390
GAP = 25
TRACKS_X = [40, 40 + COL_W + GAP, 40 + 2 * (COL_W + GAP), 40 + 3 * (COL_W + GAP)]


def track_column(x, color, icon, title, goal, pivot_label, pivot_ref, today, dataset, result, result_color):
    y = TRACK_Y
    rect(x, y, COL_W, 34, stroke=color[0], bg=color[1])
    ctext(x + COL_W / 2, y + 8, f"{icon}  {title}", size=14, color=color[0])
    y += 34 + 14
    node(x, y, COL_W, 56, "\U0001F3AF", goal, stroke=GRAY[0], icon_size=18, label_size=11)
    y += 56
    arrow_label(x + COL_W / 2, y, x + COL_W / 2, y + 40, f"PIVOT — {pivot_ref}", stroke=ORANGE[0], size=10)
    y += 40
    node(x, y, COL_W, 64, "\U0001F504", pivot_label, stroke=ORANGE[0], bg=ORANGE[1], icon_size=18, label_size=11)
    y += 64
    arrow(x + COL_W / 2, y, x + COL_W / 2, y + 26, stroke=color[0])
    y += 26
    node(x, y, COL_W, 64, "⚙", today, stroke=color[0], icon_size=18, label_size=11)
    y += 64
    arrow(x + COL_W / 2, y, x + COL_W / 2, y + 26, stroke=color[0])
    y += 26
    node(x, y, COL_W, 50, "\U0001F5C4", dataset, stroke=BLUE[0], icon_size=16, label_size=10)
    y += 50
    arrow(x + COL_W / 2, y, x + COL_W / 2, y + 26, stroke=color[0])
    y += 26
    node(x, y, COL_W, 54, "\U0001F4CA", result, stroke=result_color[0], bg=result_color[1], icon_size=18, label_size=10)
    return y + 54


bottoms = []
bottoms.append(track_column(
    TRACKS_X[0], ORANGE, "\U0001F4F7", "CV",
    "Food recognition +\nmacro estimation from photo",
    "Dish ID only — macros\nfrom lookup, not CV",
    "ADR 0002",
    "Indonesian dish classifier\n+ TKPI dish table",
    "CGMacros ground-truth\nmacros (269/405 rows, 15 ppt)",
    "MAE 38.90 vs base 37.46\n— LOSES to baseline",
    RED,
))
bottoms.append(track_column(
    TRACKS_X[1], ORANGE, "\U0001F4DD", "NLP",
    "Bahasa text → IndoBERT\n→ glucose contribution",
    "Dish-identity → carb-\ndensity bridge",
    "ADR 0001",
    "Bahasa lexicon (digoreng,\nporsi besar) → is_fried/is_large",
    "ShanghaiT2DM (dish ridge)\n+ BIG IDEAs (666 meals)",
    "Dish ridge: tiny gain, gated,\nLIVE. is_fried: ~0 gain (B12)",
    ORANGE,
))
bottoms.append(track_column(
    TRACKS_X[2], GRAY, "\U0001F493", "Contact PPG",
    "Finger scan → glucose\nestimate, replace CGM",
    "Glucose estimation\nCLOSED — dead end",
    "ADR 0003",
    "Feeds g0 only when no\nfingerstick/CGM (experimental)",
    "rppg/PPG_Dataset, local\n(13/67 recs, 8 subjects)",
    "MAE 10.24 vs const. 10.05\n— LOSES, fails gate",
    RED,
))
bottoms.append(track_column(
    TRACKS_X[3], BLUE, "\U0001F4C8", "Forecasting",
    "LSTM/TFT fusion,\nZone A > 70%",
    "B0/B1/B2 baselines\nare the only real numbers",
    "Audit F1–F3",
    "Serves B2 by default;\nexperimental_fusion blends CV+NLP+PPG",
    "CGMacros (primary) + BIG\nIDEAs + ShanghaiT2DM",
    "\"Sprint2 62.5%\" was oracle-\ntier, mislabeled — not fusion",
    RED,
))

# =====================================================================
# ZONE D — how the live app composes them today
# =====================================================================
DY = max(bottoms) + 60
section_title(40, DY, "Today's live app — how the pieces combine at request time", size=18)
DY += 40

rect(40, DY, 760, 26, stroke=GREEN[0], bg=GREEN[1])
ctext(420, DY + 5, "baseline_demo — THE REAL DEFAULT", color=GREEN[0], size=12)
node(60, DY + 40, 150, 60, "\U0001FA78\U0001F4C8", "g0 only", stroke=GREEN[0], icon_size=20)
arrow(210, DY + 70, 250, DY + 70, stroke=GREEN[0])
node(250, DY + 40, 150, 60, "➕", "cohort mean", stroke=GREEN[0], icon_size=20)
arrow(400, DY + 70, 440, DY + 70, stroke=GREEN[0])
node(440, DY + 40, 150, 60, "\U0001F3AF", "T+60/T+120", stroke=GREEN[0], icon_size=20)
ctext(420, DY + 112, "\U0001F4F7\U0001F4DD\U0001F493 processed but ignored", color=GRAY[0], size=11)

rect(860, DY, 820, 26, stroke=ORANGE[0], bg=ORANGE[1])
ctext(1270, DY + 5, "experimental_fusion — OPT-IN, BYPASSES THE EVIDENCE GATE", color=ORANGE[0], size=12)

diamond(880, DY + 40, 170, 60, stroke=BLUE[0])
ctext(965, DY + 58, "direct g0?\n(finger/CGM)", size=10)
arrow_label(1050, DY + 70, 1110, DY + 70, "no + PPG", stroke=ORANGE[0], size=9)
node(1110, DY + 40, 130, 60, "\U0001F493", "PPG → g0\n(trust 0.30)", stroke=ORANGE[0], icon_size=18, label_size=9)
arrow(965, DY + 100, 965, DY + 150, stroke=BLUE[0])
arrow(1175, DY + 100, 1000, DY + 150, stroke=ORANGE[0])
node(900, DY + 150, 200, 44, "\U0001FA78", "g0 resolved", stroke=BLUE[0], bg=BLUE[1], icon_size=16, label_size=10)
arrow(1000, DY + 194, 1000, DY + 230, stroke=DARK)
arrow(1310, DY + 70, 1310, DY + 230, stroke=ORANGE[0])
node(1260, DY + 40, 120, 60, "\U0001F4F7➕", "CV adj.", stroke=ORANGE[0], icon_size=18, label_size=9)
arrow(1460, DY + 70, 1460, DY + 230, stroke=ORANGE[0])
node(1410, DY + 40, 120, 60, "\U0001F4DD➕", "NLP adj.", stroke=ORANGE[0], icon_size=18, label_size=9)
node(900, DY + 230, 630, 50, "➕", "g0 + ΔCV×weight + ΔNLP×weight", stroke=DARK, label_size=11)
arrow(1215, DY + 280, 1215, DY + 320, stroke=ORANGE[0])
node(1070, DY + 320, 300, 50, "\U0001F3AF", "T+60 forecast", stroke=ORANGE[0], bg=ORANGE[1], label_size=11)

# =====================================================================
# ZONE E — scoreboard
# =====================================================================
EY = DY + 400
section_title(40, EY, "Scoreboard — what's actually proven", size=18)
EY += 40
s_cols = ["Track", "Real data?", "Beats baseline?", "Affects live forecast?"]
s_x = [40, 400, 650, 950]
s_w = [360, 250, 300, 350]
for cx, cw, label in zip(s_x, s_w, s_cols):
    ctext(cx + cw / 2, EY, label, size=12, color=GRAY[0])
score_rows = [
    ("\U0001F4F7 CV (CGMacros macros)", "✅ yes", "❌ no (38.90 vs 37.46)", "✅ experimental_fusion only"),
    ("\U0001F4DD NLP (dish ridge)", "✅ yes", "✅ tiny, gated", "✅ nlp_dish_identity_forecast"),
    ("\U0001F4DD NLP (is_fried adj.)", "✅ yes", "❌ ~0 gain (B12)", "✅ experimental_fusion only"),
    ("\U0001F493 Contact PPG", "✅ yes", "❌ no (10.24 vs 10.05)", "✅ experimental_fusion only"),
    ("\U0001F4C8 Forecasting baselines", "✅ yes", "— (B2 IS the baseline)", "✅ baseline_demo (default)"),
]
SROW_H = 38
for r, cells in enumerate(score_rows):
    ry = EY + 20 + r * SROW_H
    if r % 2 == 1:
        rect(40, ry, 1260, SROW_H, stroke="transparent", bg=GRAY[1], rounded=False)
    for cx, cw, val in zip(s_x, s_w, cells):
        text(cx + 8, ry + 10, val, size=11)

# =====================================================================
# ZONE F — blockers
# =====================================================================
FY = EY + 20 + len(score_rows) * SROW_H + 50
rect(40, FY, 1640, 150, stroke=RED[0], bg=RED[1])
ctext(860, FY + 12, "⚠ BLOCKERS TO A REAL (NON-EXPERIMENTAL) FUSION FORECAST", color=RED[0], size=13)
blockers = [
    "1. Fix CGMacros data-quality violations + the participant-join bug",
    "2. Train a deployable model that actually beats baseline, with error bars",
    "3. Resolve the data-quality / bio.csv gaps blocking the oracle-tier CGMacros run",
    "4. Route CV fully through the shared contract (4 fields still unowned)",
    "5. Decide contract v2.0: PPG ring-fence, g0_source, staleness rule",
    "6. Reconcile root CLAUDE.md (tri-modal) vs DATA_STRATEGY.md (bi-modal pivot)",
]
for i, b in enumerate(blockers[:3]):
    text(60, FY + 42 + i * 30, b, size=11, color=DARK, w=780)
for i, b in enumerate(blockers[3:]):
    text(880, FY + 42 + i * 30, b, size=11, color=DARK, w=780)

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
print(f"wrote {len(elements)} elements to {OUT}, canvas height ~{FY + 150}")
