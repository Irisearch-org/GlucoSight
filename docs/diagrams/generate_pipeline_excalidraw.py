"""Generate docs/diagrams/glucosight-pipeline.excalidraw.

Regenerate after any status change: edit the TEXT constants below, re-run.
    python docs/diagrams/generate_pipeline_excalidraw.py

Grounded in: docs/INTEGRATION_AUDIT.md, docs/adr/0001-0004, docs/DATA_STRATEGY.md,
integration/README.md (as of 2026-10-20, branch feat/cgmacros-pipeline-spec).
"""
import json
import random
import time

OUT = "docs/diagrams/glucosight-pipeline.excalidraw"

# ---- status palette -------------------------------------------------------
GREEN = ("#2f9e44", "#ebfbee")   # working / validated
YELLOW = ("#e8590c", "#fff9db")  # partial / gated / mislabeled
RED = ("#e03131", "#fff5f5")     # blocked / dead end / key gap
BLUE = ("#1971c2", "#e7f5ff")    # plumbing, built, not model-bearing
GRAY = ("#495057", "#f1f3f5")    # neutral / informational
DARK = "#1e1e1e"

elements = []
_id = 0


def nid():
    global _id
    _id += 1
    return f"el{_id}"


def rect(x, y, w, h, stroke, bg):
    elements.append({
        "id": nid(), "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke, "backgroundColor": bg,
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": {"type": 3},
        "seed": random.randint(1, 2 ** 31 - 1), "version": 1,
        "versionNonce": random.randint(1, 2 ** 31 - 1), "isDeleted": False,
        "boundElements": [], "updated": int(time.time() * 1000), "link": None,
        "locked": False,
    })


def text(x, y, body, color=DARK, size=13, w=None):
    lines = body.split("\n")
    longest = max((len(l) for l in lines), default=0)
    width = w or int(longest * size * 0.62)
    height = int(len(lines) * size * 1.3)
    elements.append({
        "id": nid(), "type": "text", "x": x, "y": y, "width": width, "height": height,
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": None,
        "seed": random.randint(1, 2 ** 31 - 1), "version": 1,
        "versionNonce": random.randint(1, 2 ** 31 - 1), "isDeleted": False,
        "boundElements": [], "updated": int(time.time() * 1000), "link": None,
        "locked": False,
        "text": body, "fontSize": size, "fontFamily": 2, "textAlign": "left",
        "verticalAlign": "top", "containerId": None, "originalText": body,
        "lineHeight": 1.3, "baseline": int(size * 0.85 * len(lines)),
    })


def arrow(x1, y1, x2, y2, stroke=DARK, dashed=False):
    elements.append({
        "id": nid(), "type": "arrow", "x": x1, "y": y1,
        "width": x2 - x1, "height": y2 - y1, "angle": 0,
        "strokeColor": stroke, "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 2, "strokeStyle": "dashed" if dashed else "solid", "roughness": 1,
        "opacity": 100, "groupIds": [], "frameId": None, "roundness": {"type": 2},
        "seed": random.randint(1, 2 ** 31 - 1), "version": 1,
        "versionNonce": random.randint(1, 2 ** 31 - 1), "isDeleted": False,
        "boundElements": [], "updated": int(time.time() * 1000), "link": None,
        "locked": False,
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "triangle",
    })


def box(x, y, w, h, stroke, bg, header, body):
    rect(x, y, w, h, stroke, bg)
    text(x + 15, y + 12, header, color=stroke, size=16)
    text(x + 15, y + 55, body, color=DARK, size=13)


# ---- title ------------------------------------------------------------------
text(40, 20, "GlucoSight — Pipeline & Current State", color=DARK, size=28)
text(40, 60, "Snapshot 2026-10-20 · branch feat/cgmacros-pipeline-spec · "
             "grounded in docs/INTEGRATION_AUDIT.md, ADRs 0001-0004, DATA_STRATEGY.md",
     color="#868e96", size=14)

# ---- legend -----------------------------------------------------------------
box(1160, 20, 290, 170, GRAY[0], GRAY[1], "LEGEND",
    "green = working / validated\n"
    "yellow = partial / gated /\n  mislabeled\n"
    "red = blocked / dead end /\n  key gap\n"
    "blue = plumbing (built,\n  not model-bearing yet)")

# ---- goal vs reality banner --------------------------------------------------
box(40, 100, 1100, 170, BLUE[0], BLUE[1], "ORIGINAL GOAL vs REALITY",
    "Goal: CV + Contact PPG + NLP fuse into ONE glucose\n"
    "forecast model (root CLAUDE.md).\n\n"
    "Reality: no public dataset has photo + text + glucose\n"
    "together (ADR 0004). Pivoted to separate per-track\n"
    "validations feeding a plumbing layer that currently\n"
    "serves a labeled baseline — not a fusion result.")

# ---- row 1: four tracks -------------------------------------------------------
ROW1_Y = 310
COL_W = 330
cols = [40, 400, 760, 1120]

box(cols[0], ROW1_Y, COL_W, 230, YELLOW[0], YELLOW[1],
    "CV TRACK — Dish Identification\nStatus: PIVOTED (ADR 0002)",
    "- Identifies dish, not macros\n"
    "- Macros = lookup table by dish,\n  not a model output\n"
    "- Indonesian dish classifier built\n"
    "- On CGMacros (US food): known\n  out-of-domain failure\n"
    "- Not yet on shared contract\n  (4 fields have no producer)")

box(cols[1], ROW1_Y, COL_W, 230, RED[0], RED[1],
    "CONTACT PPG TRACK — Finger Scan\nStatus: CLOSED for glucose (ADR 0003)",
    "- Glucose estimation: dead end,\n  independently replicated (2026 paper)\n"
    "- Constant predictor hits 83.6%\n  Zone A — metric can't tell a real\n  model from a constant here\n"
    "- Evidence gate rejects all models\n  (correctly), incl. noise models\n"
    "- g0-from-PPG: wired but GATED OFF")

box(cols[2], ROW1_Y, COL_W, 230, YELLOW[0], YELLOW[1],
    "NLP TRACK — Dish Identity→Carbs\nStatus: PARTIAL / GATED (ADR 0001)",
    "- Pivoted from translation to\n  dish-identity → carb density\n"
    "- Ridge model trained on real\n  ShanghaiT2DM, now LIVE\n"
    "- Opt-in mode, T+60 ONLY\n  (T+120 falls back to baseline)\n"
    "- Gain is small: MAE 31.1 vs\n  32.3 constant baseline\n"
    "- Bahasa input: NON-FUNCTIONAL\n  (no frying/portion lexicon)")

box(cols[3], ROW1_Y, COL_W, 230, YELLOW[0], YELLOW[1],
    "FORECASTING TRACK\nStatus: BASELINES REAL, FUSION NOT TRAINED",
    "- B0/B1/B2 baselines: real,\n  committed, correct\n"
    "- Sprint 2 “Multimodal” result is\n  ORACLE-tier (labs + gut panel),\n  not phone-deployable, mislabeled\n"
    "- Margin ~4 pts, NO error bars\n"
    "- True deployable fusion model:\n  NOT TRAINED — blocked on\n  CGMacros data-quality issues")

# ---- row 2: integration layer -------------------------------------------------
INTEG_Y = 580
box(40, INTEG_Y, 1410, 240, BLUE[0], BLUE[1],
    "INTEGRATION LAYER — the fusion spine\n"
    "Status: PLUMBING DONE, NO TRAINED MODEL WIRED IN",
    "- contract.py: v1.2 validator, single source of truth for all three tracks\n"
    "- adapters/: track-native output (CV/NLP/PPG) → contract records\n"
    "- features.py: 3 tiers — deployable / deployable+askable / oracle\n"
    "- glucose_source.py: g0 provenance (fingerstick/CGM/PPG/fallback) w/ trust weights — LIVE\n"
    "- predictor.py: serves BaselineB2Predictor only (plus the gated NLP T+60 opt-in)\n"
    "- report.py: refuses to build a results table missing B0/B1/B2\n"
    "- api.py: POST /predict, GET /contract, GET /health — FastAPI, live")

# ---- row 3: output -------------------------------------------------------------
OUT_Y = 860
box(40, OUT_Y, 1410, 150, RED[0], RED[1],
    "/predict RESPONSE — what a user actually gets today",
    "= g0 + cohort-mean excursion (BaselineB2). CV / PPG / NLP inputs are validated,\n"
    "weighted and carried through — but do NOT change the number (except the opt-in\n"
    "NLP T+60 mode). The API reports “is_trained_model: false” honestly.")

# ---- row 4: blockers ------------------------------------------------------------
BLOCK_Y = 1050
box(40, BLOCK_Y, 1410, 280, GRAY[0], GRAY[1],
    "BLOCKERS TO A REAL FUSION FORECAST (ordered)",
    "1. Fix CGMacros data-quality violations (136 rows outside documented ranges) and the\n"
    "   positional participant-join bug — no clinical/gut number means anything until then\n"
    "2. Train a deployable-tier model (CV+NLP+PPG features) and report it WITH error bars\n"
    "3. Carry Δt and add an overlapping-meal guard — likely cause of flat T+120 result\n"
    "4. Route CV fully through the shared contract (4 fields currently have no producer)\n"
    "5. Build a Bahasa Indonesia lexicon for NLP — note input is decorative without it\n"
    "6. Settle contract v2.0: PPG ring-fence amendment, g0_source, staleness rule\n"
    "7. Reconcile root CLAUDE.md (tri-modal / Indonesian) vs DATA_STRATEGY.md (bi-modal /\n"
    "   public-data pivot) — still open, unresolved")

# ---- arrows: tracks -> integration -> output ------------------------------------
centers = [c + COL_W // 2 for c in cols]
for cx, dashed, color in zip(centers, [False, True, False, False],
                              [DARK, RED[0], DARK, DARK]):
    arrow(cx, ROW1_Y + 230, cx, INTEG_Y, stroke=color, dashed=dashed)

text(centers[1] + 15, ROW1_Y + 232, "gated off", color=RED[0], size=12)

mid_x = 40 + 1410 // 2
arrow(mid_x, INTEG_Y + 240, mid_x, OUT_Y, stroke=DARK)
text(mid_x + 15, INTEG_Y + 242, "ignores CV/PPG/NLP today", color=RED[0], size=12)

# ---- write -----------------------------------------------------------------------
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
