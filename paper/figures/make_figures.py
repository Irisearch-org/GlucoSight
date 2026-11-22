#!/usr/bin/env python3
"""Build the GlucoSight paper figures (vector PDF + 300-dpi PNG) into paper/figures/.

Step 1 (prepare): if the original sources exist (/tmp out-of-fold predictions, the
BUT PPG branch, repo reports) write small derived CSVs to paper/figures/data/.
Step 2 (plot): every figure is drawn from those CSVs only, so the figures
reproduce after /tmp is cleaned. Run:  python paper/figures/make_figures.py
Derived CSVs hold anonymous indices only: no raw recordings, no TKPI values.
"""
import io
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, ROOT)
from utils.clarke_grid import classify_zone  # noqa: E402

COL1, COLW = 3.2, 6.5
# Okabe-Ito
BLK, ORG, SKY, GRN, YEL, BLU, VER, PUR = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GRY = "#7f7f7f"

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False,
})

BUT_BRANCH = "origin/wayfinder/butppg-glucose"
BUT_DIR = "rppg/experiments/butppg_glucose/results"


def p(*a):
    return os.path.join(DATA, *a)


def save(fig, name, **kw):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300, **kw)


def panel(ax, s, dx=-0.02, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


# ----------------------------------------------------------------------------- prepare
def prepare():
    os.makedirs(DATA, exist_ok=True)
    # forecast: per-participant MAE + Clarke points (CGM g0 source, GBDT = 'cur', B2)
    for h in (60, 120):
        src = f"/tmp/em_raw/_predictions_T{h}.csv.gz"
        if not os.path.exists(src):
            continue
        d = pd.read_csv(src)
        d = d[d.g0_source == "CGM"].copy()
        subs = sorted(d.subject.unique())
        anon = {s: i + 1 for i, s in enumerate(subs)}
        d["participant"] = d.subject.map(anon)
        d["ae_gbdt"] = (d.cur - d.y).abs()
        d["ae_b2"] = (d.b2 - d.y).abs()
        # per-repeat per-participant mean, then mean over the 5 repeats
        g = d.groupby(["participant", "repeat"])[["ae_gbdt", "ae_b2"]].mean().groupby("participant").mean()
        g["n_meals"] = d.groupby("participant").meal_id.nunique()
        g = g.rename(columns={"ae_gbdt": "mae_gbdt", "ae_b2": "mae_b2"}).round(3).reset_index()
        g.to_csv(p(f"forecast_per_participant_T{h}.csv"), index=False)
        m = d.groupby(["participant", "meal_id"]).agg(ref=("y", "first"), gbdt=("cur", "mean")).reset_index()
        m = m.drop(columns="meal_id")
        m.round(2).to_csv(p(f"forecast_meals_T{h}.csv"), index=False)
    # dish operating curve
    s = os.path.join(ROOT, "cv/reports/dish_id_trial/threshold_embedding.csv")
    if os.path.exists(s):
        pd.read_csv(s).to_csv(p("dish_threshold_embedding.csv"), index=False)
    # PPG_Dataset served ridge
    s = os.path.join(ROOT, "rppg/models/reports/ppg_served_residuals.csv")
    if os.path.exists(s):
        r = pd.read_csv(s)
        r = r.rename(columns={"subject_id": "subject"})[["subject", "glucose_true", "glucose_pred", "fold"]]
        r.to_csv(p("ppg_dataset_served.csv"), index=False)
    # BUT PPG v2 finger: OOF ridge + within-person deltas
    cache = "/tmp/butppg_cache/features.npz"
    if os.path.exists(cache):
        try:
            blob = subprocess.run(["git", "show", f"{BUT_BRANCH}:{BUT_DIR}/oof_hand_raw_ridge.npz"],
                                  cwd=ROOT, check=True, capture_output=True).stdout
            j = json.loads(subprocess.run(["git", "show", f"{BUT_BRANCH}:{BUT_DIR}/ppg28.json"],
                                          cwd=ROOT, check=True, capture_output=True).stdout)
        except subprocess.CalledProcessError:
            return
        o = np.load(io.BytesIO(blob))
        z = np.load(cache, allow_pickle=True)
        idx = o["idx"]
        subj = z["subj"][idx]
        anon = {s: i + 1 for i, s in enumerate(sorted(np.unique(subj)))}
        df = pd.DataFrame({
            "subject": [anon[s] for s in subj], "session": z["session_id"][idx],
            "sess_order": z["sess_order"][idx], "glucose": z["glucose"][idx],
            "pred": o["pred"].mean(0), "base": o["base"].mean(0)})
        df["session"] = pd.factorize(df.session)[0] + 1
        df.round(3).to_csv(p("butppg_finger_oof.csv"), index=False)
        dl = j["finger_still"]["a_delta_features"]["delta_ridge_loso"]["hand"]
        pd.DataFrame({"pair": range(1, 18), "pred_delta": dl["pred_delta"],
                      "true_delta": dl["true_delta"]}).to_csv(p("butppg_within_person_delta.csv"), index=False)


# ----------------------------------------------------------------------------- fig 1
def box(ax, x, y, w, h, text, fc="white", ec=BLK, lw=0.8, fs=7.5, bold=False, ls="-", r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.15)


def arrow(ax, a, b, ls="-", color=BLK, rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))


def finding(ax, x, w, text, color, h=2.25, y=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=color,
                                lw=1.2, zorder=2))
    ax.text(x + 0.12, y + h - 0.15, "Temuan", fontsize=8, fontweight="bold", color=color, va="top", zorder=3)
    ax.text(x + 0.12, y + h - 0.55, text, fontsize=8, color=BLK, va="top", ha="left", linespacing=1.25, zorder=3)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(COLW, 3.9))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8.0)
    ax.axis("off")
    cw = 4.1
    xs = [0.0, 4.7, 9.4]
    heads = ["1  Glukosa saat ini", "2  Makanan", "3  Prakiraan glukosa"]
    for x, hd in zip(xs, heads):
        ax.add_patch(FancyBboxPatch((x, 0.0), cw, 7.95, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc="#f2f2f2", ec=GRY, lw=0.6, zorder=0))
        ax.text(x + 0.15, 7.6, hd, fontsize=9, fontweight="bold", va="center")
    # --- step 1
    x = xs[0]
    for i, t in enumerate(["Glukometer", "CGM", "Pindai PPG\nponsel"]):
        box(ax, x + 0.2, 5.6 - i * 1.25, 2.1, 0.95, t, ls="--" if i == 2 else "-", ec=VER if i == 2 else BLK, fs=8)
    box(ax, x + 3.0, 4.5, 0.95, 0.95, "$g_0$", bold=True, fs=9)
    arrow(ax, (x + 2.3, 6.07), (x + 3.2, 5.45))
    arrow(ax, (x + 2.3, 4.82), (x + 3.0, 4.97))
    arrow(ax, (x + 2.3, 3.57), (x + 3.2, 4.5), ls="--", color=VER)
    ax.text(x + 0.2, 7.05, "glukosa sebelum makan", fontsize=8, va="center", color=GRY)
    finding(ax, x + 0.1, cw - 0.2,
            "PPG tidak memuat sinyal\nglukosa: r = 0,05 (23\nsubjek); ponsel (18\nsubjek): tak lebih baik\ndari rata-rata", VER)
    # --- step 2
    x = xs[1]
    ax.text(x + 0.15, 7.05, "jawab jika sim $\\geq$ 0,80 dan\nmargin $\\geq$ 0,05", fontsize=8, va="center", color=GRY,
            linespacing=1.15)
    box(ax, x + 0.15, 5.4, 0.8, 0.85, "Foto", fs=8)
    box(ax, x + 1.2, 5.4, 0.8, 0.85, "VLM", fs=8)
    box(ax, x + 2.3, 5.4, 1.65, 0.85, "Pencocok\nembedding", fs=8)
    arrow(ax, (x + 0.95, 5.82), (x + 1.2, 5.82))
    arrow(ax, (x + 2.0, 5.82), (x + 2.3, 5.82))
    box(ax, x + 2.3, 3.7, 1.65, 0.85, "Tabel\nhidangan", fs=8)
    arrow(ax, (x + 3.7, 5.4), (x + 3.7, 4.55), color=GRN)
    ax.text(x + 3.62, 4.97, "lolos", fontsize=8, color=GRN, va="center", ha="right")
    box(ax, x + 0.15, 3.7, 1.7, 0.85, "Isi manual", fs=8, ec=ORG)
    arrow(ax, (x + 2.5, 5.4), (x + 1.3, 4.55), ls="--", color=ORG)
    ax.text(x + 1.15, 5.0, "ragu", fontsize=8, color=ORG, va="center", ha="right")
    arrow(ax, (x + 1.85, 4.2), (x + 3.0, 5.4), rad=0.0, lw=0.7)
    box(ax, x + 0.15, 2.5, 3.8, 0.7, "karbohidrat, protein,\nlemak, serat (g)", fs=8)
    arrow(ax, (x + 3.1, 3.7), (x + 3.1, 3.2), lw=0.7)
    arrow(ax, (x + 1.0, 3.7), (x + 1.0, 3.2), lw=0.7)
    finding(ax, x + 0.1, cw - 0.2,
            "63,4% dijawab, 6,6% salah\ncocok (uji); lebih baik dari\nklasifikator tertutup yang\nselalu menjawab", GRN, h=2.1)
    # --- step 3
    x = xs[2]
    ax.text(x + 0.15, 7.05, "masukan: $g_0$, makro, jam,\nusia, jenis kelamin, IMT", fontsize=8, va="center",
            color=GRY, linespacing=1.15)
    box(ax, x + 0.15, 5.4, 1.8, 0.7, "LightGBM\nT+60", fs=8, bold=True)
    box(ax, x + 2.15, 5.4, 1.8, 0.7, "LightGBM\nT+120", fs=8, bold=True)
    ins = ax.inset_axes([x + 0.55, 2.7, 3.2, 2.0], transform=ax.transData)
    t = np.array([0, 60, 120])
    gl = np.array([112, 148, 134])
    ins.fill_between(t[1:], gl[1:] - 30, gl[1:] + 30, color=BLU, alpha=0.2, lw=0)
    ins.plot(t, gl, "-", color=BLU, lw=0.8)
    ins.plot(t[1:], gl[1:], "o", color=BLU, ms=3)
    ins.plot([0], [112], "s", color=BLK, ms=3.5)
    ins.set_xticks([0, 60, 120])
    ins.set_xticklabels(["makan", "+60", "+120"], fontsize=8)
    ins.set_yticks([])
    ins.patch.set_alpha(0)
    ins.set_xlim(-12, 135)
    ins.set_ylim(70, 200)
    ins.set_ylabel("glukosa", fontsize=8, labelpad=1)
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)
    ins.tick_params(length=1.5, pad=1)
    ins.text(78, 186, "pita galat", fontsize=8, color=BLU, ha="center", va="center")
    finding(ax, x + 0.1, cw - 0.2,
            "Orang tak terlihat, MAE\n(mg/dL): 28,8 vs 31,7 (B2)\npada T+60; 28,5 vs 32,1\npada T+120", BLU, h=2.1)
    arrow(ax, (xs[0] + cw, 4.97), (xs[1], 4.97), lw=1.0)
    arrow(ax, (xs[1] + cw, 4.97), (xs[2], 4.97), lw=1.0)
    save(fig, "fig_pipeline", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ----------------------------------------------------------------------------- fig 2
CLARKE_LINES = [
    ([0, 400], [0, 400], ":"), ([0, 175 / 3], [70, 70], "-"), ([175 / 3, 400 / 1.2], [70, 400], "-"),
    ([70, 70], [84, 400], "-"), ([0, 70], [180, 180], "-"), ([70, 290], [180, 400], "-"),
    ([70, 70], [0, 56], "-"), ([70, 400], [56, 320], "-"), ([180, 180], [0, 70], "-"),
    ([180, 400], [70, 70], "-"), ([240, 240], [70, 180], "-"), ([240, 400], [180, 180], "-"),
    ([130, 180], [0, 70], "-"),
]


def fig_forecast():
    fig = plt.figure(figsize=(COLW, 2.85))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 4.2], hspace=0.05, wspace=0.28,
                          left=0.095, right=0.985, bottom=0.15, top=0.93)
    axa = fig.add_subplot(gs[:, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1], sharex=ax_h)
    stats = {}
    for h, c, mk, off in ((60, BLU, "o", -0.18), (120, ORG, "s", 0.18)):
        d = pd.read_csv(p(f"forecast_per_participant_T{h}.csv"))
        d["gain"] = d.mae_b2 - d.mae_gbdt
        stats[h] = ((d.gain > 0).sum(), len(d), d.mae_gbdt.mean(), d.mae_b2.mean(), d.gain.median())
        if h == 60:
            order = d.sort_values("gain").participant.tolist()
            rank = {pp: i for i, pp in enumerate(order)}
        axa.plot(d.participant.map(rank), d.gain, mk, color=c, ms=3.2, mfc=c if h == 60 else "white",
                 mew=0.8, label=f"T+{h} menit", zorder=3)
    axa.axhline(0, color=BLK, lw=0.7)
    axa.set_xlabel("Partisipan (urut menurut selisih T+60)")
    axa.set_ylabel("MAE B2 $-$ MAE GBDT (mg/dL)")
    axa.set_xlim(-1.5, 45.5)
    axa.text(0.02, 0.97, "GBDT lebih baik", transform=axa.transAxes, fontsize=8, va="top", color=GRY)
    axa.text(0.02, 0.03, "B2 lebih baik", transform=axa.transAxes, fontsize=8, va="bottom", color=GRY)
    axa.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), handletextpad=0.1)
    axa.set_xticks([0, 10, 20, 30, 40])
    panel(axa, "(a)", dx=0.0, dy=1.01)
    # (b) Clarke
    m = pd.read_csv(p("forecast_meals_T60.csv"))
    zones = np.array([classify_zone(r, q) for r, q in zip(m.ref, m.gbdt)])
    za = 100 * (zones == "A").mean()
    zab = 100 * np.isin(zones, ["A", "B"]).mean()
    for xs_, ys_, ls in CLARKE_LINES:
        axb.plot(xs_, ys_, color=BLK if ls == "-" else GRY, lw=0.6, ls=ls)
    axb.scatter(m.ref, m.gbdt, s=3, color=BLU, alpha=0.35, lw=0, rasterized=True)
    for s_, (x_, y_) in {"A": (350, 330), "B": (370, 250), "C": (110, 370), "D": (300, 40), "E": (25, 370)}.items():
        axb.text(x_, y_, s_, fontsize=8, fontweight="bold", color=GRY, ha="center", va="center")
    axb.set_xlim(0, 400)
    axb.set_ylim(0, 400)
    axb.set_xlabel("Glukosa referensi, +60 mnt (mg/dL)")
    axb.set_ylabel("Glukosa prediksi (mg/dL)")
    axb.set_aspect("equal", adjustable="box")
    ax_h.hist(m.ref.clip(0, 400), bins=np.arange(0, 401, 10), color=GRY, lw=0)
    ax_h.set_yticks([])
    for sp in ("top", "right", "left"):
        ax_h.spines[sp].set_visible(False)
    plt.setp(ax_h.get_xticklabels(), visible=False)
    ax_h.tick_params(axis="x", length=0)
    panel(ax_h, "(b)", dx=0.0, dy=0.9)
    save(fig, "fig_forecast")
    plt.close(fig)
    return stats, za, zab, len(m)


# ----------------------------------------------------------------------------- fig 3
def fig_dish():
    d = pd.read_csv(p("dish_threshold_embedding.csv"))
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    cols = {0.0: SKY, 0.02: ORG, 0.05: VER}
    mk = {0.0: "o", 0.02: "^", 0.05: "s"}
    for split, ls, fill in (("dev", "-", True), ("test", "--", False)):
        for mg, c in cols.items():
            s = d[(d.split == split) & (d.margin == mg)].sort_values("threshold")
            ax.plot(100 * s.coverage, 100 * s.false_match_rate, ls=ls, color=c, lw=0.9, marker=mk[mg], ms=2.2,
                    mfc=c if fill else "white", mew=0.5)
    # chosen & old
    pts = {}
    for split, mfc in (("dev", BLK), ("test", "white")):
        r = d[(d.split == split) & (d.threshold == 0.8) & (d.margin == 0.05)].iloc[0]
        ax.plot(100 * r.coverage, 100 * r.false_match_rate, "*", color=BLK, mfc=mfc, ms=9, mew=0.8, zorder=5)
        o = d[(d.split == split) & (d.threshold == 0.75) & (d.margin == 0.0)].iloc[0]
        ax.plot(100 * o.coverage, 100 * o.false_match_rate, "D", color=BLK, mfc=mfc, ms=5, mew=0.8, zorder=5)
        pts[split] = (r, o)
    ax.annotate("dipilih\n(0,80; 0,05)", xy=(100 * pts["test"][0].coverage, 100 * pts["test"][0].false_match_rate),
                xytext=(30, 13), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5), ha="center")
    ax.annotate("lama\n(0,75; 0)", xy=(100 * pts["test"][1].coverage, 100 * pts["test"][1].false_match_rate),
                xytext=(66, 27), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5), ha="center")
    ax.axvline(60, color=GRY, lw=0.5, ls=":")
    ax.text(59, 31, "60%", fontsize=7.5, color=GRY, ha="right", va="top")
    ax.set_xlabel("Cakupan (% foto dijawab)")
    ax.set_ylabel("Tingkat salah cocok (% semua foto)")
    ax.set_xlim(0, 103)
    ax.set_ylim(0, 33)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cols[m_], marker=mk[m_], ms=3, lw=1, label=f"margin {m_:g}".replace(".", ",")) for m_ in cols]
    h += [Line2D([], [], color=BLK, ls="-", lw=1, label="dev (164)"),
          Line2D([], [], color=BLK, ls="--", lw=1, mfc="white", label="uji (317)")]
    ax.legend(handles=h, loc="upper left", ncol=1, handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)
    ax.text(0.99, 0.02, "terisi: dev, kosong: uji", transform=ax.transAxes, fontsize=7.5, color=GRY, ha="right")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_dish_operating")
    plt.close(fig)
    return pts


# ----------------------------------------------------------------------------- fig 4
def ident(ax, y, base, lab_pred=None):
    lo, hi = np.nanmin(y) - 12, np.nanmax(y) + 12
    ax.plot([lo, hi], [lo, hi], color=BLK, lw=0.8, label="identitas")
    ax.axhline(base, color=VER, lw=0.9, ls="--", label="rata-rata")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def fig_ppg():
    fig, axs = plt.subplots(1, 3, figsize=(COLW, 2.4), gridspec_kw=dict(wspace=0.5))
    a = pd.read_csv(p("ppg_dataset_served.csv"))
    ax = axs[0]
    ax.scatter(a.glucose_true, a.glucose_pred, s=7, color=BLU, alpha=0.7, lw=0)
    ident(ax, a.glucose_true, a.glucose_true.mean())
    r_a = np.corrcoef(a.glucose_true, a.glucose_pred)[0, 1]
    ax.set_xlabel("Glukosa sebenarnya (mg/dL)")
    ax.set_ylabel("Glukosa prediksi (mg/dL)")
    ax.legend(loc="upper left", handlelength=1.5, borderaxespad=0.2)
    panel(ax, "(a)", dx=0.0)
    b = pd.read_csv(p("butppg_finger_oof.csv"))
    ax = axs[1]
    s = b.groupby("session").agg(g=("glucose", "first"), pr=("pred", "mean"), sd=("pred", "std")).reset_index()
    ax.scatter(b.glucose, b.pred, s=2, color=SKY, alpha=0.25, lw=0, rasterized=True)
    ax.scatter(s.g, s.pr, s=12, color=BLU, lw=0.4, ec="white", zorder=3)
    ident(ax, b.glucose, b.glucose.mean())
    r_b = np.corrcoef(b.glucose, b.pred)[0, 1]
    ax.set_xlabel("Glukosa sebenarnya (mg/dL)")
    ax.set_ylabel("Glukosa prediksi (mg/dL)")
    panel(ax, "(b)", dx=0.0)
    c = pd.read_csv(p("butppg_within_person_delta.csv"))
    ax = axs[2]
    lim = 65
    ax.plot([-lim, lim], [-lim, lim], color=BLK, lw=0.8)
    ax.axhline(0, color=VER, lw=0.9, ls="--")
    ax.axvline(0, color=GRY, lw=0.5)
    ax.scatter(c.true_delta, c.pred_delta, s=14, color=BLU, lw=0.4, ec="white", zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$\\Delta$ glukosa sebenarnya")
    ax.set_ylabel("$\\Delta$ prediksi (mg/dL)")
    panel(ax, "(c)", dx=0.0)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.9)
    save(fig, "fig_ppg")
    plt.close(fig)
    sd = lambda v: float(np.std(v, ddof=0))  # noqa: E731
    return dict(r_a=r_a, n_a=len(a), sdp_a=sd(a.glucose_pred), sdt_a=sd(a.glucose_true), r_b=r_b, n_b=len(b),
                n_s=len(s), sdp_b=sd(b.pred), sdt_b=sd(b.glucose), sdp_c=sd(c.pred_delta), sdt_c=sd(c.true_delta),
                r_c=np.corrcoef(c.true_delta, c.pred_delta)[0, 1], n_c=len(c))


if __name__ == "__main__":
    prepare()
    fig_pipeline()
    st, za, zab, nm = fig_forecast()
    pts = fig_dish()
    pp = fig_ppg()
    print("forecast stats {h: (n_better, n, meanMAE_gbdt, meanMAE_b2, median_gain)}:", st)
    print(f"clarke T+60 pooled (n={nm}): A {za:.1f}%  A+B {zab:.1f}%")
    for k, (r, o) in pts.items():
        print(k, "chosen cov/fmr", r.coverage, r.false_match_rate, "old", o.coverage, o.false_match_rate)
    print(pp)
