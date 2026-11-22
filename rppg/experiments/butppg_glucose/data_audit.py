"""Step 0: which recordings belong to which (subject, session), and how many survive each filter."""
import json
import os

import numpy as np

from common import RESULTS_DIR, load_cache, load_meta


def main():
    m = load_meta()
    c = load_cache()
    ok = np.isfinite(c["h_bright_raw"]).all(1) & np.isfinite(c["e_bright_inv"]).all(1)
    assert (c["ID"] == m["ID"].to_numpy()).all()
    m["feat_ok"] = ok
    m["clip_ok"] = c["clip"] <= 0.10
    per_sess = m.groupby("session_id").agg(subj=("subj", "first"), glucose_mmol=("gl_mmol", "first"),
                                           order=("sess_order", "first"), site=("site", "first"),
                                           first_rec=("rec", "min"), last_rec=("rec", "max"),
                                           n_all=("ID", "size"), n_still=("still", "sum"),
                                           n_still_feat_ok=("feat_ok", lambda s: int((s & m.loc[s.index, "still"]).sum())))
    per_sess = per_sess.sort_values(["subj", "order"])
    sj = m.groupby("subj").agg(site=("site", "first"), n_sessions_by_glucose=("session_id", "nunique"),
                               n_all=("ID", "size"), n_still=("still", "sum"))
    out = {
        "n_recordings_with_glucose": int(len(m)), "n_subjects": int(m["subj"].nunique()),
        "glucose_mgdl_range": [round(float(m.glucose.min()), 1), round(float(m.glucose.max()), 1)],
        "subjects_by_site": m.groupby("site")["subj"].nunique().to_dict(),
        "site_is_constant_within_subject": bool((m.groupby("subj")["site"].nunique() == 1).all()),
        "recordings_by_site_and_motion": {f"{a}|{'still' if b else 'motion'}": int(n) for (a, b), n in m.groupby(["site", "still"]).size().items()},
        "two_distinct_glucose_subjects_by_site": sj[sj.n_sessions_by_glucose == 2].groupby("site").size().to_dict(),
        "single_glucose_subjects": {str(i): r.site for i, r in sj[sj.n_sessions_by_glucose == 1].iterrows()},
        "recordings_per_session_all": {"min": int(per_sess.n_all.min()), "median": float(per_sess.n_all.median()), "max": int(per_sess.n_all.max())},
        "finger_still": {"recordings": int(((m.site == "finger") & m.still).sum()),
                         "recordings_finite_features": int(((m.site == "finger") & m.still & m.feat_ok).sum()),
                         "subjects": int(m[(m.site == "finger") & m.still].subj.nunique()),
                         "sessions": int(per_sess[per_sess.site == "finger"].shape[0]),
                         "recordings_per_session_still": {"min": int(per_sess[per_sess.site == "finger"].n_still.min()),
                                                          "median": float(per_sess[per_sess.site == "finger"].n_still.median()),
                                                          "max": int(per_sess[per_sess.site == "finger"].n_still.max())}},
        "brightest_channel_counts_rgb": {s: np.bincount(c["bright"][c["site"] == s], minlength=3).tolist() for s in ("finger", "ear")},
        "frac_clip_gt_0.10_finger": round(float((c["clip"][c["site"] == "finger"] > 0.10).mean()), 3),
        "frac_clip_gt_0.10_ear": round(float((c["clip"][c["site"] == "ear"] > 0.10).mean()), 3),
        "recordings_nonfinite_features": {"all": int((~ok).sum()), "finger_still": int(((m.site == "finger") & m.still & ~m.feat_ok).sum())},
        "sessions": per_sess.reset_index().astype({"order": int, "subj": int}).to_dict("records"),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "data_audit.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print({k: v for k, v in out.items() if k != "sessions"})


if __name__ == "__main__":
    main()
