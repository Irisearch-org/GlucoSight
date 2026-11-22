"""Print the REPORT.md tables from results/*.json (so the report quotes files, not memory)."""
import json
import os

from common import RESULTS_DIR


def load(n):
    p = os.path.join(RESULTS_DIR, n)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(x):
    return f"[{x[0]:+.2f}, {x[1]:+.2f}]"


def main():
    p = load("ppg27_primary.json")
    if p:
        print("### #27 primary (finger, still), recording level\n")
        print("| model | n subj / rec / sess | MAE | mean MAE | RMSE | r | gain | 95% CI | folds won | perm p (N) | gate |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, r in p.items():
            a = r["recording"]
            print(f"| {k} | {r['n_subjects']} / {r['n_recordings']} / {r['n_sessions']} | {a['mae']} | {a['baseline_mae']} | {a['rmse']} | {a['pearson_r']} | "
                  f"{a['gain']:+.2f} | {ci(a['ci95'])} | {a['fold_wins'][0]}/{a['fold_wins'][1]} | {a['perm_p']} ({r['permutations']}) | {'PASS' if r['gate_recording']['passes'] else 'FAIL'} |")
        print("\n### #27 primary, session level (predictions averaged within session)\n")
        print("| model | MAE | mean MAE | RMSE | r | gain | 95% CI | perm p | gate |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, r in p.items():
            a = r["session"]
            print(f"| {k} | {a['mae']} | {a['baseline_mae']} | {a['rmse']} | {a['pearson_r']} | {a['gain']:+.2f} | {ci(a['ci95'])} | {a['perm_p']} | {'PASS' if r['gate_session']['passes'] else 'FAIL'} |")
        print("\n### Clarke (out-of-fold, recording level)\n")
        print("| model | zone A pooled % | per-subject A median [IQR width] | A by reference range (mg/dL): n, A% |")
        print("|---|---|---|---|")
        for k, r in list(p.items()) + [("mean baseline", dict(clarke=next(iter(p.values()))["clarke_baseline"]))]:
            cl = r["clarke"]
            rng = "; ".join(f"{a} n={v['n_recordings']} A={v['zone_a_pct']}" for a, v in cl["zone_a_by_range"].items() if v["n_recordings"])
            print(f"| {k} | {cl['zone_a_pooled_pct']} | {cl['zone_a_per_subject_median']} [{cl['zone_a_per_subject_iqr']}] | {rng} |")
        print("\n### tracking of OOF predictions\n")
        for k, r in p.items():
            print(f"- {k}: {r['tracking']}; prediction SD {r['pred_sd_mgdl']} mg/dL")
    s = load("ppg27_sens.json")
    if s:
        print("\n### #27 sensitivities (ungated)\n")
        print("| subset | features+model | cv | n subj/rec | MAE | mean MAE | gain | 95% CI | folds won | r | session gain [CI] |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, r in s.items():
            a = r["recording"]; b = r["session"]
            print(f"| {r['subset']} | {r['features']}+{r['model']} | {'LOSO' if 'LOSO' in k else '5x5'} | {r['n_subjects']}/{r['n_recordings']} | {a['mae']} | {a['baseline_mae']} | {a['gain']:+.2f} | {ci(a['ci95'])} | {a['fold_wins'][0]}/{a['fold_wins'][1]} | {a['pearson_r']} | {b['gain']:+.2f} {ci(b['ci95'])} |")
    d = load("ppg28.json")
    if d:
        for coh, r in d.items():
            print(f"\n## #28 cohort {coh}: {r['n_two_session_subjects']} two-session subjects, {r['n_recordings']} recordings\n")
            a = r["a_delta_features"]
            print("delta glucose:", a["delta_glucose_mgdl"])
            print("\n| feature | rho | p | p Holm |\n|---|---|---|---|")
            for f in a["spearman_hand"]:
                print(f"| {f['feature']} | {f['rho']} | {f['p']} | {f['p_holm']} |")
            print("\n| delta-ridge LOSO | MAE delta | zero-baseline MAE | gain [CI] | subj better | sign acc | majority rate | binom p | perm p sign | perm p gain | N perm |\n|---|---|---|---|---|---|---|---|---|---|---|")
            for k, v in a["delta_ridge_loso"].items():
                print(f"| {k} | {v['mae_delta_model']} | {v['mae_delta_zero_baseline']} | {v['gain']:+.2f} {ci(v['ci95'])} | {v['subjects_model_better']}/{r['n_two_session_subjects']} | {v['sign_correct']} | {v['majority_sign_rate']} | {v['binom_p_two_sided']} | {v['perm_p_sign_acc']} | {v['perm_p_gain']} | {v['n_perm']} |")
            print("\ndelta-ridge as within-subject calibration:", r["delta_ridge_as_calibration"])
            print("\n| calibration sim | pairs | MAE calibrated | MAE carry-over | MAE mean | MAE uncalibrated | vs carry gain [CI] | subj better | perm p | vs mean gain [CI] | perm p | gate vs carry | model-delta r / sign |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for k, v in r["b_calibration"].items():
                c1, c2 = v["calibrated_vs_carry"], v["calibrated_vs_mean"]
                md = v["model_delta_vs_true_delta"]
                print(f"| {k} | {v['n_pairs']} | {v['mae']['calibrated']} | {v['mae']['carry']} | {v['mae']['mean']} | {v['mae']['model_dst']} | {c1['gain']:+.2f} {ci(c1['ci95'])} | {c1['subjects_calibration_better']}/{c1['of_subjects']} | {c1.get('perm_p')} | {c2['gain']:+.2f} {ci(c2['ci95'])} | {c2.get('perm_p')} | {('PASS' if v['gate_vs_carry']['passes'] else 'FAIL') if 'gate_vs_carry' in v else 'n/a'} | r={md['pearson']} sign {md['sign_correct']}/{md['of']} |")


if __name__ == "__main__":
    main()
