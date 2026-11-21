"""Re-save the served PPG estimator so the server can load it without torch.

`serve.py` writes `rppg/models/weights/ppg_served_model.pt` with torch.save; the object inside is a
plain scikit-learn pipeline (StandardScaler + Ridge on the 7 handcrafted features). The integration
server (`integration/stages/glucose.py`) loads the joblib sibling instead. Run once after `serve.py`
(needs torch only here):

    python3 rppg/experiments/ppg_served_model/export_joblib.py \
        [--src rppg/models/weights/ppg_served_model.pt] [--out rppg/models/weights/ppg_served_model.joblib]
"""
import argparse
import os

import joblib
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WEIGHTS = os.path.join(REPO_ROOT, "rppg", "models", "weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(WEIGHTS, "ppg_served_model.pt"))
    ap.add_argument("--out", default=os.path.join(WEIGHTS, "ppg_served_model.joblib"))
    args = ap.parse_args()

    blob = torch.load(args.src, weights_only=False)
    if blob["feature_key"] != "hand":
        raise SystemExit(f"served feature_key is {blob['feature_key']!r}; the server only supports 'hand'")
    joblib.dump({"estimator": blob["estimator"], "name": blob["name"], "feature_key": blob["feature_key"]},
                args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
