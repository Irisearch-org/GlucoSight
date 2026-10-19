"""`integration/predictor.py` — default_predictor() and the opt-in
autoload path for the gated NLP dish-identity model (plan: wiring the
fusion pipeline, change 5)."""

from __future__ import annotations

import json
import os

from integration import predictor


def test_default_predictor_is_still_baseline_b2():
    """default_predictor() must never silently start returning a trained
    model — promotion to default requires a results table with B0/B1/B2
    beside it (C3), not just a passed gate."""
    assert isinstance(predictor.default_predictor(), predictor.BaselineB2Predictor)


def test_autoload_nlp_dish_identity_predictor_absent_by_default():
    """Expected state in this environment: train_and_persist() has not
    been run against the real ShanghaiT2DM data through this path, so no
    artifact exists at the real ARTIFACT_PATH yet."""
    from nlp.models.dish_identity_ridge import ARTIFACT_PATH
    assert not os.path.exists(ARTIFACT_PATH)
    assert predictor.autoload_nlp_dish_identity_predictor() is None


def test_autoload_nlp_dish_identity_predictor_loads_a_real_artifact():
    from nlp.models.dish_identity_ridge import ARTIFACT_PATH, NLPDishIdentityPredictor

    blob = {
        "model_version": "test-autoload-v1",
        "target": "rise_t60",
        "continuous_names": ["g0_mgdl", "total_grams"],
        "flag_names": ["is_fried_cooking", "is_large_portion"],
        "dish_vocabulary": ["rice"],
        "min_dish_meals": 10,
        "scaler_mean": [130.0, 300.0],
        "scaler_scale": [25.0, 150.0],
        "coef": [0.0, 0.0, 0.0, 0.0, 5.0],
        "intercept": 10.0,
        "alpha": 1.0,
        "n_meals": 100, "n_subjects": 20,
        "heldout_mae_m3": 30.0, "heldout_mae_m2": 31.0,
        "bootstrap_ci": [0.1, 1.0], "permutation_p": 0.01,
    }
    assert not os.path.exists(ARTIFACT_PATH), (
        "refusing to overwrite a real artifact that shouldn't exist yet "
        "in this dev environment"
    )
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(blob, fh)
    try:
        est = predictor.autoload_nlp_dish_identity_predictor()
        assert isinstance(est, NLPDishIdentityPredictor)
        assert est.model_version == "test-autoload-v1"
    finally:
        os.remove(ARTIFACT_PATH)
