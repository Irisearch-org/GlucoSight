"""`integration/predictor.py` — default_predictor() and the opt-in
autoload path for the gated NLP dish-identity model (plan: wiring the
fusion pipeline, change 5)."""

from __future__ import annotations

import os

from integration import predictor


def test_default_predictor_is_still_baseline_b2():
    """default_predictor() must never silently start returning a trained
    model — promotion to default requires a results table with B0/B1/B2
    beside it (C3), not just a passed gate."""
    assert isinstance(predictor.default_predictor(), predictor.BaselineB2Predictor)


def test_autoload_nlp_dish_identity_predictor_loads_the_real_trained_artifact():
    """nlp.models.dish_identity_ridge --train has been run against the real
    committed ShanghaiT2DM data and its gate passed (5/5 folds, bootstrap CI
    [0.20, 1.07] excludes zero, permutation p=0.001 -- see
    nlp/models/reports/dish_identity_ridge.md), so the artifact is
    committed and this mode is live, not a no-op."""
    from nlp.models.dish_identity_ridge import ARTIFACT_PATH, NLPDishIdentityPredictor
    assert os.path.exists(ARTIFACT_PATH)
    est = predictor.autoload_nlp_dish_identity_predictor()
    assert isinstance(est, NLPDishIdentityPredictor)
    assert est.model_version == "nlp-dish-identity-ridge-v1"


def test_autoload_nlp_dish_identity_predictor_none_when_artifact_missing():
    """The no-op path still works when nobody has trained a model yet.

    load()'s default path is bound to ARTIFACT_PATH at function-definition
    time, so this can't be exercised by monkeypatching the module attribute
    -- it has to move the real file aside and put it back."""
    from nlp.models.dish_identity_ridge import ARTIFACT_PATH
    assert os.path.exists(ARTIFACT_PATH)
    moved = ARTIFACT_PATH + ".movedfortest"
    os.rename(ARTIFACT_PATH, moved)
    try:
        assert predictor.autoload_nlp_dish_identity_predictor() is None
    finally:
        os.rename(moved, ARTIFACT_PATH)
