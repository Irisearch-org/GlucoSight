"""Population-mean (B0) baseline for CGMacros meal features.

This baseline intentionally does not accept or read an image.  It is the
trivial reference that every later CV model must beat on the same meals.
"""

from typing import Dict


B0_MODEL_VERSION = "cv-b0-population-mean-v1"


def predict_population_mean(meal_id: str) -> Dict[str, object]:
    """Return the fixed CV feature block for one meal.

    ``meal_id`` is validated so callers cannot accidentally emit anonymous
    predictions.  It is part of the meal envelope, not duplicated inside the
    returned CV feature block.
    """

    if not isinstance(meal_id, str) or not meal_id.strip():
        raise ValueError("meal_id must be a non-empty string")

    return {
        "carbs_g": 45.0,
        "protein_g": 18.0,
        "fat_g": 12.0,
        "fiber_g": 4.0,
        "gi_category": 1,
        "food_class": "unknown",
        "cv_confidence": 0.0,
        "cv_present": 1,
        "carbs_source": "population_mean",
        "portion_reported": 1,
        "cv_model_version": B0_MODEL_VERSION,
    }
