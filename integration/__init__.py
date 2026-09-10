"""GlucoSight integration layer — the PM-owned fusion spine.

    integration.contract   Interface Contract v1.2: the single validator.
    integration.adapters   Track-native output -> contract records.
    integration.fusion     Assemble one validated meal record.
    integration.features   Feature tiers + train-only normalization.
    integration.predictor  The prediction step (currently a labelled baseline).
    integration.api        FastAPI MVP endpoint.
"""

__version__ = "0.1.0"
