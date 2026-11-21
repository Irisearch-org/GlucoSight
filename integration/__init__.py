"""GlucoSight integration layer (contract v2.0).

    integration.api         HTTP only: /health /dishes /glucose/ppg /meal/photo /predict, auth, user data.
    integration.pipeline    forecast_meal + the step operations; `Deps` holds the served components.
    integration.stages      glucose (step 1), meal (step 2), forecast (step 3).
    integration.dishes      the dish table and the embedding matcher.
    integration.contract    the schema: domain dataclasses and HTTP bodies.
    integration.models      served.json: what is served, with versions and error bands.
"""

__version__ = "2.0.0"
