"""Track-native output -> Interface Contract v1.2.

Each track keeps the schema it already ships. Nothing downstream reads a
track's native record; the adapter is the only translation point, so a
track can change its internals without breaking Forecasting, and a schema
drift shows up here as a failing test rather than as a silent column of
fallback values.
"""

from integration.adapters.cv import cv_to_contract
from integration.adapters.nlp import nlp_to_contract
from integration.adapters.ppg import ppg_to_contract

__all__ = ["cv_to_contract", "nlp_to_contract", "ppg_to_contract"]
