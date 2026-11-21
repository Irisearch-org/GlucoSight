"""Typed stage errors. `integration/api.py` maps each to one HTTP status; nothing is swallowed."""

from __future__ import annotations


class StageError(Exception):
    """Base for errors a stage raises on purpose. `detail` is the Bahasa message shown to the user."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class InvalidInput(StageError, ValueError):
    """The request is well-formed JSON but breaks a domain rule (HTTP 422)."""


class ModelUnavailable(StageError, RuntimeError):
    """A served artifact (PPG model, GBDT, dish table, Gemini key) is missing or failed to load (HTTP 503)."""
