"""HTTP contract of the v2 endpoints, via TestClient with the pipeline's fake-seam Deps."""

import base64

import pytest
from fastapi.testclient import TestClient

from integration.api import app
from integration.stages.meal import VisionUnavailable

from .conftest import NOW, ppg_capture

PROFILE = {"age_years": 52, "sex": "F", "height_cm": 160, "weight_kg": 70, "diabetes_type": "Tipe 2"}


@pytest.fixture
def client(deps):
    original = app.state.deps
    app.state.deps = deps
    yield TestClient(app)
    app.state.deps = original


def _predict_body(**patch) -> dict:
    body = {
        "t0_timestamp": "2026-01-15T12:00:00Z",
        "glucose": {"source": "meter", "value_mgdl": 123, "timestamp": "2026-01-15T11:50:00Z"},
        "meal": {"items": [{"row_id": "TST01", "grams": 150, "source": "manual"},
                           {"row_id": "TST03", "grams": 50, "source": "manual"}],
                 "vlm_model_version": None},
        "profile": PROFILE,
    }
    body.update(patch)
    return body


def test_health_reports_what_is_served(client, served, table):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["schema_version"] == "2.0"
    assert body["served"] == served
    assert body["dish_table_build"] == table.build_hash


def test_dishes_autocomplete_returns_per_100g_rows(client):
    r = client.get("/dishes", params={"q": "nas", "limit": 5})
    assert r.status_code == 200
    (row,) = r.json()["items"]
    assert set(row) == {"id", "name", "kind", "serving_g", "energy_kcal", "carbs_g", "protein_g",
                        "fat_g", "fibre_g", "confidence"}
    assert row["id"] == "TST01" and row["carbs_g"] == 40.0


def test_dishes_rejects_a_bad_limit_with_a_bahasa_detail(client):
    r = client.get("/dishes", params={"q": "nas", "limit": 0})
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    assert r.json()["detail"].startswith("Masukan tidak valid")


def test_ppg_scan_returns_a_g0_reading(client):
    r = client.post("/glucose/ppg", json={"capture": ppg_capture()})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "ppg" and body["capture_valid"] is True
    assert body["error_range_mgdl"] == 25.0 and body["age_minutes"] == 0.0
    assert 40 <= body["value_mgdl"] <= 400 and body["heart_rate_bpm"] is not None


def test_invalid_scan_is_409_with_the_reason(client):
    r = client.post("/glucose/ppg", json={"capture": ppg_capture(torch_verified=False)})
    assert r.status_code == 409 and "senter" in r.json()["detail"]


def test_photo_returns_accepted_and_manual_items(client, table):
    url = "data:image/jpeg;base64," + base64.b64encode(b"img").decode()
    r = client.post("/meal/photo", json={"image_data_url": url})
    assert r.status_code == 200
    body = r.json()
    assert body["dish_table_build"] == table.build_hash and body["vlm_model_version"] == "fake-vlm"
    first, second = body["items"]
    assert first["accepted"] is True and first["row"]["id"] == "TST01"
    assert second["accepted"] is False and second["row"] is None and len(second["suggestions"]) == 3


def test_photo_with_a_bad_data_url_is_422(client):
    assert client.post("/meal/photo", json={"image_data_url": "nope"}).status_code == 422


def test_photo_when_the_vision_service_is_down_is_503(client, deps):
    class Down:
        model_version = "fake-vlm"

        def name_dishes(self, image, mime):
            raise VisionUnavailable("down")

    deps.namer = Down()
    url = "data:image/jpeg;base64," + base64.b64encode(b"img").decode()
    r = client.post("/meal/photo", json={"image_data_url": url})
    assert r.status_code == 503 and "manual" in r.json()["detail"]


def test_predict_meter_and_two_manual_items(client):
    r = client.post("/predict", json=_predict_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == "2.0" and body["outside_training"] is False
    assert set(body) == {"schema_version", "g0", "meal", "forecast", "outside_training", "disclaimer"}
    assert set(body["forecast"]["t60"]) == {"glucose_mgdl", "range", "error_band_mgdl", "model_version"}
    assert body["meal"]["totals"]["carbs_g"] == 66.0
    assert body["meal"]["items"][0]["row"]["id"] == "TST01"


def test_predict_after_a_ppg_scan(client):
    reading = client.post("/glucose/ppg", json={"capture": ppg_capture()}).json()
    r = client.post("/predict", json=_predict_body(glucose={"source": "ppg", "reading": reading}))
    assert r.status_code == 200, r.text
    assert r.json()["g0"]["source"] == "ppg"


def test_predict_with_a_stale_reading_is_409(client):
    body = _predict_body(glucose={"source": "meter", "value_mgdl": 120,
                                  "timestamp": "2026-01-15T10:00:00Z"})
    r = client.post("/predict", json=body)
    assert r.status_code == 409
    assert r.json()["detail"] == "Hasil pengukuran sudah lebih dari 60 menit. Silakan ukur ulang."


def test_predict_with_an_out_of_range_value_is_422(client):
    body = _predict_body(glucose={"source": "meter", "value_mgdl": 500,
                                  "timestamp": "2026-01-15T11:55:00Z"})
    assert client.post("/predict", json=body).status_code == 422


def test_predict_with_an_unknown_row_is_422(client):
    body = _predict_body(meal={"items": [{"row_id": "NOPE", "grams": 10, "source": "manual"}]})
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and "NOPE" in r.json()["detail"]


def test_predict_with_an_empty_meal_is_422(client):
    assert client.post("/predict", json=_predict_body(meal={"items": []})).status_code == 422


def test_predict_with_a_missing_diabetes_type_is_422(client):
    r = client.post("/predict", json=_predict_body(profile={"age_years": 50}))
    assert r.status_code == 422 and isinstance(r.json()["detail"], str)


def test_predict_without_the_forecaster_is_503(client, deps):
    deps.forecaster = None
    assert client.post("/predict", json=_predict_body()).status_code == 503


@pytest.mark.parametrize("path", ["/ppg", "/contract"])
def test_retired_routes_are_gone(client, path):
    assert client.get(path).status_code == 404


def test_auth_and_user_routers_are_still_mounted():
    paths = set(app.openapi()["paths"])
    assert any(p.startswith("/auth") for p in paths)
    assert any(p.startswith("/user") for p in paths)


def test_real_served_artifacts_with_the_locally_built_dish_table(served):
    """One smoke through the real forecaster and the real (gitignored) dish table."""
    from integration.pipeline import REPO_ROOT, Deps

    if not (REPO_ROOT / served["dish_table"]["path"]).is_file():
        pytest.skip("dish table not built (python -m integration.dishes.build)")
    real = Deps.from_served(served)
    assert real.table is not None and real.forecaster is not None
    real.now = lambda: NOW
    original = app.state.deps
    app.state.deps = real
    try:
        ids = [r["id"] for r in TestClient(app).get("/dishes", params={"q": "nasi", "limit": 2}).json()["items"]]
        body = _predict_body(meal={"items": [{"row_id": i, "grams": 100, "source": "manual"} for i in ids]})
        r = TestClient(app).post("/predict", json=body)
    finally:
        app.state.deps = original
    assert r.status_code == 200, r.text
    assert r.json()["meal"]["dish_table_build"] == real.table.build_hash
