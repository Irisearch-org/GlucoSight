"""Profile and history endpoints against a real MariaDB/MySQL, in a throwaway `glucosight_test` database.

Skipped when no database server is reachable.
"""

import os
import uuid

import pytest

os.environ["GLUCOSIGHT_DB_NAME"] = "glucosight_test"

from fastapi import FastAPI
from fastapi.testclient import TestClient

from integration import auth, db, user_data


@pytest.fixture(scope="module")
def app():
    try:
        db.init_schema()
    except Exception as exc:  # no server, wrong credentials, ...
        pytest.skip(f"database unavailable: {exc}")
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(user_data.router)
    yield app
    conn = db.connect(with_database=False)
    try:
        with conn.cursor() as cur:
            cur.execute("DROP DATABASE IF EXISTS glucosight_test")
    finally:
        conn.close()


def _signed_in(app):
    client = TestClient(app)
    creds = {"name": "Siti Test", "email": f"{uuid.uuid4().hex[:8]}@example.com", "password": "correct-horse"}
    assert client.post("/auth/register", json=creds).status_code == 201
    return client, creds


def _record(meal_id=None, created="2026-11-04T03:15:00.000Z"):
    return {
        "id": meal_id or uuid.uuid4().hex, "createdAt": created, "food": "Nasi goreng", "portion": "Sedang",
        "t60": 171.5, "t120": 150.0, "range": "borderline", "g0": 110, "sourceId": "meter",
        "basis": "baseline", "usageMessage": "g0 only", "modelVersion": "baseline-b2-v1",
    }


def test_endpoints_require_login(app):
    client = TestClient(app)
    for method, path in [("get", "/user/profile"), ("get", "/user/history")]:
        assert getattr(client, method)(path).status_code == 401
    assert client.post("/user/history", json=_record()).status_code == 401


def test_profile_and_history_survive_logout_and_login(app):
    client, creds = _signed_in(app)

    first = client.get("/user/profile").json()
    assert first["saved"] is False and first["profile"]["email"] == creds["email"]

    saved = client.put("/user/profile", json={
        "name": "Siti Aminah", "age": "45", "gender": "Wanita", "heightCm": "160.5", "weightKg": "62",
        "diabetesType": "Tipe 2", "defaultSource": "cgm",
    })
    assert saved.status_code == 200
    older, newer = _record(created="2026-11-03T01:00:00Z"), _record(created="2026-11-04T01:00:00Z")
    assert client.post("/user/history", json=older).status_code == 201
    assert client.post("/user/history", json=newer).status_code == 201
    assert client.post("/user/history", json=newer).status_code == 201  # same id again: no duplicate

    assert client.post("/auth/logout").status_code == 204
    assert client.get("/user/profile").status_code == 401
    assert client.post("/auth/login", json={"email": creds["email"], "password": creds["password"]}).status_code == 200

    got = client.get("/user/profile").json()
    assert got["saved"] is True
    assert got["profile"] | {"participantId": None} == {
        "participantId": None, "name": "Siti Aminah", "email": creds["email"], "age": "45", "gender": "Wanita",
        "heightCm": "160.5", "weightKg": "62", "diabetesType": "Tipe 2", "defaultSource": "cgm",
    }
    history = client.get("/user/history").json()["history"]
    assert [h["id"] for h in history] == [newer["id"], older["id"]]  # newest first, no duplicate
    assert history[0] == {**newer, "createdAt": "2026-11-04T01:00:00.000Z"}


def test_accounts_cannot_see_each_others_data(app):
    alice, _ = _signed_in(app)
    bob, _ = _signed_in(app)
    alice.post("/user/history", json=_record())
    alice.put("/user/profile", json={"name": "Alice", "age": "30"})
    assert bob.get("/user/history").json()["history"] == []
    assert bob.get("/user/profile").json()["profile"]["age"] == ""


def test_profile_validation(app):
    client, _ = _signed_in(app)
    base = {"name": "Siti Aminah"}
    assert client.put("/user/profile", json={**base, "age": "abc"}).status_code == 422
    assert client.put("/user/profile", json={**base, "age": "200"}).status_code == 422
    assert client.put("/user/profile", json={**base, "defaultSource": "bogus"}).status_code == 422
    assert client.put("/user/profile", json={**base, "age": "", "heightCm": "", "weightKg": ""}).status_code == 200
