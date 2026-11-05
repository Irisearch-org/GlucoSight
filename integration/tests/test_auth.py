"""Auth endpoints against a real MariaDB/MySQL, in a throwaway `glucosight_test` database.

Skipped when no database server is reachable.
"""

import os
import uuid

import pytest

os.environ["GLUCOSIGHT_DB_NAME"] = "glucosight_test"

from fastapi import FastAPI
from fastapi.testclient import TestClient

from integration import auth, db


@pytest.fixture(scope="module")
def client():
    try:
        db.init_schema()
    except Exception as exc:  # no server, wrong credentials, ...
        pytest.skip(f"database unavailable: {exc}")
    app = FastAPI()
    app.include_router(auth.router)
    yield TestClient(app)
    conn = db.connect(with_database=False)
    try:
        with conn.cursor() as cur:
            cur.execute("DROP DATABASE IF EXISTS glucosight_test")
    finally:
        conn.close()


def _creds():
    return {"name": "Budi Test", "email": f"{uuid.uuid4().hex[:8]}@example.com", "password": "correct-horse"}


def test_register_login_logout_roundtrip(client):
    c = _creds()
    r = client.post("/auth/register", json=c)
    assert r.status_code == 201
    assert r.json()["user"]["email"] == c["email"]
    assert "password" not in r.text
    assert client.get("/auth/me").json()["user"]["name"] == "Budi Test"

    assert client.post("/auth/logout").status_code == 204
    assert client.get("/auth/me").status_code == 401

    r = client.post("/auth/login", json={"email": c["email"].upper(), "password": c["password"]})
    assert r.status_code == 200
    assert client.get("/auth/me").status_code == 200


def test_logout_invalidates_the_token_server_side(client):
    c = _creds()
    client.post("/auth/register", json=c)
    token = client.cookies.get(auth.COOKIE_NAME)
    client.post("/auth/logout")
    client.cookies.set(auth.COOKIE_NAME, token)  # replay the old cookie
    assert client.get("/auth/me").status_code == 401


def test_wrong_password_and_unknown_email_look_the_same(client):
    c = _creds()
    client.post("/auth/register", json=c)
    client.post("/auth/logout")
    bad = client.post("/auth/login", json={"email": c["email"], "password": "nope-nope"})
    unknown = client.post("/auth/login", json={"email": "ghost@example.com", "password": "nope-nope"})
    assert bad.status_code == unknown.status_code == 401
    assert bad.json() == unknown.json()


def test_duplicate_email_and_validation(client):
    c = _creds()
    assert client.post("/auth/register", json=c).status_code == 201
    assert client.post("/auth/register", json=c).status_code == 409
    assert client.post("/auth/register", json={**_creds(), "password": "short"}).status_code == 422
    assert client.post("/auth/register", json={**_creds(), "email": "not-an-email"}).status_code == 422


def test_password_is_stored_hashed(client):
    c = _creds()
    client.post("/auth/register", json=c)
    with db.cursor() as cur:
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (c["email"],))
        stored = cur.fetchone()["password_hash"]
    assert stored.startswith("scrypt$") and c["password"] not in stored
    assert auth.verify_password(c["password"], stored)
    assert not auth.verify_password("wrong", stored)


def test_lockout_after_repeated_failures(client):
    c = _creds()
    client.post("/auth/register", json=c)
    client.post("/auth/logout")
    for _ in range(auth.MAX_FAILURES):
        assert client.post("/auth/login", json={"email": c["email"], "password": "bad-bad-bad"}).status_code == 401
    assert client.post("/auth/login", json={"email": c["email"], "password": c["password"]}).status_code == 429
