"""Account registration, login and logout, backed by the `glucosight` database.

    POST /auth/register   create an account and sign in
    POST /auth/login      sign in
    POST /auth/logout     end the current session
    GET  /auth/me         who is signed in (401 when nobody is)

A session is an opaque random token in an HttpOnly cookie. The database keeps
only its SHA-256, so logging out (deleting the row) really ends the session
and a leaked table cannot be replayed. Passwords are hashed with scrypt from
the standard library.

Set GLUCOSIGHT_COOKIE_SECURE=1 when serving over HTTPS.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
import uuid
from typing import Optional

import pymysql
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from integration import db

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_NAME = "glucosight_session"
SESSION_DAYS = 7
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Brute-force guard: at most MAX_FAILURES wrong passwords per email per window.
MAX_FAILURES = 5
FAILURE_WINDOW_S = 15 * 60
_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


# ------------------------------------------------------------------ passwords

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt),
            n=int(n), r=int(r), p=int(p), dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Verified against when the email is unknown, so "no such user" and "wrong
# password" take the same time.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


# ------------------------------------------------------------------- sessions

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _start_session(response: Response, request: Request, user_id: int) -> None:
    token = secrets.token_urlsafe(32)
    with db.cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE expires_at < NOW()")
        cur.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at, user_agent) "
            "VALUES (%s, %s, DATE_ADD(NOW(), INTERVAL %s DAY), %s)",
            (_token_hash(token), user_id, SESSION_DAYS, (request.headers.get("user-agent") or "")[:255]),
        )
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax", path="/",
        secure=os.environ.get("GLUCOSIGHT_COOKIE_SECURE") == "1",
    )


def _user_for_request(request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    with db.cursor() as cur:
        cur.execute(
            "SELECT u.id, u.participant_id, u.name, u.email FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = %s AND s.expires_at > NOW()",
            (_token_hash(token),),
        )
        return cur.fetchone()


def _public(user: dict) -> dict:
    return {"participant_id": user["participant_id"], "name": user["name"], "email": user["email"]}


def _db_unavailable() -> HTTPException:
    return HTTPException(503, "Database tidak tersedia. Pastikan MySQL berjalan lalu coba lagi.")


def require_user(request: Request) -> dict:
    """The signed-in user row (id, participant_id, name, email), or 401. For other routers."""
    try:
        user = _user_for_request(request)
    except pymysql.err.OperationalError:
        raise _db_unavailable()
    if not user:
        raise HTTPException(401, "Belum masuk.")
    return user


# ---------------------------------------------------------------------- models

class RegisterBody(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginBody(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)


# ------------------------------------------------------------------- endpoints

@router.post("/register", status_code=201)
def register(body: RegisterBody, request: Request, response: Response) -> dict:
    email = body.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(422, "Format email tidak valid.")
    try:
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO users (participant_id, name, email, password_hash) VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), body.name.strip(), email, hash_password(body.password)),
            )
            user_id = cur.lastrowid
            cur.execute("SELECT participant_id, name, email FROM users WHERE id = %s", (user_id,))
            user = cur.fetchone()
        _start_session(response, request, user_id)
    except pymysql.err.IntegrityError:
        raise HTTPException(409, "Email sudah terdaftar. Silakan masuk.")
    except pymysql.err.OperationalError:
        raise _db_unavailable()
    return {"user": _public(user)}


@router.post("/login")
def login(body: LoginBody, request: Request, response: Response) -> dict:
    email = body.email.strip().lower()
    now = time.monotonic()
    with _failures_lock:
        recent = [t for t in _failures.get(email, []) if now - t < FAILURE_WINDOW_S]
        _failures[email] = recent
        if len(recent) >= MAX_FAILURES:
            raise HTTPException(429, "Terlalu banyak percobaan masuk. Coba lagi dalam 15 menit.")
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT id, participant_id, name, email, password_hash FROM users WHERE email = %s",
                (email,),
            )
            user = cur.fetchone()
            ok = verify_password(body.password, user["password_hash"] if user else _DUMMY_HASH)
            if not (user and ok):
                with _failures_lock:
                    _failures.setdefault(email, []).append(now)
                raise HTTPException(401, "Email atau kata sandi salah.")
            cur.execute("UPDATE users SET last_login_at = NOW() WHERE id = %s", (user["id"],))
        _start_session(response, request, user["id"])
    except pymysql.err.OperationalError:
        raise _db_unavailable()
    with _failures_lock:
        _failures.pop(email, None)
    return {"user": _public(user)}


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> Response:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        try:
            with db.cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE token_hash = %s", (_token_hash(token),))
        except pymysql.err.OperationalError:
            raise _db_unavailable()
    response.delete_cookie(COOKIE_NAME, path="/")
    response.status_code = 204
    return response


@router.get("/me")
def me(request: Request) -> dict:
    return {"user": _public(require_user(request))}
