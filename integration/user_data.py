"""Per-account profile and prediction history, stored in the `glucosight` database.

    GET  /user/profile    the signed-in user's profile (+ whether it was ever saved)
    PUT  /user/profile    save it
    GET  /user/history    prediction history, newest first
    POST /user/history    add one prediction (idempotent on its id)

Field names are camelCase to match the web app's Profile / HistoryRecord types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pymysql
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from integration import db
from integration.auth import _db_unavailable, require_user

router = APIRouter(prefix="/user", tags=["user"])

HISTORY_LIMIT = 500


class ProfileBody(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    age: str = Field(default="", max_length=3)
    gender: str = Field(default="Pria", max_length=20)
    heightCm: str = Field(default="", max_length=6)
    weightKg: str = Field(default="", max_length=6)
    diabetesType: str = Field(default="Tipe 2", max_length=30)
    defaultSource: str = Field(default="meter", pattern="^(meter|cgm|ppg)$")


class HistoryBody(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    createdAt: str
    food: str = Field(max_length=255)
    portion: str = Field(max_length=50)
    t60: float
    t120: float
    range: str = Field(pattern="^(normal|borderline|high)$")
    g0: float
    sourceId: str = Field(pattern="^(meter|cgm|ppg)$")
    basis: str
    usageMessage: str
    modelVersion: str = Field(max_length=100)


def _number(text: str, label: str, low: float, high: float) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise HTTPException(422, f"{label} harus berupa angka.")
    if not low <= value <= high:
        raise HTTPException(422, f"{label} harus antara {low:g} dan {high:g}.")
    return value


def _text(value) -> str:
    return "" if value is None else f"{float(value):g}"  # DECIMAL columns come back as Decimal('62.0')


def _profile_json(user: dict, row: Optional[dict]) -> dict:
    row = row or {}
    return {
        "participantId": user["participant_id"],
        "name": user["name"],
        "email": user["email"],
        "age": _text(row.get("age")),
        "gender": row.get("gender", "Pria"),
        "heightCm": _text(row.get("height_cm")),
        "weightKg": _text(row.get("weight_kg")),
        "diabetesType": row.get("diabetes_type", "Tipe 2"),
        "defaultSource": row.get("default_source", "meter"),
    }


def _history_json(row: dict) -> dict:
    created = row["created_at"].replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds")
    return {
        "id": row["id"],
        "createdAt": created.replace("+00:00", "Z"),
        "food": row["food"],
        "portion": row["portion"],
        "t60": row["t60"],
        "t120": row["t120"],
        "range": row["glucose_range"],
        "g0": row["g0"],
        "sourceId": row["source_id"],
        "basis": row["basis"],
        "usageMessage": row["usage_message"],
        "modelVersion": row["model_version"],
    }


@router.get("/profile")
def get_profile(request: Request) -> dict:
    user = require_user(request)
    try:
        with db.cursor() as cur:
            cur.execute("SELECT * FROM profiles WHERE user_id = %s", (user["id"],))
            row = cur.fetchone()
    except pymysql.err.OperationalError:
        raise _db_unavailable()
    return {"profile": _profile_json(user, row), "saved": row is not None}


@router.put("/profile")
def put_profile(body: ProfileBody, request: Request) -> dict:
    user = require_user(request)
    age = _number(body.age, "Usia", 0, 120)
    height = _number(body.heightCm, "Tinggi badan", 50, 260)
    weight = _number(body.weightKg, "Berat badan", 10, 500)
    name = body.name.strip()
    try:
        with db.cursor() as cur:
            cur.execute("UPDATE users SET name = %s WHERE id = %s", (name, user["id"]))
            cur.execute(
                "INSERT INTO profiles (user_id, age, gender, height_cm, weight_kg, diabetes_type, default_source) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE age = VALUES(age), gender = VALUES(gender), "
                "height_cm = VALUES(height_cm), weight_kg = VALUES(weight_kg), "
                "diabetes_type = VALUES(diabetes_type), default_source = VALUES(default_source)",
                (user["id"], None if age is None else int(age), body.gender, height, weight,
                 body.diabetesType, body.defaultSource),
            )
            cur.execute("SELECT * FROM profiles WHERE user_id = %s", (user["id"],))
            row = cur.fetchone()
    except pymysql.err.OperationalError:
        raise _db_unavailable()
    return {"profile": _profile_json({**user, "name": name}, row), "saved": True}


@router.get("/history")
def get_history(request: Request) -> dict:
    user = require_user(request)
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT * FROM predictions WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                (user["id"], HISTORY_LIMIT),
            )
            rows = cur.fetchall()
    except pymysql.err.OperationalError:
        raise _db_unavailable()
    return {"history": [_history_json(r) for r in rows]}


@router.post("/history", status_code=201)
def add_history(body: HistoryBody, request: Request) -> dict:
    user = require_user(request)
    try:
        created = datetime.fromisoformat(body.createdAt.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, "Format waktu tidak valid.")
    if created.tzinfo is not None:
        created = created.astimezone(timezone.utc).replace(tzinfo=None)
    try:
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (user_id, id, created_at, food, portion, t60, t120, glucose_range, "
                "g0, source_id, basis, usage_message, model_version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE id = id",
                (user["id"], body.id, created, body.food, body.portion, body.t60, body.t120, body.range,
                 body.g0, body.sourceId, body.basis, body.usageMessage, body.modelVersion),
            )
    except pymysql.err.OperationalError:
        raise _db_unavailable()
    return {"saved": True}
