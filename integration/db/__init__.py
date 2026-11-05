"""MariaDB/MySQL access for the GlucoSight web app (accounts and sessions).

Connection settings come from the environment and are read on every connect,
so tests can point at a throwaway database. Defaults match a stock XAMPP
install (root, no password, 127.0.0.1:3306).

    GLUCOSIGHT_DB_HOST      127.0.0.1
    GLUCOSIGHT_DB_PORT      3306
    GLUCOSIGHT_DB_USER      root
    GLUCOSIGHT_DB_PASSWORD  (empty)
    GLUCOSIGHT_DB_NAME      glucosight
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pymysql
from pymysql.cursors import DictCursor

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def db_name() -> str:
    return os.environ.get("GLUCOSIGHT_DB_NAME", "glucosight")


def connect(*, with_database: bool = True) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.environ.get("GLUCOSIGHT_DB_HOST", "127.0.0.1"),
        port=int(os.environ.get("GLUCOSIGHT_DB_PORT", "3306")),
        user=os.environ.get("GLUCOSIGHT_DB_USER", "root"),
        password=os.environ.get("GLUCOSIGHT_DB_PASSWORD", ""),
        database=db_name() if with_database else None,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
        connect_timeout=5,
    )


@contextmanager
def cursor() -> Iterator[DictCursor]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            yield cur
    finally:
        conn.close()


def init_schema() -> None:
    """Create the database if needed, then apply schema.sql. Safe to re-run."""
    name = db_name()
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"unsafe database name: {name!r}")

    conn = connect(with_database=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        conn.close()

    sql = "\n".join(
        line for line in SCHEMA_PATH.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    )
    with cursor() as cur:
        for statement in filter(None, (s.strip() for s in sql.split(";"))):
            cur.execute(statement)
