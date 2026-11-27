"""Connection settings: JawsDB URL on Heroku, GLUCOSIGHT_DB_* overrides everywhere."""

from integration import db

_VARS = ["JAWSDB_URL"] + [f"GLUCOSIGHT_DB_{k}" for k in ("HOST", "PORT", "USER", "PASSWORD", "NAME")]


def _clean(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)


def test_jawsdb_url_supplies_every_setting(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("JAWSDB_URL", "mysql://u1:p%40ss@db.example.com:3307/abc123")
    assert db._settings() == {"host": "db.example.com", "port": "3307", "user": "u1",
                              "password": "p@ss", "name": "abc123"}


def test_explicit_name_wins_over_url(monkeypatch):
    # The DB tests set GLUCOSIGHT_DB_NAME=glucosight_test and DROP it afterwards; that must
    # never resolve to the production database named in the URL.
    _clean(monkeypatch)
    monkeypatch.setenv("JAWSDB_URL", "mysql://u1:pw@db.example.com/prod")
    monkeypatch.setenv("GLUCOSIGHT_DB_NAME", "glucosight_test")
    assert db.db_name() == "glucosight_test"


def test_defaults_without_url(monkeypatch):
    _clean(monkeypatch)
    assert db._settings() == {"host": "127.0.0.1", "port": "3306", "user": "root",
                              "password": "", "name": "glucosight"}
